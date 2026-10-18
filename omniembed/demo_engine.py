"""One local demo worker, remote-only embedding, auditable exact retrieval.

This module never imports omniembed.encoder or loads a local model. RemoteEncoder
is constructed lazily with explicit endpoint/key file paths, with no network in
construction. The one shared encoder lock also covers searches and health probes.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image, ImageOps

from .demo_store import DemoStore, MODEL, REVISION, DIM, now

PREPROCESSING = {'cover':'cover-original-exif-rgb-v1',
                'visual':'video-midpoint8-long512-jpeg90-native-v1',
                'text':'keyword-top-keywords-document-truncate8192-v1'}
TERMINAL = {'completed','partial','failed','cancelled','paused'}


class DemoError(ValueError):
    def __init__(self,message,details=None):
        super().__init__(message)
        self.details=details or {}


class _ControlledStop(Exception):
    def __init__(self, status):
        self.status = status


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(data).hexdigest()


def hash_file(path, checkpoint=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            if checkpoint:
                checkpoint()
            h.update(chunk)
    return h.hexdigest()


def keyword_text(asset):
    values = [asset.get('keyword_top',''), *asset.get('keywords',[])]
    unique = []
    for value in values:
        if isinstance(value,str) and value.strip() and value.strip() not in unique:
            unique.append(value.strip())
    original = '\n'.join(unique)
    return original[:8192], {'source':'keyword_top + keywords; not a human title',
                            'original_characters':len(original),'encoded_characters':min(len(original),8192),
                            'truncated':len(original)>8192}


def tokens(text):
    """Deterministic Chinese unigrams/bigrams plus Latin words; no model needed."""
    result = []
    for part in re.findall(r'[\u3400-\u9fff]+|[a-z0-9]+', text.lower()):
        if re.fullmatch(r'[\u3400-\u9fff]+', part):
            result.extend(part)
            result.extend(part[i:i+2] for i in range(len(part)-1))
        else:
            result.append(part)
    return result


def bm25(query, records):
    docs = {a['id']:tokens(keyword_text(a)[0]) for a in records}
    docs = {k:v for k,v in docs.items() if v}
    if not docs:
        return {}, 0
    frequency = Counter(t for d in docs.values() for t in set(d))
    average = sum(map(len, docs.values()))/len(docs)
    result = {}
    for ident, doc in docs.items():
        tf = Counter(doc); score = 0.
        for term in set(tokens(query)):
            if not tf[term]:
                continue
            idf = math.log(1+(len(docs)-frequency[term]+.5)/(frequency[term]+.5))
            score += idf*tf[term]*2.2/(tf[term]+1.2*(.25+.75*len(doc)/average))
        if score>0:
            result[ident] = score
    return result, len(docs)


class DemoEngine:
    def __init__(self, data_dir='data/local-demo', *, endpoint_file='data/autodl/endpoint.json',
                 key_file='data/autodl/inference.key', encoder=None):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = DemoStore(self.data_dir/'demo.sqlite3')
        self.endpoint_file, self.key_file = Path(endpoint_file), Path(key_file)
        if encoder is None:
            from .remote_encoder import RemoteEncoder
            encoder = RemoteEncoder(model_id=MODEL, revision=REVISION,
                                    endpoint_file=self.endpoint_file,key_file=self.key_file)
            # Explicit file selection wins over a legacy OMNI_INFERENCE_URL/.env value.
            encoder._url=None
        self.enc = encoder
        self.enc_lock = threading.RLock()
        self._health = None
        self._stop = threading.Event()
        self._queue = queue.Queue()
        self._thread = None
        self._life_lock = threading.RLock()
        self._worker_file = None

    def start(self):
        with self._life_lock:
            if self._thread and self._thread.is_alive():
                return
            import fcntl
            handle = (self.data_dir/'.worker.lock').open('a+')
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise DemoError('另一个DEMO进程持有任务锁；请只启动一个worker。') from None
            self._worker_file = handle
            self.store.recover_interrupted_jobs()
            self._stop.clear()
            self._queue = queue.Queue()
            for job in reversed(self.store.jobs(10000)):
                if job['status']=='queued':
                    self._queue.put(job['id'])
            self._thread = threading.Thread(target=self._worker,name='omni-demo-worker',daemon=True)
            self._thread.start()

    def stop(self, timeout=5):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        return {'stopped':not (self._thread and self._thread.is_alive()),
                'note':'当前远端请求无法强行取消；完成后保存断点并暂停。'}

    def _submit(self, kind, params):
        self.start()
        job = self.store.create_job(kind, params)
        self._queue.put(job['id'])
        return job

    def preview(self, path):
        from .demo_import import preview
        return preview(Path(path))

    def submit_import(self,path,name=None,auto_embed=False):
        return self._submit('import',{'path':str(Path(path).resolve()),'name':name,'auto_embed':bool(auto_embed)})

    def submit_embed(self,dataset_id=None,force=False,asset_id=None):
        if dataset_id and not self.store.dataset(dataset_id):
            raise DemoError('数据集不存在。')
        if asset_id and not self.store.asset(asset_id):
            raise DemoError('素材不存在。')
        return self._submit('embed',{'dataset_id':dataset_id,'force':bool(force),'asset_id':asset_id})

    def jobs(self):
        return self.store.jobs()

    def job(self, ident):
        result=self.store.job(ident)
        if result is not None:
            result.update(self.store.job_observations(ident,200))
        return result

    def _control(self,ident,mode):
        job = self.store.job(ident)
        if not job:
            raise DemoError('任务不存在。')
        if job['status'] in TERMINAL:
            return job
        fields={'control':mode,'status':('paused' if mode=='pause' else 'cancelled') if job['status']=='queued' else ('pausing' if mode=='pause' else 'cancelling')}
        if job['status']=='queued':
            fields['finished_at']=now()
        return self.store.update_job(ident,**fields)

    def cancel(self,ident):
        return self._control(ident,'cancel')

    def pause(self,ident):
        return self._control(ident,'pause')

    def resume(self,ident):
        job = self.store.job(ident)
        if not job or job['status'] not in TERMINAL:
            raise DemoError('只能续跑已暂停、取消或结束的任务。')
        return self._submit(job['kind'],job['params'])

    retry = resume

    def _checkpoint(self,ident):
        job = self.store.job(ident)
        if self._stop.is_set() or job.get('control')=='pause':
            raise _ControlledStop('paused')
        if job.get('control')=='cancel':
            raise _ControlledStop('cancelled')

    def _worker(self):
        try:
            while not self._stop.is_set():
                try:
                    ident = self._queue.get(timeout=.25)
                except queue.Empty:
                    continue
                try:
                    job = self.store.job(ident)
                    if not job or job['status']!='queued':
                        continue
                    self.store.update_job(ident,status='running',started_at=now())
                    if job['kind']=='import':
                        self._import_job(ident,job['params'])
                    else:
                        self._embed_job(ident,job['params'])
                except _ControlledStop as error:
                    self.store.update_job(ident,status=error.status,finished_at=now(),control=None)
                except InterruptedError:
                    control = self.store.job(ident).get('control')
                    self.store.update_job(ident,status='paused' if self._stop.is_set() or control=='pause' else 'cancelled',finished_at=now(),control=None)
                except Exception as error:
                    self.store.update_job(ident,status='failed',finished_at=now(),error=self._error(error))
                finally:
                    self._queue.task_done()
        finally:
            with self._life_lock:
                if self._worker_file:
                    self._worker_file.close();self._worker_file=None

    @staticmethod
    def _error(error):
        return {'class':type(error).__name__,
                'message':str(error) if isinstance(error,DemoError) else '处理失败；未生成替代向量，请查看阶段与输入记录后重试。'}

    def _import_job(self,ident,params):
        from .demo_import import materialize
        trace = self.store.new_trace('import',job_id=ident)
        beginning = time.perf_counter();events=[]
        def progress(value):
            self._checkpoint(ident)
            self.store.update_job(ident,progress=value)
            events.append({**value,'stage':value.get('phase','import'),'status':'running'})
        try:
            result = materialize(Path(params['path']),self.data_dir/'imports',progress=progress,
                                 cancelled=lambda:self._stop.is_set() or bool(self.store.job(ident).get('control')))
            self._checkpoint(ident)
            dataset = result['dataset']
            if params.get('name'):
                dataset['name']=params['name']
            saved = self.store.import_dataset(dataset,result['assets'])
            outcome = {'dataset_id':dataset['id'],'asset_count':len(result['assets'])}
            if params.get('auto_embed'):
                outcome['embed_job_id']=self.submit_embed(dataset['id'])['id']
            self.store.update_job(ident,status='completed',finished_at=now(),result=outcome,
                                  progress={'phase':'complete','completed':len(result['assets']),'total':len(result['assets']),'message':'导入完成'})
            self.store.finish_trace(trace,status='completed',elapsed_ms=(time.perf_counter()-beginning)*1000,summary=outcome,events=events)
        except BaseException as error:
            self.store.finish_trace(trace,status='cancelled' if isinstance(error,(_ControlledStop,InterruptedError)) else 'failed',
                                    elapsed_ms=(time.perf_counter()-beginning)*1000,summary={'error':self._error(error)},events=events)
            raise

    def _embed_job(self,ident,params):
        selected = [self.store.asset(params['asset_id'])] if params.get('asset_id') else self.store.all_assets(params.get('dataset_id'))
        counts = {'completed':0,'failed':0,'total':len(selected)}
        # Light ready observation supplies truthful remote inference precision metadata.
        health=self.health()
        if not health.get('ready'):
            raise DemoError('远程模型未就绪；未启用任何本机模型或占位向量。')
        for index,asset in enumerate(selected):
            self._checkpoint(ident)
            if not asset:
                continue
            self.store.update_job(ident,progress={'phase':'embedding','completed':index,'total':len(selected),'asset_id':asset['id'],'message':'逐阶段编码素材'})
            good = self._embed_asset(asset,ident,params.get('force',False))
            counts['completed' if good else 'failed']+=1
        self.store.update_job(ident,status='partial' if counts['failed'] else 'completed',finished_at=now(),result=counts,
                              progress={'phase':'complete','completed':len(selected),'total':len(selected),'message':'编码结束；错误阶段可单独重试'})

    def _fingerprint(self,kind,input_hash):
        return digest({'model':MODEL,'revision':REVISION,'dimensions':DIM,'kind':kind,
                       'input_hash':input_hash,'preprocessing':PREPROCESSING[kind],
                       'storage_dtype':'float32','inference_dtype':(self._health or {}).get('dtype','unknown_remote')})

    def _remote(self,method,*args,**kwargs):
        queued = time.perf_counter()
        with self.enc_lock:
            acquired=time.perf_counter()
            if getattr(self.enc,'model_id',MODEL)!=MODEL or getattr(self.enc,'revision',REVISION)!=REVISION:
                raise DemoError('编码器模型或revision不匹配，拒绝混入索引。')
            raw = getattr(self.enc,method)(*args,**kwargs)
            array=np.asarray(raw,dtype=np.float32)
            if array.shape!=(1,DIM) or not np.isfinite(array).all():
                raise DemoError('远程向量形状或有限值校验失败。')
            norm=float(np.linalg.norm(array[0]))
            if not math.isfinite(norm) or abs(norm-1)>.05:
                raise DemoError('远程向量不是有效单位向量。')
            boot=getattr(self.enc,'boot_id',None)
            if not self._health or self._health.get('boot_id')!=boot:
                self._health_unlocked(False)
            inference_dtype=(self._health or {}).get('dtype','unknown_remote')
            identity={'model':MODEL,'revision':REVISION,'dimensions':DIM,'boot_id':boot,
                      'inference_dtype':inference_dtype,'storage_dtype':'float32','device':'remote',
                      'original_norm':norm,'queue_ms':(acquired-queued)*1000,
                      'remote_ms':(time.perf_counter()-acquired)*1000}
            return array[0]/norm,identity

    def _stage(self,asset,job,trace,stage,events,operation,*,input_hash=None):
        self._checkpoint(job)
        started=time.perf_counter()
        self.store.set_stage(asset['id'],stage,'running',job_id=job,trace_id=trace,input_hash=input_hash)
        try:
            result,details=operation()
            elapsed=(time.perf_counter()-started)*1000
            event={'stage':stage,'status':'completed','elapsed_ms':elapsed,'input_hash':input_hash,'details':details}
            events.append(event)
            self.store.set_stage(asset['id'],stage,'completed',job_id=job,trace_id=trace,input_hash=input_hash,elapsed_ms=elapsed,details=details)
            return result
        except (_ControlledStop,InterruptedError) as error:
            elapsed=(time.perf_counter()-started)*1000
            state=error.status if isinstance(error,_ControlledStop) else 'cancelled'
            events.append({'stage':stage,'status':state,'elapsed_ms':elapsed,'input_hash':input_hash,'details':{'resumable':True}})
            self.store.set_stage(asset['id'],stage,state,job_id=job,trace_id=trace,input_hash=input_hash,elapsed_ms=elapsed,details={'resumable':True})
            raise
        except Exception as error:
            elapsed=(time.perf_counter()-started)*1000
            info=self._error(error)
            details=error.details if isinstance(error,DemoError) else {}
            events.append({'stage':stage,'status':'failed','elapsed_ms':elapsed,'input_hash':input_hash,'error':info,'details':details})
            self.store.set_stage(asset['id'],stage,'failed',job_id=job,trace_id=trace,input_hash=input_hash,elapsed_ms=elapsed,error=info,details=details)
            return None

    def _vector_stage(self,asset,kind,input_hash,job,trace,events,encode_call,force,extra=None):
        fingerprint=self._fingerprint(kind,input_hash)
        cached=self.store.vector(asset['id'],kind,include_values=False,fingerprint=fingerprint)
        if cached and not force:
            details={'skipped':True,'reason':'same input/model/revision/preprocessing/precision fingerprint','fingerprint':fingerprint,'vector':cached}
            self.store.set_stage(asset['id'],kind+'_embedding','completed',job_id=job,trace_id=trace,input_hash=input_hash,elapsed_ms=0,details=details)
            events.append({'stage':kind+'_embedding','status':'skipped','elapsed_ms':0,'input_hash':input_hash,'details':details})
            return cached
        def perform():
            vector,identity=encode_call()
            # Precision can only be confirmed after a new service boot is observed.
            actual_fp=self._fingerprint(kind,input_hash)
            beginning=time.perf_counter()
            saved=self.store.save_vector(asset['id'],kind,vector,input_hash=input_hash,
                 preprocessing=PREPROCESSING[kind],fingerprint=actual_fp,boot_id=identity['boot_id'],
                 inference_dtype=identity['inference_dtype'],device=identity['device'],stats={'original_norm':identity['original_norm']})
            return saved,{'identity':identity,'fingerprint':actual_fp,'vector_stats':saved['stats'],
                          'db_write_ms':(time.perf_counter()-beginning)*1000,**(extra or {})}
        return self._stage(asset,job,trace,kind+'_embedding',events,perform,input_hash=input_hash)

    def _embed_asset(self,asset,job,force):
        trace=self.store.new_trace('embed_asset',asset_id=asset['id'],job_id=job)
        beginning=time.perf_counter();events=[]
        self.store.asset_status(asset['id'],'processing')
        try:
            def verify_sources():
                checked={};mismatches=[]
                for field,role in (('cover','cover'),('video','visual')):
                    path=asset.get(field+'_path');declared=asset.get(field+'_sha256')
                    if not path:
                        continue
                    if not Path(path).is_file():
                        self.store.invalidate_vectors(asset['id'],[role])
                        mismatches.append(field)
                        checked[field]={'exists':False,'declared_sha256':declared}
                        continue
                    actual=hash_file(path,lambda:self._checkpoint(job))
                    checked[field]={'exists':True,'declared_sha256':declared,'actual_sha256':actual,'matches':not declared or actual==declared}
                    if declared and actual!=declared:
                        self.store.invalidate_vectors(asset['id'],[role])
                        mismatches.append(field)
                    else:
                        asset[field+'_sha256']=actual
                if mismatches:
                    raise DemoError('受管理的媒体副本已缺失或被改动（'+','.join(mismatches)+'）；已停用受影响向量，请从原始数据包重新导入修复。',details={'managed_copy_hashes':checked,'mismatched_media':mismatches})
                return checked,{'managed_copy_hashes':checked,'immutable_source_policy':True}
            verified=self._stage(asset,job,trace,'source_integrity',events,verify_sources,
                                 input_hash=digest({'cover':asset.get('cover_sha256'),'video':asset.get('video_sha256')}))
            if verified is None:
                self.store.asset_status(asset['id'],'failed')
                self.store.finish_trace(trace,status='failed',elapsed_ms=(time.perf_counter()-beginning)*1000,
                                        summary={'reason':'source_integrity_failed','resumable_after_reimport':True},events=events)
                return False
            if asset.get('cover_path') and Path(asset['cover_path']).is_file():
                content_hash=asset.get('cover_sha256') or hash_file(asset['cover_path'])
                def cover():
                    with Image.open(asset['cover_path']) as image:
                        if image.width*image.height>16_000_000 or getattr(image,'is_animated',False):
                            raise DemoError('封面超出静态图片像素限制，未静默缩放原图。')
                        with ImageOps.exif_transpose(image).convert('RGB') as rgb:
                            return self._remote('images',[rgb],bs=1)
                self._vector_stage(asset,'cover',content_hash,job,trace,events,cover,force,
                                   {'source':'original cover; EXIF orientation applied; no resize'})
            else:
                self.store.set_stage(asset['id'],'cover_embedding','unavailable',job_id=job,trace_id=trace,details={'reason':'cover_missing'})
            self._checkpoint(job)
            if asset.get('video_path') and Path(asset['video_path']).is_file():
                raw_hash=asset.get('video_sha256') or hash_file(asset['video_path'])
                sampling_hash=digest({'video_sha256':raw_hash,'preprocessing':PREPROCESSING['visual']})
                frames=self._stage(asset,job,trace,'frame_sampling',events,
                                   lambda:self._frames(asset,sampling_hash,job),input_hash=sampling_hash)
                if frames:
                    frame_rows,arrays,details=frames
                    input_hash=digest({'sampling_input_hash':sampling_hash,'frames':[r['sha256'] for r in frame_rows]})
                    self._vector_stage(asset,'visual',input_hash,job,trace,events,
                         lambda:self._remote('videos',[arrays],bs=1),force,{'frame_sampling':details,'frame_count':len(frame_rows),'native_video':True})
            else:
                self.store.set_stage(asset['id'],'frame_sampling','unavailable',job_id=job,trace_id=trace,details={'reason':'video_missing; cover remains searchable'})
                self.store.set_stage(asset['id'],'visual_embedding','unavailable',job_id=job,trace_id=trace,details={'reason':'no video vector fabricated'})
            self._checkpoint(job)
            text,origin=keyword_text(asset)
            if text.strip():
                input_hash=digest({'text':text,'source':origin['source']})
                self._vector_stage(asset,'text',input_hash,job,trace,events,
                    lambda:self._remote('docs',[''],[text],bs=1),force,origin)
            else:
                self.store.set_stage(asset['id'],'text_embedding','unavailable',job_id=job,trace_id=trace,details={'reason':'keyword_top_and_keywords_empty'})
            vectors=[k for k in PREPROCESSING if self.store.vector(asset['id'],k,include_values=False)]
            failed=any(e['status']=='failed' for e in events)
            state='partial' if failed and vectors else ('failed' if not vectors else 'ready')
            self.store.asset_status(asset['id'],state)
            self.store.finish_trace(trace,status=state,elapsed_ms=(time.perf_counter()-beginning)*1000,
                 summary={'asset_id':asset['id'],'vector_kinds':vectors,'failed_stages':[e['stage'] for e in events if e['status']=='failed']},events=events)
            return not failed and bool(vectors)
        except _ControlledStop as error:
            self.store.asset_status(asset['id'],'paused')
            self.store.finish_trace(trace,status=error.status,elapsed_ms=(time.perf_counter()-beginning)*1000,summary={'resumable':True},events=events)
            raise
        except Exception as error:
            self.store.asset_status(asset['id'],'failed')
            self.store.finish_trace(trace,status='failed',elapsed_ms=(time.perf_counter()-beginning)*1000,summary={'error':self._error(error)},events=events)
            return False

    def _frames(self,asset,input_hash,job):
        existing=self.store.artifacts(asset['id'],'frame')
        matched=[a for a in existing if a['input_hash']==input_hash]
        latest={a['metadata'].get('index'):a for a in matched}
        expected_count=len(next(iter(latest.values()))['metadata'].get('sampling',{}).get('requested_indices',[])) if latest else 0
        if latest and expected_count and len(latest)==expected_count and set(latest)==set(range(expected_count)) and all(Path(a['path']).is_file() and hash_file(a['path'])==a['metadata'].get('sha256') for a in latest.values()):
            rows=[{**latest[i]['metadata'],'path':latest[i]['path']} for i in sorted(latest)]
            pixels=[]
            for row in rows:
                with Image.open(row['path']) as image:
                    pixels.append(np.asarray(image.convert('RGB'),dtype=np.uint8))
            arrays=np.stack(pixels)
            return (rows,arrays,{'reused':True,**rows[0].get('sampling',{})}),{'reused':True,'frame_count':len(rows)}
        import cv2
        # Restrict FFmpeg container protocols: local imported bytes must not fetch external media.
        previous_options=os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='protocol_whitelist;file,pipe'
        try:
            capture=cv2.VideoCapture(str(asset['video_path']),cv2.CAP_FFMPEG)
        finally:
            if previous_options is None:
                os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS',None)
            else:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']=previous_options
        if not capture.isOpened():
            raise DemoError('视频无法由本机OpenCV解码；封面和关键词仍可单独编码。')
        rows=[];arrays=[]
        try:
            count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(capture.get(cv2.CAP_PROP_FPS))
            if count<=0 or not math.isfinite(fps) or fps<=0:
                raise DemoError('视频帧数或名义FPS不可用；未伪造时间线。')
            n=min(8,count)
            indices=sorted({min(count-1,int((i+.5)*count/n)) for i in range(n)})
            directory=self.data_dir/'artifacts'/asset['id']/input_hash
            directory.mkdir(parents=True,exist_ok=True)
            sampling={'policy':PREPROCESSING['visual'],'original_frame_count':count,'nominal_fps':fps,
                      'max_frames':8,'requested_indices':indices,'long_edge':512,'jpeg_quality':90,
                      'timing_note':'frame_index / nominal FPS; approximate for VFR and decoder seeking, not exact timestamps',
                      'remote_input':'RGB pixels decoded from these saved JPEG artifacts; transport re-encodes losslessly as PNG'}
            for index,source_index in enumerate(indices):
                self._checkpoint(job)
                capture.set(cv2.CAP_PROP_POS_FRAMES,source_index)
                ok,frame=capture.read()
                if not ok or frame is None:
                    raise DemoError('选定视频帧解码失败；未复制或补造帧。')
                h,w=frame.shape[:2];scale=min(1.,512/max(h,w))
                if scale<1:
                    frame=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                path=directory/f'frame-{index:02d}.jpg'
                with Image.fromarray(rgb) as image:
                    image.save(path,format='JPEG',quality=90)
                with Image.open(path) as image:
                    arrays.append(np.asarray(image.convert('RGB'),dtype=np.uint8))
                record={'index':index,'source_frame_index':source_index,'timestamp_s':source_index/fps,
                        'nominal_timestamp_s':source_index/fps,'decoder_position_ms':float(capture.get(cv2.CAP_PROP_POS_MSEC)),
                        'path':str(path),'sha256':hash_file(path),'width':arrays[-1].shape[1],'height':arrays[-1].shape[0],
                        'sampling':sampling}
                self.store.artifact(asset['id'],'frame',path,input_hash,PREPROCESSING['visual'],record)
                rows.append(record)
            data=np.stack(arrays)
            return (rows,data,sampling),{'frame_count':len(rows),'sampling':sampling,'frames':rows}
        finally:
            capture.release()

    def _card(self,asset):
        if not asset:
            return None
        result=dict(asset)
        result['asset_id']=asset['id']
        result['cover_url']=f"/api/demo/media/{asset['id']}/cover" if asset.get('cover_path') else None
        result['video_url']=f"/api/demo/media/{asset['id']}/video" if asset.get('video_path') else None
        result['vector_kinds']=[k for k in PREPROCESSING if self.store.vector(asset['id'],k,include_values=False)]
        result['has_cover']=bool(asset.get('cover_path') and Path(asset['cover_path']).is_file())
        result['has_video']=bool(asset.get('video_path') and Path(asset['video_path']).is_file())
        return result

    def assets(self,dataset_id=None,q='',status='',limit=100,offset=0):
        limit=max(1,min(int(limit),500));offset=max(0,int(offset))
        result=self.store.assets(dataset_id,str(q)[:500],str(status)[:50],limit,offset)
        result['items']=[self._card(a) for a in result['items']]
        return result

    def asset(self,ident):
        result=self._card(self.store.asset(ident))
        if not result:
            return None
        result['stages']=self.store.stages(ident)
        current=next((s['input_hash'] for s in result['stages'] if s['stage']=='frame_sampling'),None)
        frames=[a for a in self.store.artifacts(ident,'frame') if a['input_hash']==current]
        latest={a['metadata']['index']:a for a in frames}
        result['frames']=[{**a['metadata'],'path':a['path'],'url':f"/api/demo/media/{ident}/frame/{i}"} for i,a in sorted(latest.items())]
        result['vectors']={k:self.store.vector(ident,k,include_values=True) for k in PREPROCESSING}
        result['vectorstats']={k:{**v['stats'],'model':v['model'],'revision':v['revision'],'dimensions':v['dimensions'],
                                  'input_hash':v['input_hash'],'fingerprint':v['fingerprint'],'preprocessing':v['preprocessing'],
                                  'storage_dtype':v['storage_dtype'],'inference_dtype':v['inference_dtype'],'boot_id':v['boot_id']}
                               for k,v in result['vectors'].items() if v}
        result['text_source']=keyword_text(result)[1]
        result['traces']=[t for t in self.store.traces(1000) if t.get('asset_id')==ident][:30]
        return result

    def datasets(self):
        return self.store.datasets()

    def overview(self):
        datasets=self.datasets()
        assets=self.store.all_assets()
        return {'datasets':datasets,'dataset_count':len(datasets),'asset_count':len(assets),
                'status_counts':dict(Counter(a['status'] for a in assets)),
                'vector_counts':{k:sum(d['vector_counts'].get(k,0) for d in datasets) for k in PREPROCESSING},
                'warning_count':sum(bool(a.get('warnings') or not a.get('video_path')) for a in assets),
                'failed_count':sum(a['status']=='failed' for a in assets),
                'ready_count':sum(a['status']=='ready' for a in assets),
                'jobs':self.jobs()[:10],'storage':self.store.storage(),
                'model':MODEL,'revision':REVISION,'dimensions':DIM,'backend':'remote',
                'worker_running':bool(self._thread and self._thread.is_alive()),
                'notes':['关键词文本来源于keyword_top/keywords，不冒充人工标题。',
                         'visual只使用真实视频多帧向量；缺视频素材仍可由cover/text/keyword召回。',
                         'DEMO单进程共享一个RemoteEncoder；不能协调其他应用或其他进程的GPU请求。']}

    def _health_unlocked(self,probe=False):
        if hasattr(self.enc,'health'):
            result=self.enc.health(probe=probe)
        else:
            from .remote_encoder import _key
            url=self.enc._endpoint()
            request=urllib.request.Request(url+('/health/inference' if probe else '/health/ready'),
                  headers={'Authorization':'Bearer '+_key(self.enc.key_file),'Accept':'application/json'},
                  method='POST' if probe else 'GET')
            try:
                response=self.enc._opener.open(request,timeout=min(self.enc.timeout,120 if probe else 10))
            except urllib.error.HTTPError as error:
                response=error
            with response:
                status=response.code;raw=response.read(64*1024+1)
            if len(raw)>64*1024:
                raise DemoError('健康响应超过限制。')
            value=json.loads(raw)
            if not isinstance(value,dict):
                raise DemoError('健康响应不是有效对象。')
            result={k:value.get(k) for k in ('status','ready','model','model_revision','boot_id','device','dtype','uptime_s','busy_for_s')}
            result['http_status']=status
            if probe:
                result['probe']={k:value.get('checks',{}).get(k) for k in ('shape','finite','nonzero','unit_norm')} if isinstance(value.get('checks'),dict) else None
        safe={k:result.get(k) for k in ('status','ready','model','model_revision','boot_id','device','dtype','uptime_s','busy_for_s','http_status','probe') if k in result}
        # Restrict labels instead of persisting server messages or error bodies.
        for k in ('status','model','model_revision','boot_id','device','dtype'):
            v=safe.get(k)
            if v is not None and (not isinstance(v,str) or len(v)>256 or not re.fullmatch(r'[A-Za-z0-9_./:-]+',v)):
                safe.pop(k,None)
        safe['configured']=True
        safe['ready']=bool(safe.get('ready') is True and safe.get('status')=='ok' and safe.get('model')==MODEL
                           and safe.get('model_revision')==REVISION)
        safe['busy']=safe.get('http_status')==429
        self._health=safe
        return safe

    def health(self,probe=False):
        started=time.perf_counter()
        try:
            with self.enc_lock:
                result=self._health_unlocked(probe)
                return {**result,'elapsed_ms':(time.perf_counter()-started)*1000,'backend':'remote','local_model_fallback':False}
        except Exception as error:
            return {'configured':self.endpoint_file.is_file() and self.key_file.is_file(),
                    'ready':False,'status':'unavailable','busy':False,'backend':'remote',
                    'local_model_fallback':False,'error':self._error(error),'elapsed_ms':(time.perf_counter()-started)*1000}

    def traces(self):
        return self.store.traces()

    def trace(self,ident):
        return self.store.trace(ident)

    def _gallery(self,kind,dataset_id=None,precision=None):
        ids,matrix,metadata=self.store.gallery(kind,dataset_id)
        selected=[i for i,a in enumerate(ids) if metadata[a]['preprocessing']==PREPROCESSING[kind]
                  and (not precision or metadata[a]['inference_dtype']==precision)]
        chosen=[ids[i] for i in selected]
        return chosen,matrix[selected],{a:metadata[a] for a in chosen}

    @staticmethod
    def _query_stats(vector):
        return {'shape':[DIM],'finite':bool(np.isfinite(vector).all()),'norm':float(np.linalg.norm(vector)),
                'min':float(np.min(vector)),'max':float(np.max(vector))}

    def search(self,query,mode='visual',top_k=20,dataset_id=None):
        if not isinstance(query,str) or not query.strip() or len(query)>8192:
            raise DemoError('查询须为1–8192字符的非空文本。')
        return self._search(query,mode,top_k,dataset_id,None)

    def search_image(self,image,mode='visual',top_k=20,dataset_id=None):
        if mode not in ('visual','cover'):
            raise DemoError('图片查询只支持visual或cover向量。')
        if isinstance(image,Image.Image):
            original=image.copy()
        elif isinstance(image,(bytes,bytearray)):
            original=Image.open(io.BytesIO(image))
        else:
            original=Image.open(Path(image))
        try:
            if original.width*original.height>16_000_000 or getattr(original,'is_animated',False):
                raise DemoError('图片查询需静态且不超过1600万像素。')
            with ImageOps.exif_transpose(original).convert('RGB') as rgb:
                query={'type':'image','width':rgb.width,'height':rgb.height,
                       'sha256':digest(rgb.tobytes())}
                return self._search(query,mode,top_k,dataset_id,rgb)
        finally:
            original.close()

    def _search(self,query,mode,top_k,dataset_id,image):
        if mode not in ('visual','cover','text','keyword','hybrid'):
            raise DemoError('检索模式无效。')
        top_k=max(1,min(int(top_k),100))
        trace=self.store.new_trace('search_image' if image is not None else 'search')
        beginning=time.perf_counter();timings={};events=[]
        try:
            routes={};route_meta={};query_meta=None
            if mode in ('keyword','hybrid'):
                started=time.perf_counter()
                records=self.store.all_assets(dataset_id)
                scores,keyword_candidates=bm25(query,records)
                routes['keyword']=scores
                route_meta['keyword']={'candidate_count':keyword_candidates,'algorithm':'BM25 k1=1.2 b=.75; CJK unigram+bigram/ASCII words'}
                timings['keyword_ms']=(time.perf_counter()-started)*1000
                events.append({'stage':'keyword_bm25','status':'completed','elapsed_ms':timings['keyword_ms'],'details':route_meta['keyword']})
            if mode!='keyword':
                started=time.perf_counter()
                if image is None:
                    vector,identity=self._remote('query',[query],bs=1)
                else:
                    vector,identity=self._remote('images',[image],bs=1)
                timings['remote_query_ms']=(time.perf_counter()-started)*1000
                # Persist only in the explicit trace for reproducible score replay.
                query_meta={**identity,'stats':self._query_stats(vector),
                            'values':vector.astype(float).tolist()}
                events.append({'stage':'remote_query','status':'completed','elapsed_ms':timings['remote_query_ms'],
                               'input_hash':digest(query),'details':{k:v for k,v in query_meta.items() if k!='values'}})
                started=time.perf_counter()
                for kind in (('cover','visual','text') if mode=='hybrid' else (mode,)):
                    ids,matrix,meta=self._gallery(kind,dataset_id,identity['inference_dtype'])
                    scores=matrix@vector
                    routes[kind]={a:float(scores[i]) for i,a in enumerate(ids)}
                    route_meta[kind]={'candidate_count':len(ids),'exact_cosine':True,'model':MODEL,'revision':REVISION,
                                      'dimensions':DIM,'preprocessing':PREPROCESSING[kind],'inference_dtype':identity['inference_dtype']}
                timings['dense_cosine_ms']=(time.perf_counter()-started)*1000
                events.append({'stage':'exact_dense_cosine','status':'completed','elapsed_ms':timings['dense_cosine_ms'],'details':route_meta})
            started=time.perf_counter()
            ranked={kind:sorted(scores,key=lambda a:(-scores[a],a)) for kind,scores in routes.items()}
            ranks={kind:{a:i+1 for i,a in enumerate(order)} for kind,order in ranked.items()}
            candidates=set().union(*(set(scores) for scores in routes.values())) if routes else set()
            if mode=='hybrid':
                # Fixed RRF is exploratory. Score contributions stay visible; no superiority claim.
                totals={a:sum(1/(60+ranks[k][a]) for k in routes if a in ranks[k] and ranks[k][a]<=100) for a in candidates}
                candidates={a for a in candidates if totals[a]>0}
            else:
                totals=routes.get(mode,{})
            order=sorted(candidates,key=lambda a:(-totals[a],a))[:top_k]
            items=[]
            for rank,asset_id in enumerate(order,1):
                card=self._card(self.store.asset(asset_id))
                components={kind:{'score':routes[kind][asset_id],'rank':ranks[kind][asset_id],
                                  'rrf_contribution':1/(60+ranks[kind][asset_id]) if mode=='hybrid' and ranks[kind][asset_id]<=100 else 0}
                            for kind in routes if asset_id in routes[kind]}
                items.append({**card,'score':totals[asset_id],'rank':rank,'components':components,
                              'scores':{k:v['score'] for k,v in components.items()},'ranks':{k:v['rank'] for k,v in components.items()}})
            timings['rank_and_read_ms']=(time.perf_counter()-started)*1000
            timings['total_ms']=(time.perf_counter()-beginning)*1000
            candidate_count=len(candidates) if mode!='keyword' else route_meta['keyword']['candidate_count']
            summary={'query':query,'mode':mode,'candidate_count':candidate_count,'routes':route_meta,
                     'query_vector':query_meta,'rankings':{k:[{'asset_id':a,'rank':i+1,'score':routes[k][a]} for i,a in enumerate(order[:100])] for k,order in ranked.items()},
                     'timings':timings,'top_results':[{'asset_id':r['id'],'score':r['score'],'components':r['components']} for r in items]}
            events.append({'stage':'rank_and_return','status':'completed','elapsed_ms':timings['rank_and_read_ms'],
                           'details':{'algorithm':'RRF k=60, route rank cutoff100' if mode=='hybrid' else 'exact route score descending','returned':len(items)}})
            self.store.save_search(trace,mode,query,dataset_id,candidate_count,timings,summary['top_results'])
            self.store.finish_trace(trace,status='completed',elapsed_ms=timings['total_ms'],summary=summary,events=events)
            return {'trace_id':trace,'items':items,'timings':timings,'mode':mode,'candidate_count':candidate_count,
                    'query':query,'routes':route_meta,'notes':['hybrid采用固定RRF，仅演示融合，不宣称最优。'] if mode=='hybrid' else
                         (['关键词BM25不调用GPU；标签文字来自keyword_top/keywords。'] if mode=='keyword' else ['只使用指定类型的真实向量，不做跨类型静默回退。'])}
        except Exception as error:
            self.store.finish_trace(trace,status='failed',elapsed_ms=(time.perf_counter()-beginning)*1000,
                  summary={'query':query,'mode':mode,'error':self._error(error),'timings':timings},events=events)
            failure=DemoError('检索失败，已保存错误trace（'+trace+'）；未启用本机模型回退。')
            failure.trace_id=trace
            raise failure from None

    def galaxy(self,kind='visual',dataset_id=None):
        if kind not in PREPROCESSING:
            raise DemoError('星图只支持cover、visual或text向量。')
        ids,matrix,metadata=self._gallery(kind,dataset_id)
        available=len(ids)
        # Keep unlike inference precision groups separate even for an offline visualization.
        precision=Counter(metadata[a]['inference_dtype'] for a in ids)
        selected_precision=sorted(precision,key=lambda k:(-precision[k],k))[0] if precision else None
        keep=[i for i,a in enumerate(ids) if metadata[a]['inference_dtype']==selected_precision][:2000]
        ids=[ids[i] for i in keep];matrix=matrix[keep]
        records=self.store.all_assets(dataset_id)
        if not ids:
            return {'nodes':[],'edges':[],'pending_count':len(records),
                    'projection':{'method':'PCA/SVD','kind':kind,'count':0,'available_vector_count':available,
                                  'note':'尚无此类型真实向量；先构建索引，不使用随机占位点。'}}
        centered=np.asarray(matrix,dtype=np.float64)-matrix.mean(axis=0,dtype=np.float64)
        if len(ids)>1 and np.any(centered):
            if len(ids)<DIM:
                _,values,axes=np.linalg.svd(centered,full_matrices=False)
                basis=axes[:2].T;eigen=values**2
            else:
                covariance=centered.T@centered
                eigen,axes=np.linalg.eigh(covariance);order=np.argsort(eigen)[::-1]
                eigen=np.maximum(eigen[order],0);basis=axes[:,order[:2]]
            for column in range(basis.shape[1]):
                pivot=np.argmax(np.abs(basis[:,column]))
                if basis[pivot,column]<0:
                    basis[:,column]*=-1
            points=centered@basis
            ratios=(eigen[:2]/eigen.sum()).tolist() if eigen.sum()>0 else [0.,0.]
        else:
            points=np.zeros((len(ids),2));ratios=[0.,0.]
        if points.shape[1]<2:
            points=np.pad(points,((0,0),(0,2-points.shape[1])))
        nodes=[{**self._card(self.store.asset(a)),'x':float(points[i,0]),'y':float(points[i,1]),'embedding_kind':kind} for i,a in enumerate(ids)]
        edges=[];seen=set()
        if len(ids)>1:
            for start in range(0,len(ids),32):
                scores=matrix[start:start+32]@matrix.T
                for j,row in enumerate(scores):
                    i=start+j;row[i]=-np.inf;k=int(np.argmax(row));pair=tuple(sorted((ids[i],ids[k])))
                    if row[k]>0 and pair not in seen:
                        seen.add(pair);edges.append({'source':pair[0],'target':pair[1],'score':float(row[k])})
        return {'nodes':nodes,'edges':edges,'pending_count':max(0,len(records)-available),
                'projection':{'method':'PCA/SVD','kind':kind,'count':len(ids),'available_vector_count':available,
                  'inference_dtype':selected_precision,'excluded_precision_groups':dict(precision),
                  'explained_variance_ratio':ratios,'sampled':len(ids)<available,
                  'selection':'at most2000 deterministic asset IDs from largest compatible precision group',
                  'edge_policy':'one positive-cosine nearest neighbor per node, deduplicated undirected; original768dimvectors',
                  'model':MODEL,'revision':REVISION,'dimensions':DIM,
                  'note':'二维PCA保留部分方差，不是原始余弦距离真值；点来自真实向量，连线相似度按原始768维计算。'}}
