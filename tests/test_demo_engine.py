"""Independent demo engine fixtures: CPU media, fake remote, no service/key/model access."""
import hashlib
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from PIL import Image
import pytest

from omniembed.demo_engine import DemoEngine, DemoError, PREPROCESSING, hash_file
from omniembed.demo_store import MODEL, REVISION, DIM, DemoStore


class DemoFakeRemote:
    model_id=MODEL
    revision=REVISION
    dtype='remote'
    device='remote'
    boot_id='fake-demo-boot'

    def __init__(self):
        self.calls=[];self.fail=None;self.active=0;self.maximum=0;self.lock=threading.Lock()

    def health(self,probe=False):
        if probe:
            self._run('probe',np.ones(3))
        return {'status':'ok','ready':True,'model':MODEL,'model_revision':REVISION,
                'boot_id':self.boot_id,'dtype':'float32','device':'cpu-test','http_status':200}

    def _run(self,kind,values):
        with self.lock:
            self.calls.append(kind);self.active+=1;self.maximum=max(self.maximum,self.active)
        try:
            time.sleep(.005)
            if self.fail==kind:
                raise RuntimeError('should-not-persist-sensitive-exception-message')
            v=np.zeros((1,DIM),np.float32);v[0,:3]=values
            if not np.any(v):v[0,0]=1
            return v/np.linalg.norm(v,axis=1,keepdims=True)
        finally:
            with self.lock:self.active-=1

    def query(self,texts,bs=1):
        return self._run('query',self._text(texts[0]))
    def docs(self,titles,texts,bs=1):
        assert titles==['']  # Keyword source is not disguised as a human title.
        return self._run('text',self._text(texts[0]))
    def images(self,images,bs=1):
        return self._run('cover',np.asarray(images[0].convert('RGB')).mean(axis=(0,1)))
    def videos(self,arrays,bs=1):
        assert arrays[0].dtype==np.uint8 and arrays[0].shape[0]<=8
        return self._run('visual',arrays[0].mean(axis=(0,1,2)))
    @staticmethod
    def _text(text):
        return [float('red' in text or '红' in text),float('green' in text or '绿' in text),float('blue' in text or '蓝' in text)]


def make_video(path,color):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),6,(960,540))
    assert writer.isOpened()
    frame=np.zeros((540,960,3),np.uint8);frame[:]=color
    for _ in range(12):writer.write(frame)
    writer.release()


@pytest.fixture
def demo(tmp_path):
    enc=DemoFakeRemote();engine=DemoEngine(tmp_path/'local-demo',encoder=enc,
                                         endpoint_file=tmp_path/'does-not-exist.json',key_file=tmp_path/'does-not-exist.key')
    dataset={'id':'ds_test','name':'test','source_sha256':'test','source_path':str(tmp_path),'source_type':'directory','summary':{}}
    assets=[]
    for i,(cover,video,keyword) in enumerate([((255,0,0),(255,0,0),'red'),((0,0,255),(0,0,255),'blue'),((0,255,0),None,'green')]):
        image=tmp_path/f'cover{i}.png';Image.new('RGB',(128,96),cover).save(image)
        vp=tmp_path/f'video{i}.mp4'
        if video is not None:make_video(vp,video)  # BGR: first video blue, second red.
        assets.append({'id':f'asset_{i}','source_id':str(i),'title':f'搜索词：{keyword}',
          'title_source':'search_keyword','keyword_top':keyword,'keywords':[keyword],
          'metadata':{'original':'kept'},'cover_path':str(image),'cover_sha256':hash_file(image),
          'video_path':str(vp) if video else None,'video_sha256':hash_file(vp) if video else None,
          'source_status':{'cover':'present','video':'present' if video else 'missing'},'warnings':[]})
    engine.store.import_dataset(dataset,assets)
    yield engine,enc,assets
    engine.stop()


def encode_all(engine,force=False):
    job=engine.store.create_job('embed',{'dataset_id':'ds_test','force':force})
    engine._embed_job(job['id'],job['params'])
    return engine.store.job(job['id'])


def test_isolated_wal_and_full_provenance(demo):
    engine,enc,assets=demo
    job=encode_all(engine)
    assert job['status']=='completed'
    assert engine.overview()['vector_counts']=={'cover':3,'visual':2,'text':3}
    assert engine.overview()['storage']['journal_mode']=='wal'
    assert engine.overview()['storage']['table_counts']['assets']==3
    detail=engine.asset('asset_0')
    assert len(detail['frames'])==8
    assert all(max(f['width'],f['height'])<=512 and Path(f['path']).is_file() for f in detail['frames'])
    assert detail['frames'][0]['sampling']['timing_note'].startswith('frame_index / nominal FPS')
    for kind in ('cover','visual','text'):
        v=detail['vectors'][kind]
        assert v['model']==MODEL and v['revision']==REVISION and len(v['values'])==DIM
        assert v['storage_dtype']=='float32' and v['inference_dtype']=='float32'
        assert v['boot_id']==enc.boot_id and v['fingerprint'] and v['input_hash']
        assert v['preprocessing']==PREPROCESSING[kind] and abs(v['stats']['norm']-1)<1e-6
    coveronly=engine.asset('asset_2')
    assert coveronly['status']=='ready' and coveronly['vectors']['visual'] is None
    assert any(s['stage']=='visual_embedding' and s['status']=='unavailable' for s in coveronly['stages'])
    assert assets[0]['video_path']==detail['video_path']  # Original video was not recopied.


def test_resume_skips_fingerprints_and_force_recomputes(demo):
    engine,enc,_=demo
    encode_all(engine);calls=len(enc.calls)
    encode_all(engine)
    assert len(enc.calls)==calls
    encode_all(engine,force=True)
    assert len(enc.calls)==calls*2
    assert engine.store.storage()['table_counts']['embeddings']==8


def test_separate_routes_keyword_no_gpu_and_real_pca(demo):
    engine,enc,_=demo;encode_all(engine)
    assert engine.search('red','cover')['items'][0]['id']=='asset_0'
    visual=engine.search('red','visual')
    assert visual['items'][0]['id']=='asset_1' and visual['candidate_count']==2
    assert engine.search('red','text')['items'][0]['id']=='asset_0'
    before=len(enc.calls);keyword=engine.search('red','keyword')
    assert len(enc.calls)==before and keyword['items'][0]['id']=='asset_0'
    fused=engine.search('red','hybrid')
    assert fused['items'][0]['components'] and fused['routes']['keyword']['algorithm'].startswith('BM25')
    trace=engine.trace(fused['trace_id'])
    assert trace['summary']['query']=='red' and trace['summary']['query_vector']['stats']['shape']==[DIM]
    assert trace['events'] and trace['summary']['rankings']
    galaxy=engine.galaxy('cover')
    assert len(galaxy['nodes'])==3 and galaxy['projection']['method']=='PCA/SVD'
    assert galaxy['projection']['explained_variance_ratio'][0]>0
    assert len(engine.galaxy('visual')['nodes'])==2


def test_remote_failure_persists_stage_without_fake_vector(demo):
    engine,enc,_=demo;enc.fail='visual'
    job=encode_all(engine)
    assert job['status']=='partial'
    assert engine.store.vector('asset_0','visual') is None
    assert engine.store.vector('asset_0','cover') is not None
    detail=engine.asset('asset_0')
    assert detail['status']=='partial'
    stage=next(s for s in detail['stages'] if s['stage']=='visual_embedding')
    assert stage['status']=='failed' and stage['error']['class']=='RuntimeError'
    assert 'should-not-persist' not in str(detail)
    enc.fail=None;encode_all(engine)
    assert engine.store.vector('asset_0','visual') is not None


def test_shared_encoder_serializes_search_and_probe(demo):
    engine,enc,_=demo;encode_all(engine)
    threads=[threading.Thread(target=lambda:engine.search('red','cover')) for _ in range(4)]
    threads += [threading.Thread(target=lambda:engine.health(probe=True)) for _ in range(3)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert enc.maximum==1


def test_revision_mismatch_and_invalid_vector_never_stored(demo):
    engine,enc,_=demo;enc.revision='wrong-revision'
    encode_all(engine)
    assert engine.store.storage()['table_counts']['embeddings']==0
    enc.revision=REVISION
    with pytest.raises(ValueError):
        engine.store.save_vector('asset_0','cover',np.zeros(DIM),input_hash='bad',preprocessing='bad',fingerprint='bad')


def test_async_import_cancel_pause_resume_and_restart_checkpoint(demo,monkeypatch):
    engine,enc,assets=demo
    import omniembed.demo_import as importer
    def materialize(path,dest,progress=None,cancelled=None):
        assert not cancelled()
        progress({'phase':'copy','completed':1,'total':1,'message':'test fixture'})
        return {'dataset':{'id':'ds_import','name':'fixture','summary':{},'source_sha256':'fixture'},
                'assets':[{**assets[0],'id':'import_asset'}]}
    monkeypatch.setattr(importer,'materialize',materialize)
    job=engine.submit_import('fake-source',auto_embed=True)
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        current=engine.job(job['id'])
        if current['status'] in ('completed','failed'):break
        time.sleep(.02)
    assert current['status']=='completed'
    nextjob=current['result']['embed_job_id']
    while time.monotonic()<deadline and engine.job(nextjob)['status'] not in ('completed','failed'):
        time.sleep(.02)
    assert engine.job(nextjob)['status']=='completed'
    engine.stop()
    interrupted=engine.store.create_job('embed',{'dataset_id':'ds_test'})
    engine.store.update_job(interrupted['id'],status='running')
    engine.start()
    assert engine.job(interrupted['id'])['status']=='paused'
    resumed=engine.resume(interrupted['id'])
    deadline=time.monotonic()+10
    while time.monotonic()<deadline and engine.job(resumed['id'])['status'] not in ('completed','failed'):
        time.sleep(.02)
    assert engine.job(resumed['id'])['status']=='completed'


def test_partial_frame_artifacts_are_not_reused_as_complete(demo):
    engine,enc,assets=demo
    encode_all(engine)
    frames=engine.asset('asset_0')['frames']
    Path(frames[-1]['path']).unlink()
    encode_all(engine)
    assert len(engine.asset('asset_0')['frames'])==8 and all(Path(f['path']).is_file() for f in engine.asset('asset_0')['frames'])


def test_managed_copy_hash_mismatch_deactivates_affected_vector(demo):
    engine,enc,assets=demo;encode_all(engine)
    Image.new('RGB',(128,96),(0,255,0)).save(assets[0]['cover_path'])
    result=encode_all(engine)
    assert result['status']=='partial'
    assert engine.store.vector('asset_0','cover') is None
    assert engine.store.vector('asset_0','visual') is not None
    detail=engine.asset('asset_0')
    assert detail['status']=='failed'
    stage=next(s for s in detail['stages'] if s['stage']=='source_integrity')
    assert stage['status']=='failed' and '重新导入' in stage['error']['message']
    assert stage['details']['managed_copy_hashes']['cover']['matches'] is False


def test_cooperative_pause_persists_completed_vectors_and_stage_state(demo):
    engine,enc,_=demo
    original=enc.images
    paused=False
    def pause_after_cover(*args,**kwargs):
        nonlocal paused
        result=original(*args,**kwargs)
        if not paused:
            paused=True
            target=next(j for j in engine.jobs() if j['kind']=='embed' and j['status']=='running')
            engine.pause(target['id'])
        return result
    enc.images=pause_after_cover
    job=engine.submit_embed('ds_test');deadline=time.monotonic()+10
    while time.monotonic()<deadline and engine.job(job['id'])['status']!='paused':time.sleep(.01)
    assert engine.job(job['id'])['status']=='paused'
    assert engine.store.vector('asset_0','cover') is not None
    assert engine.store.vector('asset_0','visual') is None
    enc.images=original
    resumed=engine.resume(job['id']);deadline=time.monotonic()+10
    while time.monotonic()<deadline and engine.job(resumed['id'])['status'] not in ('completed','failed'):time.sleep(.02)
    assert engine.job(resumed['id'])['status']=='completed'
    assert enc.calls.count('cover')==3  # Completed first cover was not requested again.


def test_explicit_endpoint_paths_override_legacy_url_without_importing_local_model(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    fake=DemoFakeRemote();fake._url='https://legacy.invalid'
    def factory(**kwargs):
        assert kwargs['model_id']==MODEL and kwargs['revision']==REVISION
        assert kwargs['endpoint_file']==tmp_path/'endpoint.json'
        return fake
    monkeypatch.setitem(sys.modules,'omniembed.remote_encoder',SimpleNamespace(RemoteEncoder=factory))
    engine=DemoEngine(tmp_path/'new-demo',endpoint_file=tmp_path/'endpoint.json',key_file=tmp_path/'key')
    assert engine.enc._url is None
    assert engine.enc is fake


def test_job_detail_contains_bounded_whitebox_without_full_vectors(demo):
    engine,enc,_=demo;job=encode_all(engine)
    detail=engine.job(job['id'])
    assert detail['traces'] and detail['stages'] and detail['asset_stages']
    assert detail['trace_total']==3
    assert detail['asset_stage_total']>=10
    assert detail['observation_limit']==200
    assert all('values' not in t['details'] for t in detail['asset_stages'])
    assert any(s['details']['trace_id'] for s in detail['stages'])


def test_query_trace_values_replay_exact_cosine_without_large_search_response(demo):
    engine,enc,_=demo;encode_all(engine)
    result=engine.search('red','cover',top_k=3,dataset_id='ds_test')
    trace=engine.trace(result['trace_id'])
    values=trace['summary']['query_vector']['values']
    assert len(values)==DIM and np.isfinite(values).all()
    query=np.asarray(values,dtype=np.float32)
    ids,matrix,_=engine.store.gallery('cover','ds_test')
    replay={ident:float(matrix[i]@query) for i,ident in enumerate(ids)}
    for item in result['items']:
        assert item['score']==pytest.approx(replay[item['id']],abs=1e-7)
    assert all('values' not in event.get('details',{}) for event in trace['events'])
    assert 'query_vector' not in result
    assert all('values' not in item for item in result['items'])
