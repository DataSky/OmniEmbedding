"""Isolated, transactional SQLite storage for the local remote-inference demo."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import uuid

import numpy as np

MODEL = 'google/embeddinggemma-2'
REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
DIM = 768


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def decode(value, default=None):
    return json.loads(value) if value else default


class DemoStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        with self.connection() as con:
            con.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS datasets(
              id TEXT PRIMARY KEY,name TEXT NOT NULL,source_path TEXT,source_sha256 TEXT,
              source_type TEXT,summary_json TEXT,source_metadata_json TEXT,created_at TEXT,updated_at TEXT);
            CREATE TABLE IF NOT EXISTS assets(
              id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL REFERENCES datasets(id),source_id TEXT,
              title TEXT,keyword_top TEXT,keywords_json TEXT,metadata_json TEXT,record_json TEXT,
              status TEXT NOT NULL DEFAULT 'pending',created_at TEXT,updated_at TEXT);
            CREATE INDEX IF NOT EXISTS assets_dataset ON assets(dataset_id,status);
            CREATE TABLE IF NOT EXISTS artifacts(
              id TEXT PRIMARY KEY,asset_id TEXT NOT NULL REFERENCES assets(id),kind TEXT NOT NULL,
              path TEXT NOT NULL,input_hash TEXT,preprocessing TEXT,metadata_json TEXT,created_at TEXT);
            CREATE INDEX IF NOT EXISTS artifact_asset ON artifacts(asset_id,kind);
            CREATE TABLE IF NOT EXISTS embeddings(
              id INTEGER PRIMARY KEY,asset_id TEXT NOT NULL REFERENCES assets(id),kind TEXT NOT NULL,
              model TEXT NOT NULL,revision TEXT NOT NULL,dimensions INTEGER NOT NULL,
              storage_dtype TEXT NOT NULL,inference_dtype TEXT,device TEXT,input_hash TEXT NOT NULL,
              preprocessing TEXT NOT NULL,fingerprint TEXT NOT NULL,boot_id TEXT,stats_json TEXT,
              vector BLOB NOT NULL,active INTEGER NOT NULL DEFAULT 1,created_at TEXT,
              UNIQUE(asset_id,kind,fingerprint));
            CREATE INDEX IF NOT EXISTS embedding_gallery ON embeddings(kind,active,model,revision,dimensions);
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,params_json TEXT,
              progress_json TEXT,result_json TEXT,error_json TEXT,control TEXT,
              created_at TEXT,started_at TEXT,finished_at TEXT,updated_at TEXT);
            CREATE TABLE IF NOT EXISTS stages(
              asset_id TEXT NOT NULL REFERENCES assets(id),stage TEXT NOT NULL,status TEXT NOT NULL,
              job_id TEXT,trace_id TEXT,input_hash TEXT,elapsed_ms REAL,details_json TEXT,error_json TEXT,
              updated_at TEXT,PRIMARY KEY(asset_id,stage));
            CREATE TABLE IF NOT EXISTS traces(
              id TEXT PRIMARY KEY,kind TEXT NOT NULL,status TEXT,asset_id TEXT,job_id TEXT,
              started_at TEXT,finished_at TEXT,elapsed_ms REAL,summary_json TEXT,events_json TEXT);
            CREATE TABLE IF NOT EXISTS search_runs(
              id TEXT PRIMARY KEY REFERENCES traces(id),mode TEXT,query_json TEXT,dataset_id TEXT,
              candidate_count INTEGER,timings_json TEXT,results_json TEXT,created_at TEXT);
            ''')

    @contextmanager
    def connection(self, write=False):
        with self._write_lock if write else _NoLock():
            con = sqlite3.connect(self.path, timeout=30)
            con.row_factory = sqlite3.Row
            con.execute('PRAGMA foreign_keys=ON')
            con.execute('PRAGMA busy_timeout=30000')
            try:
                if write:
                    con.execute('BEGIN IMMEDIATE')
                yield con
                con.commit()
            except BaseException:
                con.rollback()
                raise
            finally:
                con.close()

    @staticmethod
    def _asset(row):
        if row is None:
            return None
        result = decode(row['record_json'], {})
        result.update(id=row['id'], asset_id=row['id'], dataset_id=row['dataset_id'],
                      title=row['title'], keyword_top=row['keyword_top'],
                      keywords=decode(row['keywords_json'], []), metadata=decode(row['metadata_json'], {}),
                      status=row['status'], created_at=row['created_at'], updated_at=row['updated_at'])
        return result

    def import_dataset(self, dataset, assets):
        stamp = now()
        with self.connection(write=True) as con:
            con.execute('''INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?)
              ON CONFLICT(id) DO UPDATE SET name=excluded.name,source_path=excluded.source_path,
              summary_json=excluded.summary_json,source_metadata_json=excluded.source_metadata_json,updated_at=excluded.updated_at''',
              (dataset['id'], dataset.get('name', dataset['id']), dataset.get('source_path'),
               dataset.get('source_sha256'), dataset.get('source_type'), encode(dataset.get('summary', {})),
               encode(dataset.get('source_metadata', {})), stamp, stamp))
            for asset in assets:
                previous = con.execute('SELECT record_json FROM assets WHERE id=?', (asset['id'],)).fetchone()
                if previous and decode(previous['record_json']) != asset:
                    con.execute('UPDATE embeddings SET active=0 WHERE asset_id=?', (asset['id'],))
                con.execute('''INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?)
                  ON CONFLICT(id) DO UPDATE SET title=excluded.title,keyword_top=excluded.keyword_top,
                  keywords_json=excluded.keywords_json,metadata_json=excluded.metadata_json,
                  record_json=excluded.record_json,updated_at=excluded.updated_at''',
                  (asset['id'], dataset['id'], str(asset.get('source_id', asset['id'])), asset.get('title', ''),
                   asset.get('keyword_top', ''), encode(asset.get('keywords', [])), encode(asset.get('metadata', {})),
                   encode(asset), 'pending', stamp, stamp))
        return self.dataset(dataset['id'])

    def dataset(self, ident):
        with self.connection() as con:
            row = con.execute('SELECT * FROM datasets WHERE id=?', (ident,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result['summary'] = decode(result.pop('summary_json'), {})
            result['source_metadata'] = decode(result.pop('source_metadata_json'), {})
            result['asset_count'] = con.execute('SELECT COUNT(*) FROM assets WHERE dataset_id=?', (ident,)).fetchone()[0]
            result['status_counts'] = dict(con.execute('SELECT status,COUNT(*) FROM assets WHERE dataset_id=? GROUP BY status', (ident,)).fetchall())
            result['vector_counts'] = dict(con.execute('''SELECT e.kind,COUNT(*) FROM embeddings e JOIN assets a ON a.id=e.asset_id
               WHERE a.dataset_id=? AND e.active=1 AND e.model=? AND e.revision=? AND e.dimensions=? GROUP BY e.kind''',
               (ident, MODEL, REVISION, DIM)).fetchall())
            return result

    def datasets(self):
        with self.connection() as con:
            ids = [r[0] for r in con.execute('SELECT id FROM datasets ORDER BY created_at DESC,id')]
        return [self.dataset(i) for i in ids]

    def asset(self, ident):
        with self.connection() as con:
            return self._asset(con.execute('SELECT * FROM assets WHERE id=?', (ident,)).fetchone())

    def assets(self, dataset_id=None, q='', status='', limit=100, offset=0):
        where, values = [], []
        if dataset_id:
            where.append('dataset_id=?'); values.append(dataset_id)
        if q:
            where.append('(title LIKE ? OR keyword_top LIKE ? OR keywords_json LIKE ? OR source_id LIKE ?)')
            values += ['%' + q + '%'] * 4
        if status:
            where.append('status=?'); values.append(status)
        clause = ' WHERE ' + ' AND '.join(where) if where else ''
        with self.connection() as con:
            total = con.execute('SELECT COUNT(*) FROM assets' + clause, values).fetchone()[0]
            rows = con.execute('SELECT * FROM assets' + clause + ' ORDER BY id LIMIT ? OFFSET ?', values + [limit, offset]).fetchall()
        return {'items': [self._asset(r) for r in rows], 'total': total}

    def all_assets(self, dataset_id=None):
        with self.connection() as con:
            rows = con.execute('SELECT * FROM assets' + (' WHERE dataset_id=?' if dataset_id else '') + ' ORDER BY id',
                               (dataset_id,) if dataset_id else ()).fetchall()
        return [self._asset(r) for r in rows]

    def asset_status(self, ident, status):
        with self.connection(write=True) as con:
            con.execute('UPDATE assets SET status=?,updated_at=? WHERE id=?', (status, now(), ident))

    def artifact(self, asset_id, kind, path, input_hash, preprocessing, metadata):
        ident = 'art_' + uuid.uuid4().hex
        with self.connection(write=True) as con:
            con.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?)',
                        (ident, asset_id, kind, str(path), input_hash, preprocessing, encode(metadata), now()))
        return ident

    def artifacts(self, asset_id, kind=None):
        with self.connection() as con:
            rows = con.execute('SELECT * FROM artifacts WHERE asset_id=?' + (' AND kind=?' if kind else '') + ' ORDER BY created_at,id',
                               (asset_id, kind) if kind else (asset_id,)).fetchall()
        return [{**dict(r), 'metadata': decode(r['metadata_json'], {})} for r in rows]

    def invalidate_vectors(self, asset_id, kinds):
        kinds = list(kinds)
        if not kinds:
            return
        with self.connection(write=True) as con:
            con.execute('UPDATE embeddings SET active=0 WHERE asset_id=? AND kind IN ('+','.join('?' for _ in kinds)+')', [asset_id,*kinds])

    def save_vector(self, asset_id, kind, vector, *, input_hash, preprocessing, fingerprint,
                    boot_id=None, inference_dtype='unknown', device='remote', stats=None):
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (DIM,) or not np.isfinite(array).all() or abs(float(np.linalg.norm(array)) - 1) > .001:
            raise ValueError('invalid_demo_vector')
        metrics = {'shape': [DIM], 'finite': True, 'norm': float(np.linalg.norm(array)),
                   'min': float(array.min()), 'max': float(array.max()), 'mean': float(array.mean()),
                   'nonzero': int(np.count_nonzero(array)), **(stats or {})}
        with self.connection(write=True) as con:
            con.execute('UPDATE embeddings SET active=0 WHERE asset_id=? AND kind=?', (asset_id, kind))
            con.execute('''INSERT INTO embeddings(asset_id,kind,model,revision,dimensions,storage_dtype,inference_dtype,device,
              input_hash,preprocessing,fingerprint,boot_id,stats_json,vector,active,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)
              ON CONFLICT(asset_id,kind,fingerprint) DO UPDATE SET vector=excluded.vector,boot_id=excluded.boot_id,
              inference_dtype=excluded.inference_dtype,stats_json=excluded.stats_json,active=1,created_at=excluded.created_at''',
              (asset_id, kind, MODEL, REVISION, DIM, 'float32', inference_dtype, device, input_hash,
               preprocessing, fingerprint, boot_id, encode(metrics), array.tobytes(), now()))
        return self.vector(asset_id, kind)

    @staticmethod
    def _vector(row, include_values=False):
        if row is None:
            return None
        result = dict(row)
        raw = result.pop('vector')
        result['stats'] = decode(result.pop('stats_json'), {})
        if include_values:
            result['values'] = np.frombuffer(raw, dtype=np.float32).astype(float).tolist()
        return result

    def vector(self, asset_id, kind, include_values=True, fingerprint=None):
        where = 'asset_id=? AND kind=? AND active=1 AND model=? AND revision=? AND dimensions=?'
        args = [asset_id, kind, MODEL, REVISION, DIM]
        if fingerprint:
            where += ' AND fingerprint=?'; args.append(fingerprint)
        with self.connection() as con:
            row = con.execute('SELECT * FROM embeddings WHERE ' + where + ' ORDER BY id DESC LIMIT 1', args).fetchone()
        return self._vector(row, include_values)

    def gallery(self, kind, dataset_id=None):
        where = 'e.kind=? AND e.active=1 AND e.model=? AND e.revision=? AND e.dimensions=?'
        args = [kind, MODEL, REVISION, DIM]
        if dataset_id:
            where += ' AND a.dataset_id=?'; args.append(dataset_id)
        with self.connection() as con:
            rows = con.execute('SELECT e.* FROM embeddings e JOIN assets a ON a.id=e.asset_id WHERE ' + where + ' ORDER BY e.asset_id', args).fetchall()
        ids, vectors, metadata = [], [], {}
        for row in rows:
            a = np.frombuffer(row['vector'], dtype=np.float32).copy()
            if a.shape == (DIM,) and np.isfinite(a).all() and abs(float(np.linalg.norm(a)) - 1) <= .001:
                ids.append(row['asset_id']); vectors.append(a); metadata[row['asset_id']] = self._vector(row)
        return ids, np.stack(vectors) if vectors else np.empty((0, DIM), np.float32), metadata

    def storage(self):
        tables = ('datasets','assets','artifacts','embeddings','jobs','stages','traces','search_runs')
        with self.connection() as con:
            counts = {name:con.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in tables}
            schema = [{'table':name,'columns':[{'name':r['name'],'type':r['type'],'primary_key':bool(r['pk'])} for r in con.execute('PRAGMA table_info('+name+')')]} for name in tables]
            mode = con.execute('PRAGMA journal_mode').fetchone()[0]
        return {'backend':'SQLite','db_path':str(self.path.resolve()),'db_bytes':self.path.stat().st_size,
                'wal_bytes':Path(str(self.path)+'-wal').stat().st_size if Path(str(self.path)+'-wal').exists() else 0,
                'journal_mode':mode,'vector_dtype':'float32','dimensions':DIM,'table_counts':counts,'schema':schema}

    def set_stage(self, asset_id, stage, status, *, job_id=None, trace_id=None, input_hash=None,
                  elapsed_ms=None, details=None, error=None):
        with self.connection(write=True) as con:
            con.execute('''INSERT INTO stages VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(asset_id,stage) DO UPDATE SET
              status=excluded.status,job_id=excluded.job_id,trace_id=excluded.trace_id,input_hash=excluded.input_hash,
              elapsed_ms=excluded.elapsed_ms,details_json=excluded.details_json,error_json=excluded.error_json,updated_at=excluded.updated_at''',
              (asset_id, stage, status, job_id, trace_id, input_hash, elapsed_ms, encode(details or {}), encode(error) if error else None, now()))

    def stages(self, asset_id):
        with self.connection() as con:
            rows = con.execute('SELECT * FROM stages WHERE asset_id=? ORDER BY updated_at,stage', (asset_id,)).fetchall()
        return [{**dict(r), 'details': decode(r['details_json'], {}), 'error': decode(r['error_json'])} for r in rows]

    def create_job(self, kind, params):
        ident, stamp = 'job_' + uuid.uuid4().hex, now()
        with self.connection(write=True) as con:
            con.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                        (ident, kind, 'queued', encode(params), encode({}), None, None, None, stamp, None, None, stamp))
        return self.job(ident)

    @staticmethod
    def _job(row):
        if row is None:
            return None
        result = dict(row)
        for key in ('params', 'progress', 'result', 'error'):
            result[key] = decode(result.pop(key + '_json'), {} if key in ('params', 'progress') else None)
        return result

    def job(self, ident):
        with self.connection() as con:
            return self._job(con.execute('SELECT * FROM jobs WHERE id=?', (ident,)).fetchone())

    def jobs(self, limit=100):
        with self.connection() as con:
            return [self._job(r) for r in con.execute('SELECT * FROM jobs ORDER BY created_at DESC,id LIMIT ?', (limit,))]

    def update_job(self, ident, **fields):
        allowed = {'status','control','started_at','finished_at','progress','result','error'}
        if not fields.keys() <= allowed:
            raise ValueError('invalid_job_update')
        changes, args = [], []
        for key, value in fields.items():
            column = key + '_json' if key in {'progress','result','error'} else key
            changes.append(column + '=?'); args.append(encode(value) if column.endswith('_json') else value)
        changes.append('updated_at=?'); args += [now(), ident]
        with self.connection(write=True) as con:
            con.execute('UPDATE jobs SET ' + ','.join(changes) + ' WHERE id=?', args)
        return self.job(ident)

    def job_observations(self, ident, limit=200):
        with self.connection() as con:
            trace_total = con.execute('SELECT COUNT(*) FROM traces WHERE job_id=?',(ident,)).fetchone()[0]
            stage_total = con.execute('SELECT COUNT(*) FROM stages WHERE job_id=?',(ident,)).fetchone()[0]
            traces = con.execute('SELECT * FROM traces WHERE job_id=? ORDER BY started_at DESC,id LIMIT ?',(ident,limit)).fetchall()
            stages = con.execute('SELECT * FROM stages WHERE job_id=? ORDER BY updated_at DESC,asset_id,stage LIMIT ?',(ident,limit)).fetchall()
        trace_summaries=[];events=[]
        for row in traces:
            summary=decode(row['summary_json'],{})
            trace_summaries.append({**{k:row[k] for k in ('id','kind','status','asset_id','job_id','started_at','finished_at','elapsed_ms')},'summary':summary})
            if row['kind']=='import':
                events.extend(decode(row['events_json'],[])[:limit])
        aggregate=[{'stage':t['kind'],'status':t['status'],'elapsed_ms':t['elapsed_ms'],
                    'details':{'asset_id':t['asset_id'],'trace_id':t['id'],'summary':t['summary']}} for t in trace_summaries]
        actual=[{**{k:r[k] for k in ('asset_id','stage','status','trace_id','input_hash','elapsed_ms','updated_at')},
                 'details':decode(r['details_json'],{}),'error':decode(r['error_json'])} for r in stages]
        return {'traces':trace_summaries,'stages':aggregate,'asset_stages':actual,'events':events[:limit],
                'trace_total':trace_total,'asset_stage_total':stage_total,'observation_limit':limit,
                'observations_truncated':trace_total>limit or stage_total>limit or len(events)>limit}

    def recover_interrupted_jobs(self):
        with self.connection(write=True) as con:
            con.execute("UPDATE jobs SET status='paused',control=NULL,updated_at=?,error_json=? WHERE status IN ('running','cancelling','pausing')",
                        (now(), encode({'class':'WorkerRestarted','message':'进程重启后暂停；可重试，已完成向量保留。'})))

    def new_trace(self, kind, *, asset_id=None, job_id=None):
        ident = 'trace_' + uuid.uuid4().hex
        with self.connection(write=True) as con:
            con.execute('INSERT INTO traces VALUES(?,?,?,?,?,?,?,?,?,?)',
                        (ident, kind, 'running', asset_id, job_id, now(), None, None, encode({}), encode([])))
        return ident

    def finish_trace(self, ident, *, status, elapsed_ms, summary, events):
        with self.connection(write=True) as con:
            con.execute('UPDATE traces SET status=?,finished_at=?,elapsed_ms=?,summary_json=?,events_json=? WHERE id=?',
                        (status, now(), elapsed_ms, encode(summary), encode(events), ident))

    def traces(self, limit=100):
        with self.connection() as con:
            rows = con.execute('SELECT * FROM traces ORDER BY started_at DESC,id LIMIT ?', (limit,)).fetchall()
        return [{**{k:r[k] for k in ('id','kind','status','asset_id','job_id','started_at','finished_at','elapsed_ms')},
                 'summary':decode(r['summary_json'], {})} for r in rows]

    def trace(self, ident):
        with self.connection() as con:
            row = con.execute('SELECT * FROM traces WHERE id=?', (ident,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result['summary'] = decode(result.pop('summary_json'), {})
        result['events'] = decode(result.pop('events_json'), [])
        return result

    def save_search(self, ident, mode, query, dataset_id, count, timings, results):
        with self.connection(write=True) as con:
            con.execute('INSERT INTO search_runs VALUES(?,?,?,?,?,?,?,?)',
                        (ident, mode, encode(query), dataset_id, count, encode(timings), encode(results), now()))


class _NoLock:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
