"""SQLite snapshots, FTS5 BM25, vectors and resumable job metadata."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import uuid

import numpy as np

from ..config import DEFAULT_MODEL, PINNED_REVISION

DIMENSIONS = 768
MODEL = DEFAULT_MODEL
REVISION = PINNED_REVISION
RRF_K = 60


def now():
    return datetime.now(timezone.utc).isoformat()


def tokens(text):
    """Identifiers, camelCase components, Chinese unigrams and bigrams.

    Store the same pretokenized representation for documents and queries so
    SQLite's default Unicode tokenizer does not treat a Chinese sentence as
    a single term. Retain full identifiers in addition to their components.
    """
    result = []
    for value in re.findall(r"[\u3400-\u9fff]+|[A-Za-z0-9_]+", text):
        if "\u3400" <= value[0] <= "\u9fff":
            result.extend(value)
            result.extend(value[i:i+2] for i in range(len(value)-1))
        else:
            result.append(value.lower())
            pieces = re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9_]|$)|[A-Z]?[a-z]+|[0-9]+", value)
            lowered = [p.lower() for p in pieces]
            if lowered != [value.lower()]:
                result.extend(lowered)
    return result


class Store:
    def __init__(self, path, root=None):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.generation = 0
        self._pending_after = 0
        self._coverage_integrity_verified = False
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, project TEXT NOT NULL,
                    extension TEXT, size INTEGER, mtime_ns INTEGER, sha256 TEXT,
                    status TEXT, error TEXT, chunk_count INTEGER DEFAULT 0,
                    updated_at TEXT, content TEXT NOT NULL DEFAULT '', policy TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    ordinal INTEGER, start_line INTEGER, end_line INTEGER,
                    content TEXT NOT NULL, embedding_text TEXT NOT NULL, sha256 TEXT, fts_rowid INTEGER,
                    UNIQUE(file_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
                CREATE INDEX IF NOT EXISTS files_project ON files(project, extension);
                CREATE INDEX IF NOT EXISTS files_summary ON files(project,extension,id,chunk_count);
                CREATE INDEX IF NOT EXISTS chunks_coverage ON chunks(file_id,id);
                CREATE TABLE IF NOT EXISTS vectors (
                    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL, revision TEXT NOT NULL, dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL, created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS vectors_space ON vectors(model,revision,dimensions);
                CREATE UNIQUE INDEX IF NOT EXISTS vectors_coverage ON vectors(model,revision,dimensions,chunk_id);
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, action TEXT, status TEXT, phase TEXT,
                    files_done INTEGER DEFAULT 0, files_total INTEGER DEFAULT 0,
                    chunks_done INTEGER DEFAULT 0, chunks_total INTEGER DEFAULT 0,
                    vectors_done INTEGER DEFAULT 0, vectors_total INTEGER DEFAULT 0,
                    skipped INTEGER DEFAULT 0, errors INTEGER DEFAULT 0,
                    message TEXT, started_at TEXT, finished_at TEXT, details TEXT
                );
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, query TEXT NOT NULL,
                    mode TEXT NOT NULL, elapsed_ms REAL NOT NULL, payload TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED, path, body, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    status TEXT NOT NULL, phase TEXT NOT NULL, items INTEGER NOT NULL,
                    characters INTEGER NOT NULL, elapsed_ms REAL NOT NULL, http_status INTEGER,
                    error_code TEXT, boot_id TEXT, input_hashes TEXT NOT NULL,
                    started_at TEXT, finished_at TEXT, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS batches_job ON batches(job_id,created_at);
                CREATE TABLE IF NOT EXISTS job_errors (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    path TEXT NOT NULL, reason TEXT NOT NULL, is_dir INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS errors_job ON job_errors(job_id,created_at);
            """)
            # Legacy FTS identifiers were UNINDEXED. Materialize one mapping once;
            # a correlated lookup for each chunk would itself be quadratic.
            if "fts_rowid" not in {r[1] for r in db.execute("PRAGMA table_info(chunks)")}:
                db.execute("ALTER TABLE chunks ADD COLUMN fts_rowid INTEGER")
                db.execute("CREATE TEMP TABLE fts_identity(chunk_id TEXT PRIMARY KEY, fts_rowid INTEGER)")
                db.execute("INSERT INTO fts_identity SELECT chunk_id,rowid FROM chunk_fts")
                db.execute("UPDATE chunks SET fts_rowid=(SELECT fts_rowid FROM fts_identity WHERE chunk_id=chunks.id)")
                db.execute("DROP TABLE fts_identity")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS chunks_fts_identity ON chunks(fts_rowid)")
            if "policy" not in {r[1] for r in db.execute("PRAGMA table_info(files)")}:
                db.execute("ALTER TABLE files ADD COLUMN policy TEXT")
                previous_policy = db.execute("SELECT value FROM meta WHERE key='chunk_policy'").fetchone()
                if previous_policy and previous_policy[0] == "line-v1:2400:6":
                    db.execute("UPDATE files SET policy=?", (previous_policy[0],))
            db.execute("PRAGMA user_version=2")
            # Enabling FKs does not validate rows written by an older/external
            # connection. Prove the two relationships once using key indexes
            # before replacing the unfiltered join with independent counts.
            if not db.in_transaction:
                db.execute("BEGIN")
            links = {table: {(r["table"], r["from"], r["to"], r["on_delete"])
                              for r in db.execute("PRAGMA foreign_key_list("+table+")")}
                     for table in ("chunks", "vectors")}
            constrained = (("files", "file_id", "id", "CASCADE") in links["chunks"]
                           and ("chunks", "chunk_id", "id", "CASCADE") in links["vectors"])
            if constrained:
                orphan_chunk = db.execute("SELECT 1 FROM chunks c LEFT JOIN files f ON f.id=c.file_id "
                                          "WHERE f.id IS NULL LIMIT 1").fetchone()
                orphan_vector = db.execute("SELECT 1 FROM vectors v LEFT JOIN chunks c ON c.id=v.chunk_id "
                                           "WHERE c.id IS NULL LIMIT 1").fetchone()
                self._coverage_integrity_verified = orphan_chunk is None and orphan_vector is None
        for private in (self.path, Path(str(self.path)+"-wal"), Path(str(self.path)+"-shm")):
            try:
                os.chmod(private, 0o600)
            except FileNotFoundError:
                pass
        if root is not None:
            configured = str(Path(root).resolve())
            previous = self.meta_get("root")
            if previous and previous != configured:
                raise ValueError("此数据库已绑定另一个源目录，请指定新的数据目录。")
            self.meta_set("root", configured)
        self.meta_set("tokenizer", "code-camel-cjk-bigram-v1")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def mark_dirty(self):
        self.generation += 1

    def meta_get(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def meta_set(self, key, value):
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def file_by_path(self, path):
        with self.connect() as db:
            row = db.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()
        return dict(row) if row else None

    def patch_file(self, record):
        allowed = {"project", "extension", "size", "mtime_ns", "sha256", "status", "error", "updated_at", "chunk_count", "policy"}
        fields = {k:v for k,v in record.items() if k in allowed}
        fields.setdefault("updated_at", now())
        with self.connect() as db:
            db.execute("UPDATE files SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?",
                       [*fields.values(), record["id"]])

    def replace_file(self, record, chunks):
        fields = ["id", "path", "project", "extension", "size", "mtime_ns", "sha256", "status", "error", "chunk_count", "updated_at", "content", "policy"]
        record = {**record, "chunk_count":len(chunks), "updated_at":now()}
        values = [record.get(k, "" if k == "content" else None) for k in fields]
        path_tokens = " ".join(tokens(record["path"]))
        with self.lock, self.connect() as db:
            old = db.execute("SELECT fts_rowid FROM chunks WHERE file_id=?", (record["id"],)).fetchall()
            if old:
                db.executemany("DELETE FROM chunk_fts WHERE rowid=?", ((r[0],) for r in old))
            db.execute("DELETE FROM chunks WHERE file_id=?", (record["id"],))
            db.execute("INSERT INTO files (" + ",".join(fields) + ") VALUES (" + ",".join("?" for _ in fields) + ") "
                       "ON CONFLICT(id) DO UPDATE SET " + ",".join(f"{k}=excluded.{k}" for k in fields if k != "id"), values)
            for c in chunks:
                cursor = db.execute("INSERT INTO chunk_fts(chunk_id,path,body) VALUES (?,?,?)",
                                    (c["id"], path_tokens, " ".join(tokens(c["content"]))))
                db.execute("INSERT INTO chunks(id,file_id,ordinal,start_line,end_line,content,embedding_text,sha256,fts_rowid) VALUES (?,?,?,?,?,?,?,?,?)",
                           (c["id"], record["id"], c["ordinal"], c["start_line"], c["end_line"], c["content"],
                            c["embedding_text"], c.get("sha256"), cursor.lastrowid))
        self.mark_dirty()

    def prune_files(self, seen_paths):
        # TEMP table avoids SQL variable limits with large source trees.
        with self.lock, self.connect() as db:
            db.execute("CREATE TEMP TABLE seen(path TEXT PRIMARY KEY)")
            db.executemany("INSERT OR IGNORE INTO seen VALUES (?)", ((str(p),) for p in seen_paths))
            removed = db.execute("SELECT count(*) FROM files WHERE path NOT IN (SELECT path FROM seen)").fetchone()[0]
            rows = db.execute("SELECT c.fts_rowid FROM chunks c JOIN files f ON f.id=c.file_id WHERE f.path NOT IN (SELECT path FROM seen)").fetchall()
            if rows:
                db.executemany("DELETE FROM chunk_fts WHERE rowid=?", ((r[0],) for r in rows))
            db.execute("DELETE FROM files WHERE path NOT IN (SELECT path FROM seen)")
        self.mark_dirty()
        self._pending_after = 0
        return removed

    def vector_put(self, chunk_id, vector, model=MODEL, revision=REVISION):
        self.vector_put_many([(chunk_id, vector)], model, revision)

    def vector_put_many(self, items, model=MODEL, revision=REVISION):
        prepared = []
        for chunk_id, vector in items:
            prepared.append((self._vector_chunk_id(chunk_id), model, revision, DIMENSIONS, self._vector_bytes(vector), now()))
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?)", prepared)
        self.mark_dirty()

    def vector_put_many_with_audit(self, items, model=MODEL, revision=REVISION, *, job_id, batch):
        prepared = [(self._vector_chunk_id(cid), model, revision, DIMENSIONS, self._vector_bytes(vector), now()) for cid, vector in items]
        audit = self._prepare_batch(job_id, batch)
        if audit["status"] != "success" or len(prepared) != audit["items"]:
            raise ValueError("成功批次与向量数量不一致")
        if {r[0] for r in prepared} != {r["chunk_id"] for r in audit["input_hashes"]}:
            raise ValueError("批次输入标识不一致")
        with self.lock, self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?)", prepared)
            self._insert_batch(db, audit)
        self.mark_dirty()
        return audit

    @staticmethod
    def _prepare_batch(job_id, fields):
        allowed = {"status", "phase", "items", "characters", "elapsed_ms", "http_status", "error_code",
                   "boot_id", "input_hashes", "started_at", "finished_at"}
        if set(fields) - allowed:
            raise ValueError("未知批次字段")
        value = dict(id=uuid.uuid4().hex, job_id=job_id, status="failed", phase="embed", items=0,
                     characters=0, elapsed_ms=0.0, http_status=None, error_code=None, boot_id=None,
                     input_hashes=[], started_at=None, finished_at=now(), created_at=now())
        value.update(fields)
        if (value["status"] not in {"success", "failed"} or value["phase"] not in {"scan", "embed"}
                or type(value["items"]) is not int or not 0 <= value["items"] <= 16
                or type(value["characters"]) is not int or not 0 <= value["characters"] <= 32768
                or not np.isfinite(value["elapsed_ms"]) or value["elapsed_ms"] < 0
                or (value["http_status"] is not None and (type(value["http_status"]) is not int or not 100 <= value["http_status"] <= 599))):
            raise ValueError("批次元数据无效")
        for key in ("error_code", "boot_id"):
            if value[key] is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value[key]):
                raise ValueError("批次诊断字段必须为安全短码")
        hashes = value["input_hashes"]
        if (not isinstance(hashes, list) or len(hashes) != value["items"]
                or any(not isinstance(h, dict) or set(h) != {"chunk_id", "sha256"}
                       or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", h["chunk_id"])
                       or not re.fullmatch(r"[a-f0-9]{64}", h["sha256"]) for h in hashes)):
            raise ValueError("批次输入摘要无效")
        return value

    @staticmethod
    def _insert_batch(db, value):
        fields = list(value)
        row = {**value, "input_hashes":json.dumps(value["input_hashes"], separators=(",", ":"))}
        db.execute("INSERT INTO batches ("+",".join(fields)+") VALUES ("+",".join("?" for _ in fields)+")", [row[k] for k in fields])

    def batch_record(self, job_id, **fields):
        value = self._prepare_batch(job_id, fields)
        with self.connect() as db:
            self._insert_batch(db, value)
        return value

    def list_batches(self, job_id=None, limit=30):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM batches" + (" WHERE job_id=?" if job_id else "") + " ORDER BY rowid DESC LIMIT ?",
                              [*([job_id] if job_id else []), limit]).fetchall()
        values = [dict(r) for r in rows]
        for value in values:
            value["input_hashes"] = json.loads(value["input_hashes"])
        return values

    def error_record(self, job_id, path, reason, is_dir=False):
        if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,128}", reason):
            raise ValueError("错误原因必须为安全短码")
        value = dict(id=uuid.uuid4().hex, job_id=job_id, path=path, reason=reason, is_dir=bool(is_dir), created_at=now())
        with self.connect() as db:
            db.execute("INSERT INTO job_errors VALUES (?,?,?,?,?,?)", tuple(value.values()))
        return value

    def job_detail(self, ident):
        job = self.job_get(ident)
        if job is None:
            return None
        with self.connect() as db:
            errors = [dict(r) for r in db.execute("SELECT * FROM job_errors WHERE job_id=? ORDER BY rowid DESC LIMIT 1000", (ident,))]
            counts = [dict(r) for r in db.execute("SELECT status,count(*) count,sum(items) items FROM batches WHERE job_id=? GROUP BY status", (ident,))]
        return {"job":job, "batches":self.list_batches(ident, 100), "errors":errors, "batch_summary":counts}

    def exclusions(self, q="", reason=None, project=None, limit=50, offset=0):
        where, args = ["chunk_count=0", "status IN ('skipped','error')"], []
        for field, value in (("error", reason), ("project", project)):
            if value:
                where.append(field+"=?"); args.append(value)
        if q:
            where.append("instr(lower(path),lower(?))>0"); args.append(q)
        condition = " AND ".join(where)
        with self.connect() as db:
            db.execute("BEGIN")
            total = db.execute("SELECT count(*) FROM files WHERE "+condition, args).fetchone()[0]
            rows = db.execute("SELECT id,path,project,extension,size,status,error,chunk_count FROM files WHERE "+condition+" ORDER BY path LIMIT ? OFFSET ?", [*args,limit,offset])
            values = [dict(r) for r in rows]
            reasons = [dict(r) for r in db.execute("SELECT error reason,count(*) count FROM files WHERE chunk_count=0 AND status IN ('skipped','error') GROUP BY error ORDER BY count DESC")]
        return dict(total=total, items=values, reasons=reasons)

    @staticmethod
    def _vector_chunk_id(chunk_id):
        # Rowid-table TEXT PRIMARY KEY permits NULL, and FK checks ignore it.
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("分块标识必须为非空字符串")
        return chunk_id

    @staticmethod
    def _vector_bytes(vector):
        vector = np.asarray(vector, dtype="<f4")
        if vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=.05):
            raise ValueError("向量维度、数值或归一化校验失败")
        vector = vector / np.linalg.norm(vector)
        return vector.astype("<f4").tobytes()

    def pending_chunks(self, model=MODEL, revision=REVISION, limit=16):
        with self.connect() as db:
            def fetch(after):
                return db.execute("""SELECT c.*,c.rowid cursor_row,f.path FROM chunks c JOIN files f ON f.id=c.file_id
                    LEFT JOIN vectors v ON v.chunk_id=c.id
                    WHERE c.rowid>? AND (v.chunk_id IS NULL OR v.model!=? OR v.revision!=? OR v.dimensions!=?)
                    ORDER BY c.rowid LIMIT ?""", (after, model, revision, DIMENSIONS, limit)).fetchall()
            rows = fetch(self._pending_after)
            if not rows and self._pending_after:
                rows = fetch(0)
            self._pending_after = rows[-1]["cursor_row"] if rows else 0
        return [dict(r) for r in rows]

    def stats(self):
        with self.connect() as db:
            db.execute("BEGIN")
            files = db.execute("SELECT count(*) FROM files WHERE chunk_count>0").fetchone()[0]
            chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            vectors = db.execute("SELECT count(*) FROM vectors WHERE model=? AND revision=? AND dimensions=?", (MODEL, REVISION, DIMENSIONS)).fetchone()[0]
            projects = [dict(r) for r in db.execute("SELECT project name,count(*) files,sum(chunk_count) chunks FROM files WHERE chunk_count>0 GROUP BY project ORDER BY files DESC")]
            extensions = [dict(r) for r in db.execute("SELECT extension,count(*) files FROM files WHERE chunk_count>0 GROUP BY extension ORDER BY files DESC")]
        return dict(files=files, chunks=chunks, vectors=vectors, projects=projects, extensions=extensions,
                    storage_bytes=sum(p.stat().st_size for p in [self.path, Path(str(self.path)+"-wal")] if p.exists()))

    def vector_count(self):
        with self.connect() as db:
            return db.execute("SELECT count(*) FROM vectors WHERE model=? AND revision=? AND dimensions=?", (MODEL, REVISION, DIMENSIONS)).fetchone()[0]

    def coverage(self, project=None, extension=None):
        where, args = [], [MODEL, REVISION, DIMENSIONS]
        if project:
            where.append("f.project=?"); args.append(project)
        if extension:
            where.append("f.extension=?"); args.append(extension.lower())
        with self.connect() as db:
            # Evaluation readers may already hold a frozen transaction.
            if not db.in_transaction:
                db.execute("BEGIN")
            if not where and getattr(self, "_coverage_integrity_verified", False):
                # Foreign keys are enabled on every writer and both edges
                # cascade on deletion; each chunk has at most one vector.
                # Count narrow indexes instead of probing source/blob rows.
                chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
                vectors = db.execute("SELECT count(*) FROM vectors WHERE model=? AND revision=? AND dimensions=?", args).fetchone()[0]
                return dict(chunks=chunks, vectors=vectors)
            row = db.execute("SELECT count(*) chunks,count(v.chunk_id) vectors FROM chunks c "
                             "JOIN files f ON f.id=c.file_id LEFT JOIN vectors v ON v.chunk_id=c.id "
                             "AND v.model=? AND v.revision=? AND v.dimensions=?" +
                             (" WHERE " + " AND ".join(where) if where else ""), args).fetchone()
        return dict(row)

    @staticmethod
    def _job(row):
        if row is None:
            return None
        value = dict(row)
        value["details"] = json.loads(value.get("details") or "{}")
        return value

    def job_create(self, action):
        ident = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id,action,status,phase,message,started_at,details) VALUES (?,?,?,?,?,?,?)",
                       (ident, action, "queued", "queued", "任务已排队", now(), "{}"))
        return self.job_get(ident)

    def job_get(self, ident):
        with self.connect() as db:
            return self._job(db.execute("SELECT * FROM jobs WHERE id=?", (ident,)).fetchone())

    def job_update(self, ident, **fields):
        allowed = {"status", "phase", "files_done", "files_total", "chunks_done", "chunks_total", "vectors_done", "vectors_total", "skipped", "errors", "message", "started_at", "finished_at", "details"}
        if fields.keys() - allowed:
            raise ValueError("未知任务字段")
        if isinstance(fields.get("details"), dict):
            fields["details"] = json.dumps(fields["details"], ensure_ascii=False)
        if fields:
            with self.connect() as db:
                db.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?", [*fields.values(), ident])
        return self.job_get(ident)

    def job_current(self):
        with self.connect() as db:
            return self._job(db.execute("SELECT * FROM jobs ORDER BY rowid DESC LIMIT 1").fetchone())

    def job_recent(self, limit=20):
        with self.connect() as db:
            return [self._job(r) for r in db.execute("SELECT * FROM jobs ORDER BY rowid DESC LIMIT ?", (limit,))]

    def list_files(self, project=None, q="", limit=50, offset=0):
        where, args = ["chunk_count>0"], []
        if project:
            where.append("project=?"); args.append(project)
        if q:
            where.append("instr(lower(path),lower(?))>0"); args.append(q)
        condition = " AND ".join(where)
        with self.connect() as db:
            db.execute("BEGIN")
            total = db.execute("SELECT count(*) FROM files WHERE " + condition, args).fetchone()[0]
            rows = db.execute("SELECT id,path,project,extension,size,chunk_count,status,error,updated_at,"
                              "(SELECT count(*) FROM vectors v JOIN chunks c ON c.id=v.chunk_id WHERE c.file_id=files.id AND v.model=? AND v.revision=? AND v.dimensions=?) vector_count "
                              "FROM files WHERE " + condition + " ORDER BY path LIMIT ? OFFSET ?",
                              [MODEL, REVISION, DIMENSIONS, *args, limit, offset]).fetchall()
        return {"total":total, "items":[dict(r) for r in rows]}

    def file_detail(self, ident):
        with self.connect() as db:
            db.execute("BEGIN")
            record = db.execute("SELECT * FROM files WHERE id=?", (ident,)).fetchone()
            if record is None:
                return None
            file = dict(record); content = file.pop("content")
            chunks = [dict(r) for r in db.execute("SELECT c.id,c.ordinal,c.start_line,c.end_line,c.content,"
                "CASE WHEN v.chunk_id IS NOT NULL AND v.model=? AND v.revision=? AND v.dimensions=? THEN 'ready' ELSE 'pending' END vector_status "
                "FROM chunks c LEFT JOIN vectors v ON v.chunk_id=c.id WHERE c.file_id=? ORDER BY c.ordinal", (MODEL, REVISION, DIMENSIONS, ident))]
        return {"file":file, "chunks":chunks, "content":content}

    def chunk_detail(self, ident):
        with self.connect() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT c.*,f.path,f.project,f.extension FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.id=?", (ident,)).fetchone()
            if row is None:
                return None
            chunk = dict(row)
            file = dict(db.execute("SELECT * FROM files WHERE id=?", (chunk["file_id"],)).fetchone())
            lines = file.pop("content").splitlines()
            begin, end = max(1, chunk["start_line"]-8), min(len(lines), chunk["end_line"]+8)
            vector = db.execute("SELECT * FROM vectors WHERE chunk_id=?", (ident,)).fetchone()
        vector_info = None
        if vector:
            values = np.frombuffer(vector["vector"], dtype="<f4")
            vector_info = {k:vector[k] for k in ("model", "revision", "dimensions", "created_at")}
            vector_info.update(norm=float(np.linalg.norm(values)), values=values.tolist())
        return {"chunk":chunk, "file":file, "context":[{"number":i, "text":lines[i-1]} for i in range(begin, end+1)], "vector":vector_info}

    def add_trace(self, result, params):
        payload = {**result, "parameters":params, "created_at":now()}
        # Keep trace metadata, not duplicate source text at each query.
        payload["results"] = [{k:v for k,v in r.items() if k not in {"content", "embedding_text"}} for r in result["results"]]
        with self.connect() as db:
            db.execute("INSERT INTO traces VALUES (?,?,?,?,?,?)", (result["trace_id"], payload["created_at"], result["query"], result["mode"], result["elapsed_ms"], json.dumps(payload, ensure_ascii=False, allow_nan=False)))
            db.execute("DELETE FROM traces WHERE rowid < (SELECT max(rowid)-1000 FROM traces)")

    def traces(self, limit=20):
        with self.connect() as db:
            return {"items":[json.loads(r[0]) for r in db.execute("SELECT payload FROM traces ORDER BY rowid DESC LIMIT ?", (limit,))]}
