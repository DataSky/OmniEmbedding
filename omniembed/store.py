"""存储层：SQLite 保存元数据、任务状态和向量（float16 BLOB），单文件即可备份迁移。

表：
  videos   一条视频一行：来源、标题、描述、标签、状态、错误、媒体信息
  vectors  每条视频的向量：kind = title / visual / fused，dim=768，float16
  frames   抽帧记录：时间点、关键帧缩略图路径
  evalsets 评测集：名称 + JSON（查询 → 标准答案 video_id 列表）
"""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np

STATUSES = ("queued", "downloading", "probing", "extracting", "embedding", "ready", "failed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,            -- upload / url / folder
  source_ref TEXT,                 -- 原始文件名或 URL
  path TEXT,                       -- 本地视频文件
  sha256 TEXT,
  title TEXT DEFAULT '',
  description TEXT DEFAULT '',
  tags TEXT DEFAULT '[]',          -- JSON 数组
  extra TEXT DEFAULT '{}',         -- 业务方自带字段（JSON）
  status TEXT NOT NULL DEFAULT 'queued',
  error TEXT,
  bytes INTEGER,
  duration_s REAL, width INTEGER, height INTEGER, fps REAL, codec TEXT, has_audio INTEGER,
  n_frames INTEGER,
  cover TEXT,
  timings TEXT DEFAULT '{}',       -- 各阶段耗时（JSON）
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_sha ON videos(sha256);
CREATE TABLE IF NOT EXISTS vectors (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  dim INTEGER NOT NULL,
  model TEXT NOT NULL,
  vec BLOB NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (video_id, kind)
);
CREATE TABLE IF NOT EXISTS frames (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  t_s REAL NOT NULL,
  thumb TEXT,
  PRIMARY KEY (video_id, idx)
);
CREATE TABLE IF NOT EXISTS evalsets (
  name TEXT PRIMARY KEY,
  description TEXT DEFAULT '',
  items TEXT NOT NULL,             -- JSON: [{"query": "...", "gold": ["video_id", ...]}]
  created_at REAL NOT NULL
);
"""

VIDEO_COLS = ("id", "source", "source_ref", "path", "sha256", "title", "description", "tags", "extra", "status",
              "error", "bytes", "duration_s", "width", "height", "fps", "codec", "has_audio", "n_frames", "cover",
              "timings", "created_at", "updated_at")


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
        self.version = 0  # 每次向量变化 +1，索引据此判断是否需要重载

    # ---------- videos ----------
    def add_video(self, source, source_ref, title="", description="", tags=None, extra=None, path=None) -> str:
        vid = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO videos(id, source, source_ref, path, title, description, tags, extra, status, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?, 'queued', ?, ?)",
                (vid, source, source_ref, str(path) if path else None, title or "", description or "",
                 json.dumps(tags or [], ensure_ascii=False), json.dumps(extra or {}, ensure_ascii=False), now, now))
        return vid

    def update_video(self, vid, **fields):
        if not fields:
            return
        bad = set(fields) - set(VIDEO_COLS)
        if bad:
            raise ValueError(f"unknown columns: {bad}")
        for k in ("tags", "extra", "timings"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k], ensure_ascii=False)
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE videos SET {cols} WHERE id=?", (*fields.values(), vid))
        if "status" in fields or "title" in fields or "description" in fields:
            self.version += 1  # 召回只用 ready 的视频，标题也参与 BM25，变化时让索引重载

    def get_video(self, vid):
        with self._lock:
            r = self._conn.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
        return self._row(r) if r else None

    def find_by_sha(self, sha):
        with self._lock:
            r = self._conn.execute("SELECT id FROM videos WHERE sha256=? AND status='ready' LIMIT 1", (sha,)).fetchone()
        return r["id"] if r else None

    def list_videos(self, status=None, q=None, limit=50, offset=0):
        sql, args = "SELECT * FROM videos", []
        conds = []
        if status:
            conds.append("status=?")
            args.append(status)
        if q:
            conds.append("(title LIKE ? OR description LIKE ? OR source_ref LIKE ?)")
            args += [f"%{q}%"] * 3
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
            total = self._conn.execute("SELECT COUNT(*) FROM videos" + (" WHERE " + " AND ".join(conds) if conds else ""),
                                       args[:-2]).fetchone()[0]
        return [self._row(r) for r in rows], total

    def counts(self):
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) n FROM videos GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def next_queued(self):
        with self._lock:
            r = self._conn.execute("SELECT id FROM videos WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not r:
                return None
            # 原子地领取任务，避免多个 worker 处理同一条
            cur = self._conn.execute("UPDATE videos SET status='downloading', updated_at=? WHERE id=? AND status='queued'",
                                     (time.time(), r["id"]))
            return r["id"] if cur.rowcount else None

    def requeue_stale(self):
        """服务重启时，把上次没跑完的任务放回队列。"""
        with self._lock:
            cur = self._conn.execute("UPDATE videos SET status='queued' WHERE status IN ('downloading','probing','extracting','embedding')")
        return cur.rowcount

    def delete_video(self, vid):
        with self._lock:
            self._conn.execute("DELETE FROM videos WHERE id=?", (vid,))
        self.version += 1

    # ---------- vectors ----------
    def put_vector(self, vid, kind, vec: np.ndarray, model: str):
        v = np.asarray(vec, dtype=np.float32).ravel()
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO vectors(video_id, kind, dim, model, vec, created_at) VALUES (?,?,?,?,?,?)",
                               (vid, kind, int(v.size), model, v.astype(np.float16).tobytes(), time.time()))
        self.version += 1

    def delete_vector(self, vid, kind):
        with self._lock:
            self._conn.execute("DELETE FROM vectors WHERE video_id=? AND kind=?", (vid, kind))
        self.version += 1

    def get_vectors(self, vid):
        with self._lock:
            rows = self._conn.execute("SELECT kind, dim, model, vec, created_at FROM vectors WHERE video_id=?", (vid,)).fetchall()
        return {r["kind"]: {"dim": r["dim"], "model": r["model"], "created_at": r["created_at"],
                            "vec": np.frombuffer(r["vec"], dtype=np.float16).astype(np.float32)} for r in rows}

    def all_vectors(self, kind):
        """返回 (video_ids, 矩阵)；只取 ready 的视频。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT v.video_id, v.vec FROM vectors v JOIN videos x ON x.id=v.video_id "
                "WHERE v.kind=? AND x.status='ready' ORDER BY x.created_at", (kind,)).fetchall()
        if not rows:
            return [], np.zeros((0, 768), dtype=np.float32)
        ids = [r["video_id"] for r in rows]
        M = np.stack([np.frombuffer(r["vec"], dtype=np.float16) for r in rows]).astype(np.float32)
        return ids, M

    # ---------- frames ----------
    def put_frames(self, vid, items):
        with self._lock:
            self._conn.execute("DELETE FROM frames WHERE video_id=?", (vid,))
            self._conn.executemany("INSERT INTO frames(video_id, idx, t_s, thumb) VALUES (?,?,?,?)",
                                   [(vid, i, float(t), th) for i, (t, th) in enumerate(items)])

    def get_frames(self, vid):
        with self._lock:
            rows = self._conn.execute("SELECT idx, t_s, thumb FROM frames WHERE video_id=? ORDER BY idx", (vid,)).fetchall()
        return [dict(r) for r in rows]

    # ---------- evalsets ----------
    def put_evalset(self, name, items, description=""):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO evalsets(name, description, items, created_at) VALUES (?,?,?,?)",
                               (name, description, json.dumps(items, ensure_ascii=False), time.time()))

    def get_evalset(self, name):
        with self._lock:
            r = self._conn.execute("SELECT * FROM evalsets WHERE name=?", (name,)).fetchone()
        return {"name": r["name"], "description": r["description"], "items": json.loads(r["items"])} if r else None

    def list_evalsets(self):
        with self._lock:
            rows = self._conn.execute("SELECT name, description, items, created_at FROM evalsets ORDER BY created_at").fetchall()
        return [{"name": r["name"], "description": r["description"], "n": len(json.loads(r["items"])),
                 "created_at": r["created_at"]} for r in rows]

    @staticmethod
    def _row(r):
        d = dict(r)
        for k in ("tags", "extra", "timings"):
            try:
                d[k] = json.loads(d.get(k) or ("[]" if k == "tags" else "{}"))
            except json.JSONDecodeError:
                pass
        return d
