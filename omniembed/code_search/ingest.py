"""Read-only, resumable source ingestion for the independent code search app.

Source documents are untrusted *data*, including files named AGENTS.md. This
module never imports, executes, or follows instructions found in the corpus.
"""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Iterator

import numpy as np

from ..config import DEFAULT_MODEL, PINNED_REVISION
from ..remote_encoder import DIMENSIONS, RemoteEncoder, RemoteEncoderError, _key


LOGGER = logging.getLogger(__name__)
MAX_FILE_BYTES = 1024 * 1024
MAX_CHUNK_CHARS = 2400
MAX_EMBEDDING_CHARS = 8192
MAX_BATCH_CHARS = 32768
MAX_BATCH_SIZE = 16
CHUNK_POLICY = "line-v1:2400:6"
TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "paused", "interrupted"})

# Exclude dependency/output trees structurally, rather than excluding useful
# source directories such as "config", "data", or "docs" wholesale.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".bzr", "node_modules", "bower_components",
    ".venv", "venv", ".conda", "site-packages", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", ".nox", ".cache", ".next",
    ".nuxt", ".output", ".turbo", ".parcel-cache", ".yarn", ".pnpm-store",
    "dist", "build", "target", "out", "coverage", "htmlcov", ".gradle",
    ".idea", ".vscode", ".DS_Store", ".terraform", ".serverless",
    ".aws", ".ssh", ".gnupg", ".config", ".kube", ".pio", ".cargo", ".rustup", "cache", "secrets",
    "credentials", "runtime", ".runtime", "logs", ".logs", "wandb", ".data",
    "modulecache.noindex", "deriveddata", "playwright-report", "test-results",
    ".build", ".swiftpm", ".svelte-kit", ".vite", "pods",
})
BINARY_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff",
    ".avif", ".heic", ".mp4", ".mov", ".mkv", ".avi", ".webm", ".mp3",
    ".wav", ".flac", ".ogg", ".m4a", ".pdf", ".doc", ".docx", ".xls",
    ".xlsx", ".ppt", ".pptx", ".zip", ".gz", ".tgz", ".tar", ".bz2",
    ".xz", ".7z", ".rar", ".zst", ".jar", ".war", ".class", ".pyc",
    ".pyo", ".so", ".dylib", ".dll", ".exe", ".o", ".a", ".lib",
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm",
    ".sqlite-wal", ".sqlite-shm", ".npy", ".npz", ".parquet", ".arrow",
    ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".safetensors", ".onnx",
    ".h5", ".hdf5", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".lock",
    ".DS_Store", ".ipynb",  # notebook outputs may contain secrets/binary data
})
SENSITIVE_NAMES = frozenset({
    "credentials", "credentials.json", "credentials.yaml", "credentials.yml",
    "credential.json", "token", "token.json", "tokens.json", "token.txt",
    "secrets.json", "secrets.yaml", "secrets.yml", "secrets.toml",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "known_hosts",
    ".npmrc", ".pypirc", ".netrc", ".git-credentials", ".bash_history",
    ".zsh_history", "inference.key", "client.env", "endpoint.json",
    "control.json", "runtime.json", "runtime.yaml", "runtime.yml",
})

# Match concrete credential values. Variable names and examples such as
# API_KEY=os.getenv("API_KEY") remain searchable; a matching real value causes
# the *whole file* to be withheld from both SQLite and remote inference.
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{15,}\b"),
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\b(?:authorization[\s\"']*[:=][\s\"']*bearer|bearer)\s+[A-Za-z0-9_.~+/=-]{18,}"),
    re.compile(r"(?im)\b(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|db[_-]?password|inference[_-]?key|aws[_-]?secret[_-]?access[_-]?key|aws[_-]?session[_-]?token|aws[_-]?security[_-]?token)\b[\s\"']*[:=]\s*[\"']([A-Za-z0-9+/=_\-.!@#$%^&*]{12,})[\"']"),
    re.compile(r"(?im)^\s*(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|db[_-]?password|inference[_-]?key|aws[_-]?secret[_-]?access[_-]?key|aws[_-]?session[_-]?token|aws[_-]?security[_-]?token)\s*[:=]\s*([A-Za-z0-9+/=_\-.!@#$%^&*]{16,})\s*(?:#.*)?$"),
)
PLACEHOLDER = re.compile(r"(?i)^(?:your[-_].*|example[-_].*|sample[-_].*|dummy[-_].*|replace[-_].*|change[-_]?me.*|[x*_-]+|test[-_].*|placeholder.*|\$\{.*)$")
URL_CREDENTIALS = re.compile(r"(?i)\b(?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://([^\r\n/@]{1,2048})@")
TEMPLATE_CREDENTIAL = re.compile(r"(?:\$\{\{[^\r\n{}]+\}\}|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*|\{\{[^\r\n{}]+\}\})")


class Cancelled(RuntimeError):
    """Cooperative cancellation at a file or remote-batch boundary."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def sensitive_content(text: str) -> bool:
    """Conservative credential detection without returning the matched value."""
    for match in URL_CREDENTIALS.finditer(text):
        username, separator, password = match.group(1).partition(":")
        if separator and username and password and not TEMPLATE_CREDENTIAL.fullmatch(password.strip()):
            return True
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if match.lastindex and PLACEHOLDER.match(match.group(match.lastindex)):
                continue
            return True
    return False


def diagnostic_path(path: str) -> str:
    """A credential-shaped filename must not itself leak through diagnostics."""
    return f"[redacted-path-{sha256(path)[:12]}]" if sensitive_content(path) else path


def safe_error(exc: BaseException) -> str:
    """Keep remote bodies, source snippets, keys and URLs out of UI/job logs."""
    if isinstance(exc, RemoteEncoderError):
        message = str(exc)
        message = re.sub(r"https?://\S+", "[远程地址]", message)
        if not sensitive_content(message):
            return message[:240]
    if isinstance(exc, Cancelled):
        return "任务已暂停；已入库内容保留，下次同步自动续跑"
    if isinstance(exc, PermissionError):
        return "没有读取文件或数据库的权限"
    if isinstance(exc, FileNotFoundError):
        return "文件在扫描过程中被移除"
    if isinstance(exc, UnicodeError):
        return "文件不是有效 UTF-8 文本"
    return f"操作失败（{type(exc).__name__}）；详细数据未写入错误信息"


def path_skip_reason(path: Path, *, directory: bool = False) -> str | None:
    name = path.name.lower()
    if (name.startswith(".env") or name.endswith((".pem", ".key", ".p12", ".pfx", ".keystore"))
            or name in SENSITIVE_NAMES
            or re.search(r"(?:^|[._-])(?:credentials?|secrets?|access[._-]?tokens?|auth[._-]?tokens?)(?:[._-]|$)", name)):
        return "sensitive_path"
    if directory:
        return "excluded_directory" if name in SKIP_DIRS or name.endswith((".xcresult", ".app", ".dsym")) else None
    if (path.suffix.lower() in BINARY_SUFFIXES or name.endswith((".min.js", ".min.css", ".map"))
            or name in {".ds_store", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "cargo.lock", "composer.lock"}
            or re.search(r"(?:^|[._-])(?:embed(?:ding)?|vector)[._-]?cache(?:[._-]|$)", name)):
        return "binary_or_generated"
    return None


def excluded_relative_path(path: str | Path) -> str | None:
    """Apply current structural rules to old SQLite paths before uploading.

    Root maintenance can use this helper to remove previously indexed runtime
    history, including OmniVoice/.data. Plain data/ directories remain allowed.
    """
    parts = Path(path).parts
    for part in parts[:-1]:
        reason = path_skip_reason(Path(part), directory=True)
        if reason:
            return reason
    return path_skip_reason(Path(parts[-1])) if parts else None


def chunk_text(text: str, path: str = "", *, max_chars: int = MAX_CHUNK_CHARS,
               overlap_lines: int = 6) -> list[dict]:
    """Split on source line boundaries, preserving exact line numbers.

    A very long single line remains one source line. Its chunk is permitted up
    to the remote 8192-character limit; longer lines are handled by scan() as
    unsupported rather than silently truncated or assigned invented line nums.
    """
    if max_chars < 1 or overlap_lines < 0:
        raise ValueError("invalid chunk limits")
    lines = text.splitlines(keepends=True)
    chunks, begin = [], 0
    while begin < len(lines):
        end, cost = begin, 0
        while end < len(lines):
            length = len(lines[end])
            if end > begin and cost + length > max_chars:
                break
            cost += length
            end += 1
            if cost >= max_chars:
                break
        content = "".join(lines[begin:end])
        if content.strip():
            # The path is context for embeddings and never changes the content
            # shown to a user at the stored source line range.
            header = f"File: {path}\nLines: {begin + 1}-{end}\n"
            if len(header) + len(content) > MAX_EMBEDDING_CHARS:
                raise ValueError("source_line_exceeds_embedding_limit")
            content_hash = sha256(content)
            chunks.append({"id": sha256(f"{path}\0{begin + 1}\0{end}\0{content_hash}"),
                           "ordinal": len(chunks), "start_line": begin + 1,
                           "end_line": end, "content": content,
                           "embedding_text": header + content, "sha256": content_hash})
        if end >= len(lines):
            break
        # Always make progress even if a single line exceeds the target size.
        candidate = max(begin + 1, end - overlap_lines)
        # Avoid overlap consuming almost the whole next target chunk.
        while candidate < end and sum(len(line) for line in lines[candidate:end]) >= max_chars // 2:
            candidate += 1
        begin = candidate
    return chunks


def _file_record(relative: str, info=None, *, status: str = "skipped", error: str | None = None) -> dict:
    display = diagnostic_path(relative)
    parts = Path(display).parts
    return {"id": sha256(relative), "path": display,
            "project": parts[0] if len(parts) > 1 else "(root)",
            "extension": Path(relative).suffix.lower(), "size": int(info.st_size) if info else 0,
            "mtime_ns": int(info.st_mtime_ns) if info else 0, "sha256": "",
            "status": status, "error": error, "chunk_count": 0,
            "content": "", "policy": CHUNK_POLICY, "updated_at": utc_now()}


def scan(root: Path, checkpoint: Callable[[], None] | None = None, *,
         max_file_bytes: int = MAX_FILE_BYTES, max_chars: int = MAX_CHUNK_CHARS,
         overlap_lines: int = 6) -> Iterator[dict]:
    """Yield safe source snapshots and explicit skip/error records.

    Paths are relative to the fixed root. File descriptors are opened relative
    to fwalk's directory descriptor with O_NOFOLLOW, so symlink targets cannot
    become corpus data. No source files are modified. checkpoint() is called at
    entry boundaries and may raise to pause scanning.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source root must be a directory")
    if max_file_bytes <= 0:
        raise ValueError("file size limit must be positive")
    traversal_errors: list[OSError] = []
    for directory, subdirs, names, dirfd in os.fwalk(root, topdown=True, onerror=traversal_errors.append, follow_symlinks=False):
        if checkpoint:
            checkpoint()
        relative_dir = Path(directory).relative_to(root)
        for name in sorted(list(subdirs)):
            if checkpoint:
                checkpoint()
            relative = (relative_dir / name).as_posix()
            try:
                info = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
                reason = "symlink" if stat.S_ISLNK(info.st_mode) else path_skip_reason(Path(name), directory=True)
            except OSError as exc:
                subdirs.remove(name)
                yield {"file": _file_record(relative, status="error", error=safe_error(exc)),
                       "chunks": [], "reason": "read_error", "is_dir": True}
                continue
            if reason:
                subdirs.remove(name)
                yield {"file": _file_record(relative, info, error=reason), "chunks": [], "reason": reason, "is_dir": True}
        subdirs.sort()
        for name in sorted(names):
            if checkpoint:
                checkpoint()
            relative = (relative_dir / name).as_posix()
            record = _file_record(relative)
            descriptor = None
            try:
                info = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
                record = _file_record(relative, info)
                reason = ("symlink" if stat.S_ISLNK(info.st_mode) else
                          "not_regular" if not stat.S_ISREG(info.st_mode) else path_skip_reason(Path(name)))
                if reason is None and info.st_size > max_file_bytes:
                    reason = "too_large"
                if reason:
                    record["error"] = reason
                    yield {"file": record, "chunks": [], "reason": reason, "is_dir": False}
                    continue
                descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), dir_fd=dirfd)
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = None
                    before = os.fstat(handle.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        reason = "not_regular"
                        raw = b""
                    elif before.st_size > max_file_bytes:
                        reason = "too_large"
                        raw = b""
                    else:
                        raw = handle.read(max_file_bytes + 1)
                    after = os.fstat(handle.fileno())
                if reason:
                    record["error"] = reason
                    yield {"file": record, "chunks": [], "reason": reason, "is_dir": False}
                    continue
                if len(raw) > max_file_bytes:
                    reason = "too_large"
                elif (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
                      or len(raw) != after.st_size):
                    reason = "changed_during_read"
                elif b"\x00" in raw:
                    reason = "binary_content"
                else:
                    try:
                        text = raw.decode("utf-8-sig", errors="strict")
                    except UnicodeDecodeError:
                        reason = "non_utf8"
                    else:
                        controls = sum(ord(c) < 32 and c not in "\n\r\t\f" for c in text)
                        if controls > max(2, len(text) // 1000):
                            reason = "binary_content"
                        elif sensitive_content(text):
                            reason = "sensitive_content"
                        else:
                            try:
                                chunks = chunk_text(text, relative, max_chars=max_chars, overlap_lines=overlap_lines)
                            except ValueError:
                                reason = "overlong_source_line"
                            else:
                                if any(sensitive_content(chunk["embedding_text"]) for chunk in chunks):
                                    record["error"] = "sensitive_embedding_text"
                                    yield {"file": record, "chunks": [], "reason": "sensitive_embedding_text", "is_dir": False}
                                    continue
                                record.update(size=after.st_size, mtime_ns=after.st_mtime_ns,
                                              sha256=sha256(raw), status="indexed", error=None,
                                              content=text, chunk_count=len(chunks))
                                yield {"file": record, "chunks": chunks, "reason": None, "is_dir": False}
                                continue
                record["error"] = reason
                yield {"file": record, "chunks": [], "reason": reason, "is_dir": False}
            except OSError as exc:
                record.update(status="error", error=safe_error(exc))
                yield {"file": record, "chunks": [], "reason": "read_error", "is_dir": False}
            finally:
                if descriptor is not None:
                    os.close(descriptor)
    for exc in traversal_errors:
        filename = Path(exc.filename) if exc.filename else root
        try:
            relative = filename.relative_to(root).as_posix()
        except ValueError:
            relative = "."
        yield {"file": _file_record(relative, status="error", error=safe_error(exc)),
               "chunks": [], "reason": "read_error", "is_dir": True}


class Engine:
    """One background ingestion job; completed vectors survive cancellation."""

    def __init__(self, root: Path, store, encoder=None, endpoint_file=None, key_file=None):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("source root must be a directory")
        self.store = store
        self.endpoint_file = Path(endpoint_file) if endpoint_file else Path(__file__).resolve().parents[2] / "data/autodl/endpoint.json"
        self.key_file = Path(key_file) if key_file else self.endpoint_file.parent / "inference.key"
        self.enc = encoder or RemoteEncoder(model_id=DEFAULT_MODEL, revision=PINNED_REVISION,
                                            endpoint_file=self.endpoint_file, key_file=self.key_file, timeout=45)
        if encoder is None:
            # RemoteEncoder supports a historical OMNI_INFERENCE_URL override;
            # corpus upload is deliberately bound only to explicit local files.
            self.enc._url = None
        self.enc_lock = threading.RLock()
        self._health_lock = threading.Lock()
        self._health_cache = None
        self._health_at = 0.0
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._job_id: str | None = None
        self._closing = False
        self._progress = {}
        self._details = {}
        self._skip_reasons = Counter()
        self._confirmed_files = deque()
        self._in_scan = False
        with self.store.connect() as conn:
            conn.execute("UPDATE jobs SET status='interrupted', phase='interrupted', finished_at=?, message=? WHERE status IN ('running','queued','cancelling')",
                         (utc_now(), "服务重启中断了任务；已入库内容保留，可重新同步续跑"))

    def _job(self, job_id: str | None = None) -> dict | None:
        if job_id:
            if hasattr(self.store, "job_get"):
                return self.store.job_get(job_id)
            with self.store.connect() as conn:
                row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                result = dict(row) if row else None
                if result and isinstance(result.get("details"), str):
                    result["details"] = json.loads(result["details"] or "{}")
                return result
        return self.store.job_current()

    def start_job(self, action: str = "sync") -> dict:
        if action not in {"scan", "embed", "sync"}:
            raise ValueError("任务类型必须为 scan、embed 或 sync")
        with self._lock:
            if self._closing:
                raise RuntimeError("服务正在停止")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("已有导入任务正在运行")
            self._cancel.clear()
            created = self.store.job_create(action)
            self._job_id = created["id"]
            self.store.job_update(self._job_id, status="running", phase="queued", started_at=utc_now(),
                                  message="任务已开始", details={"skip_reasons": {}, "skip_samples": [], "error_samples": [], "unchanged": 0})
            self._thread = threading.Thread(target=self._run, args=(self._job_id, action), daemon=True,
                                            name="code-search-ingestion")
            self._thread.start()
            return self._job(self._job_id) or created

    def cancel_job(self) -> dict | None:
        with self._lock:
            current = self._job(self._job_id) if self._job_id else self._job()
            # A live thread may already have persisted its final result. The
            # same lock also guards worker finalization, closing that race.
            if current and current.get("status") in TERMINAL_STATUSES:
                return current
            if self._thread is not None and self._thread.is_alive() and current:
                self._cancel.set()
                self.store.job_update(self._job_id, status="cancelling", message="正在暂停；当前远程请求结束后保留进度")
                return self._job(self._job_id)
            return current

    def pause_sync(self) -> dict | None:
        """Pause at the next file/batch boundary, keeping committed vectors."""
        return self.cancel_job()

    def _checkpoint(self):
        if self._cancel.is_set():
            raise Cancelled()

    def _run(self, job_id: str, action: str):
        self._progress = {key: 0 for key in ("files_done", "files_total", "chunks_done", "chunks_total",
                                             "vectors_done", "vectors_total", "skipped", "errors")}
        self._details = {"skip_reasons": {}, "skip_samples": [], "error_samples": [], "unchanged": 0,
                         "excluded_directories": 0, "changed": 0, "scan_complete": False,
                         "interleaved": action == "sync", "batches_succeeded": 0, "batches_failed": 0,
                         "last_batch": None, "last_work_phase": "scan" if action != "embed" else "embed"}
        self._skip_reasons = Counter()
        self._confirmed_files = deque()
        self._in_scan = False
        try:
            if action in {"embed", "sync"}:
                self._progress["vectors_total"] = self._pending_count()
            if action in {"scan", "sync"}:
                self._scan(job_id, interleave=action == "sync")
            self._checkpoint()
            if action in {"embed", "sync"}:
                deferred = action == "sync" and self._details.get("prune_deferred", False)
                self._embed(job_id, confirmed_only=deferred)
                if deferred:
                    self._details["deferred_unconfirmed"] = self._pending_count()
            self._checkpoint()
        except Cancelled as exc:
            with self._lock:
                self._persist(job_id, status="paused", phase="paused", finished_at=utc_now(), message=safe_error(exc))
        except Exception as exc:
            # Persist only sanitized diagnostics. Existing committed batches
            # are the durable checkpoint for a later embed/sync job.
            message = safe_error(exc)
            LOGGER.error("Code search ingestion failed: %s", message)
            self._progress["errors"] += 1
            with self._lock:
                self._persist(job_id, status="failed", phase="failed", finished_at=utc_now(),
                              message=message + "；已完成的向量保留，可重试续跑")
        else:
            with self._lock:
                if self._cancel.is_set():
                    self._persist(job_id, status="paused", phase="paused", finished_at=utc_now(), message=safe_error(Cancelled()))
                else:
                    errors = self._progress["errors"]
                    self._persist(job_id, status="partial" if errors else "completed", phase="completed",
                                  finished_at=utc_now(), message=f"任务完成，存在 {errors} 项错误；请查看任务详情" if errors else "任务完成")

    def _persist(self, job_id: str, **fields):
        self._details["skip_reasons"] = dict(self._skip_reasons)
        return self.store.job_update(job_id, **{**self._progress, "details": self._details, **fields})

    def _phase(self, job_id: str, phase: str, message: str):
        self._details["last_work_phase"] = phase
        self._persist(job_id, phase=phase, message=message)

    def _pending_count(self, file_id: str | None = None) -> int:
        with self.store.connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks c LEFT JOIN vectors v ON c.id=v.chunk_id "
                                "AND v.model=? AND v.revision=? AND v.dimensions=? WHERE v.chunk_id IS NULL" +
                                (" AND c.file_id=?" if file_id else ""),
                                (DEFAULT_MODEL, PINNED_REVISION, DIMENSIONS, *([file_id] if file_id else []))).fetchone()[0]

    def _record_error(self, job_id: str, path: str, reason: str, is_dir: bool = False):
        path = diagnostic_path(path)
        sample = {"path": path, "reason": reason, "directory": is_dir}
        if len(self._details["error_samples"]) < 200:
            self._details["error_samples"].append(sample)
        if hasattr(self.store, "error_record"):
            self.store.error_record(job_id, path, reason, is_dir=is_dir)

    def _note_skip(self, path: str, reason: str, is_dir: bool = False):
        self._skip_reasons[reason] += 1
        if len(self._details["skip_samples"]) < 80:
            self._details["skip_samples"].append({"path": diagnostic_path(path), "reason": reason, "directory": is_dir})

    def _scan(self, job_id: str, interleave: bool = False):
        self._in_scan = True
        self._phase(job_id, "scan", "正在只读扫描并分块；同步任务将交错执行有界向量批次")
        with self.store.connect() as conn:
            previous = {row["id"]: dict(row) for row in conn.execute("SELECT id,path,sha256,status,chunk_count,policy FROM files")}
        seen: set[str] = set()
        had_traversal_error = False
        slice_files = slice_chunks = 0
        slice_started = progress_at = time.monotonic()
        for entry in scan(self.root, self._checkpoint):
            record, chunks, reason = entry["file"], entry["chunks"], entry["reason"]
            record["policy"] = CHUNK_POLICY
            if entry["is_dir"]:
                if record["status"] == "error":
                    self._progress["errors"] += 1
                    had_traversal_error = True
                    self._record_error(job_id, record["path"], reason or "read_error", is_dir=True)
                else:
                    self._details["excluded_directories"] += 1
                self._note_skip(record["path"], reason, is_dir=True)
                continue
            seen.add(record["path"])
            old = previous.get(record["id"])
            if (record["status"] == "indexed" and old and old.get("policy") == CHUNK_POLICY
                    and old["status"] == "indexed" and old["sha256"] == record["sha256"]):
                self.store.patch_file(record)
                self._details["unchanged"] += 1
            else:
                old_pending = self._pending_count(record["id"]) if old and interleave else 0
                self.store.replace_file(record, chunks)
                if interleave:
                    self._progress["vectors_total"] += len(chunks) - old_pending
                self._details["changed"] += 1
                slice_chunks += len(chunks)
            if record["status"] == "indexed" and chunks:
                self._confirmed_files.append(record["id"])
            self._progress["files_done"] += 1
            self._progress["files_total"] = self._progress["files_done"]
            self._progress["chunks_done"] += len(chunks)
            self._progress["chunks_total"] = self._progress["chunks_done"]
            slice_files += 1
            if reason:
                self._note_skip(record["path"], reason)
                if record["status"] == "error":
                    self._progress["errors"] += 1
                    self._record_error(job_id, record["path"], reason or "read_error")
                else:
                    self._progress["skipped"] += 1
            now = time.monotonic()
            if interleave and (slice_chunks >= 128 or slice_files >= 100 or now - slice_started >= 2):
                self._embed(job_id, max_batches=1, confirmed_only=True)
                self._checkpoint()
                self._phase(job_id, "scan", "继续只读扫描；已完成向量批次保留")
                slice_files = slice_chunks = 0
                slice_started = time.monotonic()
            if self._progress["files_done"] % 10 == 0 or now - progress_at >= 2:
                self._persist(job_id, message=f"已扫描 {self._progress['files_done']} 个文件，分块 {self._progress['chunks_done']} 个")
                progress_at = now
        self._checkpoint()
        # If a directory could not be read, its unseen descendants may still
        # exist. Retain them until a fully readable scan establishes deletion.
        if not had_traversal_error:
            self.store.prune_files(seen)
        self._details["prune_deferred"] = had_traversal_error
        self._details["scan_complete"] = True
        self.store.meta_set("chunk_policy", CHUNK_POLICY)
        self._details["skip_reasons"] = dict(self._skip_reasons)
        self.store.meta_set("scan_summary", {**self._details, "files": self._progress["files_done"],
                                             "chunks": self._progress["chunks_done"], "skipped": self._progress["skipped"],
                                             "errors": self._progress["errors"], "scanned_at": utc_now()})
        if interleave:
            self._progress["vectors_total"] = self._progress["vectors_done"] + self._pending_count()
        self._persist(job_id, message=f"扫描完成：{self._progress['files_done']} 个文件，{self._progress['chunks_done']} 个分块")
        self._in_scan = False
        self.store.mark_dirty()

    def _confirmed_pending(self) -> list[dict]:
        """During scanning, upload only files already checked in this run."""
        result, inspected = [], 0
        visited = set()
        with self.store.connect() as conn:
            while self._confirmed_files and len(result) < MAX_BATCH_SIZE and inspected < 100:
                file_id = self._confirmed_files[0]
                if file_id in visited:
                    break
                visited.add(file_id)
                self._confirmed_files.popleft()
                remaining = MAX_BATCH_SIZE - len(result)
                rows = conn.execute("SELECT c.*,f.path FROM chunks c JOIN files f ON f.id=c.file_id "
                                    "LEFT JOIN vectors v ON c.id=v.chunk_id AND v.model=? AND v.revision=? AND v.dimensions=? "
                                    "WHERE c.file_id=? AND v.chunk_id IS NULL ORDER BY c.ordinal LIMIT ?",
                                    (DEFAULT_MODEL, PINNED_REVISION, DIMENSIONS, file_id, remaining)).fetchall()
                result.extend(dict(row) for row in rows)
                inspected += 1
                # Retain a file until a later query proves it has no pending
                # rows. A character-limited batch may not consume every row
                # just fetched, even when len(rows) is smaller than remaining.
                if rows:
                    self._confirmed_files.append(file_id)
        return result

    def _quarantine(self, job_id: str, chunk: dict, reason: str):
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE id=?", (chunk["file_id"],)).fetchone()
        if row is None:
            return
        record = dict(row)
        pending = self._pending_count(record["id"])
        record.update(path=diagnostic_path(record["path"]), status="skipped", error=reason,
                      content="", chunk_count=0, policy=CHUNK_POLICY)
        self.store.replace_file(record, [])
        self._progress["vectors_total"] = max(self._progress["vectors_done"], self._progress["vectors_total"] - pending)
        self._progress["skipped"] += 1
        if reason == "invalid_embedding_text":
            self._progress["errors"] += 1
        self._details["quarantined_files"] = self._details.get("quarantined_files", 0) + 1
        self._note_skip(record["path"], reason)
        self._record_error(job_id, record["path"], reason)

    def _embed(self, job_id: str, max_batches: int | None = None, confirmed_only: bool = False) -> int:
        self._phase(job_id, "embed", "正在使用 AutoDL 向量化；每次仅提交有界批次")
        done = rounds = 0
        while max_batches is None or rounds < max_batches:
            self._checkpoint()
            pending = self._confirmed_pending() if confirmed_only else self.store.pending_chunks(DEFAULT_MODEL, PINNED_REVISION, limit=MAX_BATCH_SIZE)
            if not pending:
                if confirmed_only and self._confirmed_files:
                    rounds += 1
                    continue
                break
            rounds += 1
            rejected = set()
            safe = []
            for chunk in pending:
                if chunk["file_id"] in rejected:
                    continue
                text = chunk.get("embedding_text")
                reason = excluded_relative_path(chunk.get("path", ""))
                if reason is None and (not isinstance(text, str) or not text.strip() or len(text) > MAX_EMBEDDING_CHARS):
                    reason = "invalid_embedding_text"
                if reason is None and sensitive_content(text):
                    reason = "sensitive_embedding_text"
                if reason:
                    rejected.add(chunk["file_id"])
                    self._quarantine(job_id, chunk, reason)
                else:
                    safe.append(chunk)
            batch, cost = [], 0
            for chunk in safe:
                if chunk["file_id"] in rejected:
                    continue
                text = chunk["embedding_text"]
                if batch and cost + len(text) > MAX_BATCH_CHARS:
                    break
                batch.append(chunk)
                cost += len(text)
            if not batch:
                self._persist(job_id, message="已隔离不符合规则的历史分块；继续处理其它文件")
                continue
            texts = [chunk["embedding_text"] for chunk in batch]
            # RemoteEncoder already bounds 429 retries. Replaying arbitrary
            # timeouts/500s here could spend GPU again on an accepted request;
            # leave the failed batch pending for an explicit resumable job.
            started = time.perf_counter()
            audit = {"input_hashes": [{"chunk_id": chunk["id"], "sha256": sha256(chunk["embedding_text"])} for chunk in batch],
                     "items": len(batch), "characters": cost, "phase": "scan" if self._in_scan else "embed",
                     "started_at": utc_now()}
            try:
                with self.enc_lock:
                    vectors = np.asarray(self.enc.docs([""] * len(texts), texts, bs=MAX_BATCH_SIZE), dtype=np.float32)
                norms = np.linalg.norm(vectors, axis=1) if vectors.ndim == 2 else np.asarray([])
                if (vectors.shape != (len(batch), DIMENSIONS) or not np.isfinite(vectors).all()
                        or not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=0.05)):
                    raise RemoteEncoderError("远程向量校验失败（维度、数值或单位向量异常）")
            except Exception as exc:
                failed = {**audit, "status": "failed", "http_status": getattr(exc, "http_status", None),
                          "elapsed_ms": round((time.perf_counter()-started)*1000, 2),
                          "error_code": type(exc).__name__, "boot_id": getattr(self.enc, "boot_id", None),
                          "finished_at": utc_now()}
                if hasattr(self.store, "batch_record"):
                    self.store.batch_record(job_id, **failed)
                self._details["batches_failed"] += 1
                self._details["last_batch"] = failed
                raise
            # A cancellation that arrives during a successful request is
            # honored *after* committing that completed, billable batch.
            items = [(chunk["id"], vector) for chunk, vector in zip(batch, vectors)]
            success = {**audit, "status": "success", "http_status": 200, "elapsed_ms": round((time.perf_counter()-started)*1000, 2),
                       "error_code": None, "boot_id": getattr(self.enc, "boot_id", None), "finished_at": utc_now()}
            if hasattr(self.store, "vector_put_many_with_audit"):
                self.store.vector_put_many_with_audit(items, DEFAULT_MODEL, PINNED_REVISION, job_id=job_id, batch=success)
            elif hasattr(self.store, "vector_put_many"):
                self.store.vector_put_many(items, DEFAULT_MODEL, PINNED_REVISION)
                if hasattr(self.store, "batch_record"):
                    self.store.batch_record(job_id, **success)
            else:
                for chunk_id, vector in items:
                    self.store.vector_put(chunk_id, vector, DEFAULT_MODEL, PINNED_REVISION)
                if hasattr(self.store, "batch_record"):
                    self.store.batch_record(job_id, **success)
            self._details["batches_succeeded"] += 1
            self._details["last_batch"] = success
            done += len(batch)
            self._progress["vectors_done"] += len(batch)
            self._persist(job_id, message=f"AutoDL 本任务已完成 {self._progress['vectors_done']}/{self._progress['vectors_total']} 个待处理分块")
            self.store.mark_dirty()
            # A short cooperative gap lets interactive queries acquire the
            # shared inference lock between background batches.
            self._cancel.wait(0.01)
        return done

    def health(self) -> dict:
        """Cached authenticated readiness GET; never trigger GPU inference."""
        configured = self.endpoint_file.is_file() and self.key_file.is_file()
        remote = {"ready": False, "status": "unconfigured", "message": "请配置 AutoDL 端点与私有密钥文件"}
        if configured and isinstance(self.enc, RemoteEncoder):
            with self._health_lock:
                if self._health_cache is None or time.monotonic() - self._health_at >= 3:
                    try:
                        request = urllib.request.Request(self.enc._endpoint() + "/health/ready", headers={
                            "Authorization": "Bearer " + _key(self.key_file),
                            "Accept": "application/json", "User-Agent": "omniembedding-code-search/1"})
                        try:
                            response = self.enc._opener.open(request, timeout=2)
                        except urllib.error.HTTPError as exc:
                            response = exc
                        with response:
                            status = response.code
                            if status not in {200, 503}:
                                raise RemoteEncoderError(f"远程健康检查失败（HTTP {status}）")
                            raw = response.read(65537)
                        if len(raw) > 65536:
                            raise RemoteEncoderError("远程健康响应超过大小限制")
                        value = json.loads(raw)
                        if (not isinstance(value, dict) or value.get("model") != DEFAULT_MODEL
                                or value.get("model_revision") != PINNED_REVISION
                                or type(value.get("ready")) is not bool
                                or value.get("status") not in {"ok", "loading", "degraded", "stalled"}):
                            raise RemoteEncoderError("远程健康响应无效或模型版本不匹配")
                        ready = status == 200 and value["ready"] and value["status"] == "ok"
                        remote = {"ready": ready, "status": value["status"],
                                  "message": "AutoDL 推理服务已就绪" if ready else "AutoDL 推理服务尚未就绪"}
                    except Exception as exc:
                        remote = {"ready": False, "status": "unavailable", "message": safe_error(exc)}
                    self._health_cache, self._health_at = remote, time.monotonic()
                remote = dict(self._health_cache)
        elif configured:
            remote = {"ready": False, "status": "configured", "message": "已配置自定义编码器；未进行远程健康检查"}
        return {"backend": "remote", "provider": "AutoDL", "model": DEFAULT_MODEL,
                "revision": PINNED_REVISION, "dimensions": DIMENSIONS,
                "configured": configured, **remote,
                "loaded": bool(getattr(self.enc, "loaded", False)),
                "last_success_at": getattr(self.enc, "last_success_at", None),
                "last_error": safe_error(RemoteEncoderError(getattr(self.enc, "last_error", ""))) if getattr(self.enc, "last_error", None) else None,
                "active_job": self._job()}

    def shutdown(self):
        with self._lock:
            self._closing = True
            self._cancel.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            # The remote client has a 45-second deadline. Let the current
            # successful billable batch commit before the daemon exits.
            thread.join(timeout=55)
