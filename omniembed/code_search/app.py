"""A separate loopback-only Web service; source files are never served by path."""
from __future__ import annotations

from contextlib import asynccontextmanager
import ipaddress
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ..demo_app import RequestLimitMiddleware
from .search import Search
from .store import DIMENSIONS, MODEL, REVISION, Store

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parents[1] / "web/code_search"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    candidates: int = Field(default=100, ge=1, le=1000)
    vector_weight: float = Field(default=.6, ge=0, le=1, allow_inf_nan=False)
    project: str | None = Field(default=None, max_length=512)
    extension: str | None = Field(default=None, max_length=64)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["scan", "embed", "sync"]


def loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(*, root=None, data_dir=None, endpoint_file=None, key_file=None, engine=None):
    root = Path(root or Path.home()/"AiCode").resolve()
    data_dir = Path(data_dir or ROOT/"data/code-search").resolve()
    if engine is None:
        from .ingest import Engine
        store = Store(data_dir/"code-search.sqlite3", root=root)
        engine = Engine(root=root, store=store,
                        endpoint_file=endpoint_file or ROOT/"data/autodl/endpoint.json",
                        key_file=key_file or ROOT/"data/autodl/inference.key")
    else:
        store = engine.store
    search = Search(store, engine)

    @asynccontextmanager
    async def lifespan(app):
        yield
        engine.shutdown()

    app = FastAPI(title="OmniIndex · 本地混合检索", version="1.0", lifespan=lifespan)
    app.state.engine, app.state.store = engine, store
    app.add_middleware(RequestLimitMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Never echo a rejected body: it may contain NaN (not JSON serializable)
        # or user data. Report only the location and human-readable constraint.
        return JSONResponse({"detail":[{"loc":list(e["loc"]), "type":e["type"], "msg":e["msg"]}
                                       for e in exc.errors()]}, status_code=422)

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        # Reject DNS rebinding and cross-origin requests, including reads of
        # source snapshots. The launcher itself only binds to 127.0.0.1.
        try:
            host = urlsplit("http://"+request.headers.get("host", "")).hostname
            origin = request.headers.get("origin")
            origin_parts = urlsplit(origin) if origin else None
            origin_ok = not origin or (origin_parts.scheme == "http" and loopback(origin_parts.hostname or "") and origin_parts.netloc == request.headers.get("host"))
            host_ok = loopback(host or "")
        except ValueError:
            host_ok = origin_ok = False
        if not host_ok or not origin_ok or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail":"此服务仅允许从本机同源页面访问。"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.get("/")
    def index():
        return FileResponse(WEB/"index.html")

    app.mount("/code-static", StaticFiles(directory=WEB, check_dir=False), name="code-static")

    @app.get("/api/code/overview")
    def overview():
        try:
            summary = json.loads(store.meta_get("scan_summary", "{}"))
        except (TypeError, ValueError):
            summary = {}
        return {**store.stats(), "root":str(root), "model":MODEL, "revision":REVISION, "dimensions":DIMENSIONS,
                "job":store.job_current(), "exclusions":["依赖与版本库", "构建输出、缓存与锁文件", "密钥及疑似凭据", "符号链接、二进制与超过 1 MiB 的文件"],
                "scan_summary":summary if isinstance(summary, dict) else {}}

    @app.get("/api/code/health")
    def health():
        return engine.health()

    @app.get("/api/code/files")
    def files(project: str | None=Query(default=None, max_length=512), q: str=Query(default="", max_length=512),
              limit: int=Query(default=50, ge=1, le=200), offset: int=Query(default=0, ge=0)):
        return store.list_files(project, q, limit, offset)

    @app.get("/api/code/files/{ident}")
    def file(ident: str):
        result = store.file_detail(ident)
        if result is None:
            raise HTTPException(404, "文件不存在")
        return result

    @app.get("/api/code/chunks/{ident}")
    def chunk(ident: str):
        result = store.chunk_detail(ident)
        if result is None:
            raise HTTPException(404, "分块不存在")
        return result

    @app.post("/api/code/search")
    def do_search(params: SearchRequest):
        try:
            return search.run(**params.model_dump())
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/code/jobs")
    def jobs():
        return {"job":store.job_current(), "recent":store.job_recent()}

    @app.post("/api/code/jobs")
    def submit(params: JobRequest):
        try:
            return engine.start_job(params.action)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/api/code/jobs/{ident}")
    def job_detail(ident: str):
        result = store.job_detail(ident)
        if result is None:
            raise HTTPException(404, "任务不存在")
        return result

    @app.get("/api/code/batches")
    def batches(job_id: str | None=Query(default=None, max_length=128), limit: int=Query(default=30, ge=1, le=200)):
        return {"items":store.list_batches(job_id, limit)}

    @app.get("/api/code/exclusions")
    def exclusions(q: str=Query(default="", max_length=512), reason: str | None=Query(default=None, max_length=128),
                   project: str | None=Query(default=None, max_length=512), limit: int=Query(default=50, ge=1, le=200),
                   offset: int=Query(default=0, ge=0)):
        return store.exclusions(q, reason, project, limit, offset)

    @app.post("/api/code/jobs/cancel")
    def cancel():
        try:
            return engine.cancel_job()
        except (ValueError, RuntimeError) as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/api/code/traces")
    def traces(limit: int=Query(default=20, ge=1, le=100)):
        return store.traces(limit)

    def evaluation_entries():
        directory = data_dir / "evaluations"
        if directory.is_symlink() or not directory.is_dir():
            return []
        roots = sorted((p for p in directory.iterdir() if p.is_dir() and not p.is_symlink()),
                       key=lambda p:p.name, reverse=True)[:60]
        entries = []
        for folder in roots:
            entries.append((folder.name, folder))
            for child in sorted(folder.glob("trial-*"))[:10]:
                if child.is_dir() and not child.is_symlink():
                    entries.append((folder.name + "__" + child.name, child))
        return entries

    def evaluation_json(path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, UnicodeError, ValueError):
            return {}

    def evaluation_record(ident, folder, detail=False):
        value = {}
        for filename in ("manifest.json", "summary.json", "recommendation.json", "failure.json"):
            value.update(evaluation_json(folder/filename))
        events = []
        event_path = folder/("stages.jsonl" if (folder/"stages.jsonl").is_file() else "events.jsonl")
        if not event_path.is_symlink() and event_path.is_file() and event_path.stat().st_size <= 4 * 1024 * 1024:
            try:
                for line in event_path.read_text(encoding="utf-8").splitlines()[-100:]:
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict):
                            events.append(event)
                    except ValueError:
                        continue
            except (OSError, UnicodeError):
                pass
        if not value and not events:
            return None
        keys = {"label", "scope", "parameters", "summary", "selected_metrics", "gate", "snapshot", "snapshot_id", "algorithm_sha256", "fixture_sha256"}
        record = {k:v for k,v in value.items() if k in keys}
        record["summary"] = value.get("summary") or value.get("selected_metrics") or {}
        fixture_metadata = evaluation_json(folder/"fixture.json")
        record["scope"] = {"description":value.get("scope") or fixture_metadata.get("scope"),
                           "snapshot":value.get("snapshot", value.get("snapshot_id")),
                           "benchmark_subset":value.get("benchmark_subset", False),
                           "whole_ai_code_quality":value.get("whole_ai_code_quality", False)}
        record.update(id=ident, status=value.get("status", "completed" if value.get("summary") else "running"),
                      created_at=value.get("created_at") or (events[0].get("at", events[0].get("time")) if events else None))
        if detail:
            record.update(cases=value.get("cases", []), events=events)
        return record

    @app.get("/api/code/evaluations")
    def evaluations():
        if (data_dir/"evaluations").is_symlink():
            return {"items":[], "latest":{}}
        items = [value for ident, folder in evaluation_entries() if (value := evaluation_record(ident, folder))]
        latest = evaluation_json(data_dir/"evaluations/latest.json")
        return {"items":items, "latest":latest}

    @app.get("/api/code/evaluations/{ident}")
    def evaluation_detail(ident: str):
        # Resolve only enumerated local runs; user input is never a file path.
        folders = dict(evaluation_entries())
        if ident not in folders or (value := evaluation_record(ident, folders[ident], detail=True)) is None:
            raise HTTPException(404, "评测记录不存在")
        return value

    return app
