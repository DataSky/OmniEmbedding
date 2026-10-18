"""Local-only media retrieval demo. Uses its own database and remote encoder.

Run via scripts/run_local_demo.py. Importing this module never imports a model.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import ipaddress
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
import io

REPO = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web/demo"


class RequestLimitMiddleware:
    """Bound request bytes before multipart parsing can spool a huge file."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"}:
            return await self.app(scope, receive, send)
        limit = 12 * 1024**2 if scope["path"] == "/api/demo/search/image" else 512 * 1024
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0:
                raise ValueError
        except ValueError:
            return await JSONResponse({"detail": "Content-Length无效"}, status_code=400)(scope, receive, send)
        if declared > limit:
            return await JSONResponse({"detail": "请求体超过本地服务上限"}, status_code=413)(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > limit:
                return await JSONResponse({"detail": "请求体超过本地服务上限"}, status_code=413)(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break
        position = 0

        async def replay():
            nonlocal position
            if position < len(messages):
                value = messages[position]
                position += 1
                return value
            return await receive()

        await self.app(scope, replay, send)


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=4096)
    name: str | None = Field(default=None, max_length=160)
    auto_embed: bool = False


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["embed"] = "embed"
    dataset_id: str | None = None
    asset_id: str | None = None
    force: bool = False


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=8192)
    mode: Literal["visual", "cover", "text", "keyword", "hybrid"] = "visual"
    top_k: int = Field(default=20, ge=1, le=100)
    dataset_id: str | None = None


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def create_app(*, data_dir=None, endpoint_file=None, key_file=None, engine=None):
    directory = Path(data_dir or os.environ.get("OMNI_DEMO_DATA_DIR", REPO / "data/local-demo")).resolve()
    if engine is None:
        from .demo_engine import DemoEngine
        engine = DemoEngine(
            data_dir=directory,
            endpoint_file=Path(endpoint_file or os.environ.get("OMNI_INFERENCE_ENDPOINT_FILE", REPO / "data/autodl/endpoint.json")),
            key_file=Path(key_file or os.environ.get("OMNI_INFERENCE_KEY_FILE", REPO / "data/autodl/inference.key")),
        )

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(title="OmniEmbedding · 素材检索实验室", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine
    app.add_middleware(RequestLimitMiddleware)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # Loopback binding + Host guard prevents using a DNS-rebinding origin to
        # read local files. Writes also require same-origin browser requests.
        if not _loopback(request.url.hostname):
            return JSONResponse({"detail": "该应用仅供本机访问"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            try:
                parsed = urlsplit(origin)
                same = parsed.scheme == request.url.scheme and parsed.netloc == request.headers.get("host")
            except ValueError:
                same = False
            if not same:
                return JSONResponse({"detail": "写操作只接受本应用页面的同源请求"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def invalid(_, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(_, error):
        return JSONResponse({"detail": "记录不存在"}, status_code=404)

    # The existing adapter already sanitizes transport failures and remote errors.
    from .remote_encoder import RemoteEncoderError

    @app.exception_handler(RemoteEncoderError)
    async def remote_error(_, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status_code)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/demo/")

    @app.get("/api/demo/overview")
    def overview():
        return engine.overview()

    @app.get("/api/demo/datasets")
    def datasets():
        return {"items": engine.datasets()}

    @app.post("/api/demo/import/preview")
    def preview(body: ImportRequest):
        return engine.preview(body.path)

    @app.post("/api/demo/import", status_code=202)
    def import_source(body: ImportRequest):
        return engine.submit_import(body.path, name=body.name, auto_embed=body.auto_embed)

    @app.get("/api/demo/jobs")
    def jobs():
        return {"items": engine.jobs()}

    @app.post("/api/demo/jobs", status_code=202)
    def submit_job(body: JobRequest):
        return engine.submit_embed(dataset_id=body.dataset_id, asset_id=body.asset_id, force=body.force)

    @app.get("/api/demo/jobs/{job_id}")
    def job(job_id: str):
        result = engine.job(job_id)
        if result is None:
            raise HTTPException(404, "任务不存在")
        return result

    @app.post("/api/demo/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return engine.cancel(job_id)

    @app.post("/api/demo/jobs/{job_id}/pause")
    def pause(job_id: str):
        return engine.pause(job_id)

    @app.post("/api/demo/jobs/{job_id}/resume", status_code=202)
    @app.post("/api/demo/jobs/{job_id}/retry", status_code=202)
    def resume(job_id: str):
        return engine.resume(job_id)

    @app.get("/demo-guide", include_in_schema=False)
    def guide():
        return FileResponse(REPO / "docs/LOCAL_DEMO.html")

    @app.get("/api/demo/assets")
    def assets(dataset_id: str | None = None, q: str = "", status: str = "", limit: int = 100, offset: int = 0):
        if not 1 <= limit <= 200 or offset < 0 or len(q) > 8192:
            raise HTTPException(400, "分页或检索参数无效")
        return engine.assets(dataset_id=dataset_id, q=q, status=status, limit=limit, offset=offset)

    @app.get("/api/demo/assets/{asset_id}")
    def asset(asset_id: str):
        result = engine.asset(asset_id)
        if result is None:
            raise HTTPException(404, "素材不存在")
        return result

    @app.get("/api/demo/assets/{asset_id}/vectors/{kind}")
    def vector(asset_id: str, kind: str):
        if kind not in {"cover", "visual", "text"}:
            raise HTTPException(400, "向量类型无效")
        result = engine.store.vector(asset_id, kind, include_values=True)
        if result is None:
            raise HTTPException(404, "该素材还没有此类向量")
        return result

    def safe_file(asset_id, kind, index=None):
        detail = engine.asset(asset_id)
        if detail is None:
            raise HTTPException(404, "素材不存在")
        if kind == "frame":
            frames = detail.get("frames", [])
            selected = next((x for i, x in enumerate(frames) if x.get("index", i) == index), None)
            value = selected.get("path") if selected else None
        else:
            value = detail.get(kind + "_path")
        if not value:
            raise HTTPException(404, "该素材没有此媒体文件")
        path = Path(value).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise HTTPException(404, "媒体文件不可用")
        allowed = {"cover": {".jpg", ".jpeg", ".png", ".webp"},
                   "video": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"},
                   "frame": {".jpg", ".jpeg", ".png", ".webp"}}
        if path.suffix.lower() not in allowed[kind]:
            raise HTTPException(415, "媒体格式不支持")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/demo/media/{asset_id}/cover")
    def cover(asset_id: str):
        return safe_file(asset_id, "cover")

    @app.get("/api/demo/media/{asset_id}/video")
    def video(asset_id: str):
        return safe_file(asset_id, "video")

    @app.get("/api/demo/media/{asset_id}/frame/{index}")
    def frame(asset_id: str, index: int):
        return safe_file(asset_id, "frame", index)

    @app.post("/api/demo/search")
    def search(body: SearchRequest):
        return engine.search(body.query, mode=body.mode, top_k=body.top_k, dataset_id=body.dataset_id)

    @app.post("/api/demo/search/image")
    async def image_search(file: UploadFile = File(...), mode: str = Form("visual"), top_k: int = Form(20),
                           dataset_id: str | None = Form(None)):
        if mode not in {"visual", "cover"} or not 1 <= top_k <= 100:
            raise HTTPException(400, "以图检索只支持画面或封面向量")
        contents = await file.read(10 * 1024**2 + 1)
        await file.close()
        if len(contents) > 10 * 1024**2:
            raise HTTPException(413, "查询图片最多10MiB")
        try:
            with Image.open(io.BytesIO(contents)) as source:
                if source.width * source.height > 16_000_000 or source.format not in {"JPEG", "PNG", "WEBP"} or getattr(source, "is_animated", False):
                    raise HTTPException(400, "请使用1600万像素以内的静态JPEG/PNG/WebP")
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, "查询图片无法解码") from None
        from starlette.concurrency import run_in_threadpool
        try:
            return await run_in_threadpool(engine.search_image, image, mode=mode, top_k=top_k, dataset_id=dataset_id)
        finally:
            image.close()

    @app.get("/api/demo/galaxy")
    def galaxy(kind: str = "visual", dataset_id: str | None = None):
        if kind not in {"visual", "cover", "text"}:
            raise HTTPException(400, "向量类型无效")
        return engine.galaxy(kind=kind, dataset_id=dataset_id)

    @app.get("/api/demo/traces")
    def traces():
        return {"items": engine.traces()}

    @app.get("/api/demo/traces/{trace_id}")
    def trace(trace_id: str):
        result = engine.trace(trace_id)
        if result is None:
            raise HTTPException(404, "追踪记录不存在")
        return result

    @app.get("/api/demo/health")
    def health():
        return engine.health(probe=False)

    @app.post("/api/demo/health/probe")
    def probe():
        return engine.health(probe=True)

    app.mount("/demo", StaticFiles(directory=str(WEB), html=True, check_dir=False), name="demo-web")
    return app
