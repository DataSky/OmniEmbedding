"""Small, authenticated Colab inference service; no SQLite or ingestion worker.

Run with ``uvicorn omniembed.inference:create_app --factory --workers 1``.
The lifespan starts model loading in a worker thread so liveness remains useful.
Only a completed, validated model warmup makes readiness succeed. A deep probe
can recover readiness after a transient inference failure.
"""
from __future__ import annotations

import asyncio
import hmac
import io
import hashlib
import json
import logging
import os
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Callable, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool

from .health import InferenceError, InferenceInputError

LOGGER = logging.getLogger("omniembed.inference")
SAFE_FAILURE_CATEGORIES = frozenset({"InferenceError", "InferenceOOM", "InferenceInputError"})
SAFE_EXCEPTION_TYPES = frozenset({"InferenceError", "InferenceInputError", "RuntimeError", "ValueError",
                                 "TypeError", "OutOfMemoryError", "InvalidEmbeddingOutput"})


class InvalidEmbeddingOutput(ValueError):
    """Invalid model output remains a model-health failure, never an input error."""


def _safe_failure(exc):
    name = type(exc).__name__
    category = exc.category if isinstance(exc, InferenceError) else name
    if category not in SAFE_FAILURE_CATEGORIES and category not in SAFE_EXCEPTION_TYPES:
        category = "InferenceError"
    cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else None
    return {"category": category,
            "exception_type": name if name in SAFE_EXCEPTION_TYPES else "OtherError",
            "cause_type": cause if cause in SAFE_EXCEPTION_TYPES else ("OtherError" if cause else None),
            "reason_code": exc.reason_code if isinstance(exc, InferenceInputError) else None,
            "failure_stage": "processor_validation" if isinstance(exc, InferenceInputError) else
                             ("vector_validation" if isinstance(exc, InvalidEmbeddingOutput) else "model_operation")}

BASE_DIMENSIONS = 768
MAX_BATCH_SIZE = 16
MAX_TEXT_CHARS = 8192
MAX_TITLE_CHARS = 1024
MAX_TOTAL_TEXT_CHARS = 32768
MAX_JSON_BYTES = 256 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_VIDEO_FRAMES = 8
MAX_VIDEO_BYTES = 32 * 1024 * 1024
MAX_VIDEO_MULTIPART_BYTES = MAX_VIDEO_BYTES + 64 * 1024
MAX_VIDEO_PIXELS = 32_000_000
PROBE_TEXT = "A small embedding health check."
Dimension = Literal[128, 256, 512, 768]


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: StrictStr | list[StrictStr]
    input_type: Literal["query", "document"] = "query"
    dimensions: Dimension = 768
    titles: list[StrictStr] | None = None

    @model_validator(mode="after")
    def bounded_input(self):
        texts = [self.input] if isinstance(self.input, str) else self.input
        if not 1 <= len(texts) <= MAX_BATCH_SIZE:
            raise ValueError(f"input must contain 1 to {MAX_BATCH_SIZE} texts")
        if any(len(text) > MAX_TEXT_CHARS for text in texts):
            raise ValueError(f"each text must not exceed {MAX_TEXT_CHARS} characters")
        if self.input_type == "query" and "titles" in self.model_fields_set:
            raise ValueError("titles are only supported for document input")
        titles = self.titles if self.titles is not None else [""] * len(texts)
        if len(titles) != len(texts):
            raise ValueError("titles must have the same length as input")
        if any(len(title) > MAX_TITLE_CHARS for title in titles):
            raise ValueError(f"each title must not exceed {MAX_TITLE_CHARS} characters")
        if any(not text.strip() and not title.strip() for title, text in zip(titles, texts)):
            raise ValueError("each input must contain nonempty text or a document title")
        if sum(map(len, texts)) + sum(map(len, titles)) > MAX_TOTAL_TEXT_CHARS:
            raise ValueError(f"total title and text length must not exceed {MAX_TOTAL_TEXT_CHARS} characters")
        return self


def _decode_image(content: bytes, maximum_pixels: int = MAX_IMAGE_PIXELS):
    """Fully decode bounded still images before entering model inference."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(415, "use a JPEG, PNG or WebP image")
                if image.width * image.height > maximum_pixels:
                    raise HTTPException(413, "image pixel limit exceeded")
                if getattr(image, "is_animated", False):
                    raise HTTPException(415, "animated images are not supported")
                return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "invalid image") from None
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, "image pixel limit exceeded") from None


def _validated_vectors(raw, rows: int, dimensions: int = BASE_DIMENSIONS):
    """Reject poisoned or incorrectly shaped model output before serialization."""
    vectors = np.asarray(raw, dtype=np.float32)
    if vectors.shape != (rows, BASE_DIMENSIONS):
        raise InvalidEmbeddingOutput("unexpected embedding shape")
    if not np.isfinite(vectors).all():
        raise InvalidEmbeddingOutput("non-finite embeddings")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=0.05):
        raise InvalidEmbeddingOutput("embeddings are not finite, nonzero unit vectors")
    vectors = vectors[:, :dimensions]
    truncated_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(truncated_norms).all() or (truncated_norms <= 1e-12).any():
        raise InvalidEmbeddingOutput("zero or non-finite truncated embeddings")
    return vectors / truncated_norms


class InferenceState:
    def __init__(self, encoder):
        self.encoder = encoder
        self.boot_id = str(uuid.uuid4())
        self.started = time.monotonic()
        self.ready = False
        self.warming = True
        self.last_success = None
        self.last_error = None
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.busy_since = None
        self.inference_timeout_s = float(os.environ.get("OMNI_INFERENCE_TIMEOUT_S", "120"))
        self.startup_timeout_s = float(os.environ.get("OMNI_MODEL_STARTUP_TIMEOUT_S", "900"))
        if (not np.isfinite(self.inference_timeout_s) or self.inference_timeout_s <= 0
                or not np.isfinite(self.startup_timeout_s) or self.startup_timeout_s <= 0):
            raise ValueError("inference and startup timeouts must be positive finite seconds")

    def snapshot(self):
        with self.state_lock:
            busy_for_s = 0 if self.busy_since is None else time.monotonic() - self.busy_since
            timeout = self.startup_timeout_s if self.warming else self.inference_timeout_s
            stalled = self.busy_since is not None and busy_for_s > timeout
            return {
                "status": "stalled" if stalled else ("ok" if self.ready else ("loading" if self.warming else "degraded")),
                "ready": self.ready and not stalled,
                "boot_id": self.boot_id,
                "model": self.encoder.model_id,
                "model_revision": getattr(self.encoder, "revision", None),
                "text_input_policy": getattr(self.encoder, "text_input_policy", None),
                "device": str(getattr(self.encoder, "device", "unknown")),
                "dtype": str(getattr(self.encoder, "dtype", "unknown")),
                "uptime_s": round(time.monotonic() - self.started, 3),
                "last_success": self.last_success,
                # Exception messages can contain prompts, URLs or credentials.
                # Expose only the exception class; never a traceback or message.
                "last_error": "InferenceStalled" if stalled else self.last_error,
                "busy_for_s": round(busy_for_s, 3),
                "inference_timeout_s": self.inference_timeout_s,
                "startup_timeout_s": self.startup_timeout_s,
            }

    def _execute(self, operation: Callable, *, probe=False, startup=False,
                 operation_name="inference", input_sha256=None):
        # The worker owns the lock, including when an HTTP client disconnects.
        # Requests never queue behind model inference and probes report BUSY.
        if not self.lock.acquire(blocking=False):
            if self.snapshot()["status"] == "stalled":
                raise HTTPException(503, "inference stalled; process restart required")
            raise HTTPException(429, "inference busy", headers={"Retry-After": "2"})
        try:
            with self.state_lock:
                self.busy_since = time.monotonic()
            if not probe and not self.ready:
                raise HTTPException(503, "model is not ready", headers={"Retry-After": "10"})
            if probe and self.warming and not startup:
                raise HTTPException(503, "model is loading", headers={"Retry-After": "10"})
            try:
                result = operation()
            except HTTPException:
                raise
            except Exception as exc:
                diagnostic = _safe_failure(exc)
                LOGGER.error(json.dumps({"event": "inference_failure", "boot_id": self.boot_id,
                                         "operation": operation_name, "input_sha256": input_sha256,
                                         **diagnostic}, sort_keys=True))
                if isinstance(exc, InferenceInputError) and not probe:
                    # A rejected input says nothing about model readiness. A probe
                    # rejection, however, means the service self-test failed.
                    raise HTTPException(422, {"code": "InferenceInputError",
                                              "reason": exc.reason_code}) from None
                with self.state_lock:
                    self.ready = False
                    self.warming = False
                    self.last_error = diagnostic["category"]
                if startup:
                    return None
                raise HTTPException(503, "model inference failed; run a deep health probe") from None
            with self.state_lock:
                self.ready = True
                self.warming = False
                self.last_success = datetime.now(timezone.utc).isoformat()
                self.last_error = None
            return result
        finally:
            with self.state_lock:
                self.busy_since = None
            self.lock.release()

    def probe(self, *, startup=False):
        def run():
            vectors = _validated_vectors(self.encoder.query([PROBE_TEXT], bs=1), 1)
            return {"shape": list(vectors.shape), "finite": True,
                    "nonzero": True, "unit_norm": True}

        checks = self._execute(run, probe=True, startup=startup, operation_name="health_probe")
        return {**self.snapshot(), "checks": checks}

    def text_embeddings(self, request: EmbeddingsRequest):
        texts = [request.input] if isinstance(request.input, str) else request.input

        def run():
            if request.input_type == "document":
                titles = request.titles if request.titles is not None else [""] * len(texts)
                raw = self.encoder.docs(titles, texts, bs=4)
            else:
                raw = self.encoder.query(texts, bs=4)
            return _validated_vectors(raw, len(texts), request.dimensions)

        # Bind diagnostics to the original request without storing its contents.
        fingerprint = hashlib.sha256(json.dumps(
            {"type": request.input_type, "texts": texts, "titles": request.titles},
            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        vectors = self._execute(run, operation_name="text_" + request.input_type,
                                input_sha256=fingerprint)
        return self.response(vectors, request.dimensions)

    def image_embeddings(self, content: bytes, dimensions: int):
        # Invalid images are client errors and must not degrade model readiness.
        rgb = _decode_image(content)
        try:
            vectors = self._execute(lambda: _validated_vectors(
                self.encoder.images([rgb], bs=1), 1, dimensions), operation_name="image")
            return self.response(vectors, dimensions)
        finally:
            rgb.close()

    def video_embeddings(self, contents: list[bytes], dimensions: int):
        # Decode before taking the inference lock: bad uploads do not change
        # readiness. One ordered frame tensor represents one native video.
        arrays = []
        expected_size = None
        pixels = 0
        for content in contents:
            image = _decode_image(content, MAX_VIDEO_PIXELS - pixels)
            try:
                if expected_size is not None and image.size != expected_size:
                    raise HTTPException(400, "video frames must have identical dimensions")
                expected_size = image.size
                pixels += image.width * image.height
                arrays.append(np.array(image, dtype=np.uint8))
            finally:
                image.close()
        frames = np.stack(arrays)
        vectors = self._execute(lambda: _validated_vectors(
            self.encoder.videos([frames], bs=1), 1, dimensions), operation_name="video")
        return self.response(vectors, dimensions)

    def response(self, vectors, dimensions):
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": index, "embedding": vector.tolist()}
                     for index, vector in enumerate(vectors)],
            "model": self.encoder.model_id,
            "model_revision": getattr(self.encoder, "revision", None),
            "text_input_policy": getattr(self.encoder, "text_input_policy", None),
            "dimensions": dimensions,
            "boot_id": self.boot_id,
        }


async def _bounded_body(request: Request, maximum: int):
    if request.headers.get("content-length"):
        try:
            content_length = int(request.headers["content-length"])
        except ValueError:
            raise HTTPException(400, "invalid Content-Length") from None
        if content_length < 0:
            raise HTTPException(400, "invalid Content-Length")
        if content_length > maximum:
            raise HTTPException(413, "request body too large")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > maximum:
            raise HTTPException(413, "request body too large")
        content.extend(chunk)
    return bytes(content)


def create_app(encoder=None, api_key: str | None = None):
    """App factory. Inject an encoder in tests; production uses the real model."""
    token = api_key if api_key is not None else os.environ.get("OMNI_INFERENCE_API_KEY", "")
    if not token or not token.strip():
        raise ValueError("OMNI_INFERENCE_API_KEY must be set to a nonempty secret")
    if encoder is None:
        from .encoder import Encoder
        encoder = Encoder(model_id=os.environ.get("OMNI_MODEL"), device=os.environ.get("OMNI_DEVICE"),
                          revision=os.environ.get("OMNI_MODEL_REVISION"))
    state = InferenceState(encoder)

    @asynccontextmanager
    async def lifespan(app):
        async def warmup():
            await run_in_threadpool(state.probe, startup=True)

        task = asyncio.create_task(warmup())
        app.state.warmup_task = task
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="OmniEmbedding inference", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.inference = state

    def authenticate(request: Request):
        authorization = request.headers.get("authorization", "")
        scheme, _, credential = authorization.partition(" ")
        # Compare byte strings so malformed non-ASCII headers cannot raise a 500.
        if scheme.lower() != "bearer" or not hmac.compare_digest(credential.encode(), token.encode()):
            raise HTTPException(401, "valid Bearer token required", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health/live")
    async def live():
        return {"status": "ok", "boot_id": state.boot_id,
                "uptime_s": round(time.monotonic() - state.started, 3)}

    @app.get("/health/ready")
    async def ready(request: Request):
        authenticate(request)
        snapshot = state.snapshot()
        return JSONResponse(snapshot, status_code=200 if snapshot["ready"] else 503)

    @app.post("/health/inference")
    async def inference_health(request: Request):
        authenticate(request)
        return await run_in_threadpool(state.probe)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        authenticate(request)
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise HTTPException(415, "Content-Type must be application/json")
        content = await _bounded_body(request, MAX_JSON_BYTES)
        try:
            payload = EmbeddingsRequest.model_validate_json(content)
        except ValidationError as exc:
            errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                      for error in exc.errors()]
            raise HTTPException(422, errors) from None
        return await run_in_threadpool(state.text_embeddings, payload)

    @app.post("/v1/embeddings/image")
    async def image_embeddings(request: Request, dimensions: int = 768):
        authenticate(request)
        if dimensions not in {128, 256, 512, 768}:
            raise HTTPException(422, "dimensions must be 128, 256, 512 or 768")
        if request.headers.get("content-type", "").split(";")[0].strip() != "multipart/form-data":
            raise HTTPException(415, "Content-Type must be multipart/form-data")
        # Bound actual incoming bytes, including chunked uploads, before a
        # multipart parser can spool unlimited data to the Colab disk.
        content = await _bounded_body(request, MAX_MULTIPART_BYTES)
        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": content, "more_body": False}

        bounded_request = Request(request.scope, receive=receive)
        async with bounded_request.form(max_files=1, max_fields=0, max_part_size=1024) as form:
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read") or len(form) != 1:
                raise HTTPException(422, "one image in multipart field 'file' is required")
            image_content = bytearray()
            while chunk := await upload.read(64 * 1024):
                if len(image_content) + len(chunk) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "image file exceeds 10 MiB")
                image_content.extend(chunk)
        return await run_in_threadpool(state.image_embeddings, bytes(image_content), dimensions)

    @app.post("/v1/embeddings/video")
    async def video_embeddings(request: Request, dimensions: int = 768):
        authenticate(request)
        if dimensions not in {128, 256, 512, 768}:
            raise HTTPException(422, "dimensions must be 128, 256, 512 or 768")
        if request.headers.get("content-type", "").split(";")[0].strip() != "multipart/form-data":
            raise HTTPException(415, "Content-Type must be multipart/form-data")
        content = await _bounded_body(request, MAX_VIDEO_MULTIPART_BYTES)
        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": content, "more_body": False}

        bounded_request = Request(request.scope, receive=receive)
        contents = []
        total_bytes = 0
        async with bounded_request.form(max_files=MAX_VIDEO_FRAMES, max_fields=0,
                                        max_part_size=1024) as form:
            uploads = list(form.multi_items())
            if (not 1 <= len(uploads) <= MAX_VIDEO_FRAMES
                    or any(name != "files" or not hasattr(upload, "read")
                           for name, upload in uploads)):
                raise HTTPException(422, f"1 to {MAX_VIDEO_FRAMES} frames in multipart field 'files' are required")
            for _, upload in uploads:
                frame = bytearray()
                while chunk := await upload.read(64 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_VIDEO_BYTES:
                        raise HTTPException(413, "video frames exceed 32 MiB in total")
                    frame.extend(chunk)
                contents.append(bytes(frame))
        return await run_in_threadpool(state.video_embeddings, contents, dimensions)

    return app
