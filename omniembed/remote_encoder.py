"""Authenticated remote encoder with the same interface as ``Encoder``.

No network calls happen during construction. Each request rereads the endpoint
and private key files so a restarted Colab tunnel does not require a Web restart.
Failures are explicit: this adapter never downloads or falls back to a local
model, follows redirects, or substitutes image averages for video embeddings.
"""
from __future__ import annotations

from datetime import datetime, timezone
import http.client
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np
from PIL import Image

from .config import CONFIG

DIMENSIONS = 768
MAX_TEXT_CHARS = 8192
MAX_TITLE_CHARS = 1024
MAX_TOTAL_TEXT_CHARS = 32768
MAX_TEXT_BATCH = 16
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_VIDEO_BYTES = 32 * 1024 * 1024
MAX_VIDEO_PIXELS = 32_000_000
MAX_VIDEO_FRAMES = 8
MAX_RESPONSE_BYTES = 1024 * 1024


class RemoteEncoderError(RuntimeError):
    """Sanitized error safe for the pipeline's persisted failure message."""

    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _url(value):
    if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        raise RemoteEncoderError("远程推理地址配置无效")
    try:
        parsed = urllib.parse.urlsplit(value)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        raise RemoteEncoderError("远程推理地址配置无效") from None
    if (parsed.scheme not in {"https", "http"} or not host or port == 0
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise RemoteEncoderError("远程推理地址配置无效")
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host.lower() == "localhost"
    if parsed.scheme == "http" and not local:
        raise RemoteEncoderError("远程推理必须使用 HTTPS")
    return value.rstrip("/")


def _key(path):
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                    or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
                raise RemoteEncoderError("推理密钥文件必须属于当前用户且权限为 600")
            raw = handle.read(8193)
        value = raw.decode("ascii").strip()
    except (OSError, UnicodeError, ValueError):
        raise RemoteEncoderError("无法读取私有推理密钥文件") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not value or len(raw) > 8192 or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in value):
        raise RemoteEncoderError("推理密钥文件内容无效")
    return value


def _multipart(files, field):
    boundary = "omniembed-" + uuid.uuid4().hex
    body = bytearray()
    for index, content in enumerate(files):
        body.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
                     f"filename=\"frame-{index:02d}.png\"\r\nContent-Type: image/png\r\n\r\n").encode("ascii"))
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


class RemoteEncoder:
    backend = "remote"

    def __init__(self, model_id=None, device=None, revision=None, dtype=None, *,
                 endpoint_file=None, key_file=None, url=None, timeout=None):
        self.model_id = model_id or CONFIG.model_id
        self.revision = revision or os.environ.get("OMNI_MODEL_REVISION") or None
        self.device = "remote"
        self.dtype = "remote"
        self.load_s = None
        self.boot_id = None
        self._loaded = False
        self.last_success_at = None
        self.last_error = None
        self._url = url or os.environ.get("OMNI_INFERENCE_URL") or None
        self.endpoint_file = Path(endpoint_file or os.environ.get("OMNI_INFERENCE_ENDPOINT_FILE")
                                  or CONFIG.data_dir / "colab/endpoint.json")
        self.key_file = Path(key_file or os.environ.get("OMNI_INFERENCE_KEY_FILE")
                             or CONFIG.data_dir / "colab/inference.key")
        try:
            self.timeout = float(timeout if timeout is not None else os.environ.get("OMNI_INFERENCE_TIMEOUT_S", "120"))
        except (ValueError, TypeError):
            raise ValueError("remote inference timeout must be positive finite seconds") from None
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 600:
            raise ValueError("remote inference timeout must be between 0 and 600 seconds")
        # Disable environment proxies as well as redirects: the Bearer secret
        # must go only to the endpoint selected by the local configuration.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        self._lock = threading.Lock()

    @property
    def loaded(self):
        """Whether this client has ever received a valid embedding. Not health."""
        return self._loaded

    def _endpoint(self):
        if self._url is not None:
            return _url(self._url)
        try:
            with self.endpoint_file.open("rb") as handle:
                raw = handle.read(8193)
            if len(raw) > 8192:
                raise ValueError
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError
        except (OSError, ValueError, UnicodeError, RecursionError):
            raise RemoteEncoderError("推理连接资料缺失或无效；请检查 Colab 服务") from None
        if value.get("status") not in {None, "ready"}:
            raise RemoteEncoderError("Colab 推理服务尚未就绪")
        expires = value.get("expires_at")
        if expires is not None:
            try:
                when = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                if when.tzinfo is None:
                    raise ValueError
            except (ValueError, TypeError, AttributeError):
                raise RemoteEncoderError("推理租约时间配置无效") from None
            if when <= datetime.now(timezone.utc):
                raise RemoteEncoderError("Colab 推理租约已到期；需要用户重新部署")
        return _url(value.get("url"))

    def _request(self, path, body, content_type, rows):
        try:
            result = self._request_locked(path, body, content_type, rows)
        except RemoteEncoderError as exc:
            self.last_error = str(exc)
            raise
        self.last_error = None
        self.last_success_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return result

    def _request_locked(self, path, body, content_type, rows):
        # Serialize requests made by local ingestion/search workers because the
        # remote GPU accepts one inference at a time. Probes can still be BUSY.
        with self._lock:
            deadline = time.monotonic() + self.timeout
            for attempt in range(3):
                url = self._endpoint() + path
                secret = _key(self.key_file)
                request = urllib.request.Request(url, data=body, method="POST", headers={
                    "Authorization": "Bearer " + secret, "Accept": "application/json",
                    "Content-Type": content_type, "User-Agent": "omniembedding-remote/1",
                })
                retry_delay = None
                try:
                    try:
                        response = self._opener.open(request, timeout=max(0.01, deadline - time.monotonic()))
                    except urllib.error.HTTPError as exc:
                        response = exc
                    with response:
                        status = response.code
                        if status == 429:
                            try:
                                retry_delay = min(5.0, max(0.1, float(response.headers.get("Retry-After", "2"))))
                                if not math.isfinite(retry_delay):
                                    retry_delay = 2.0
                            except ValueError:
                                retry_delay = 2.0
                            raw = b""
                        elif status != 200:
                            # Never include a server error body: it can echo
                            # submitted text, credentials or internal URLs.
                            if status in {401, 403}:
                                raise RemoteEncoderError("远程推理鉴权失败；请检查密钥与访问权限")
                            if 300 <= status < 400:
                                raise RemoteEncoderError("远程推理返回重定向，已拒绝转发密钥")
                            raise RemoteEncoderError(f"远程推理请求失败（HTTP {status}）")
                        else:
                            raw = response.read(MAX_RESPONSE_BYTES + 1)
                except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError):
                    raise RemoteEncoderError("无法连接远程推理服务或请求超时；未启用本机模型回退") from None
                if retry_delay is not None:
                    if attempt == 2 or time.monotonic() + retry_delay >= deadline:
                        raise RemoteEncoderError("远程推理繁忙，请稍后重试（HTTP 429）", status_code=429)
                    time.sleep(retry_delay)
                    continue
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise RemoteEncoderError("远程推理响应超过大小限制")
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeError, RecursionError):
                    raise RemoteEncoderError("远程推理返回无效 JSON") from None
                vectors = self._vectors(value, rows)
                self._loaded = True
                return vectors
        raise RemoteEncoderError("远程推理未返回结果")

    def _vectors(self, value, rows):
        try:
            if (not isinstance(value, dict) or value.get("model") != self.model_id
                    or type(value.get("dimensions")) is not int or value["dimensions"] != DIMENSIONS):
                raise ValueError
            revision = value.get("model_revision")
            if not isinstance(revision, str) or not 1 <= len(revision) <= 256:
                raise ValueError
            if self.revision is not None and revision != self.revision:
                raise ValueError
            data = value.get("data")
            if not isinstance(data, list) or len(data) != rows:
                raise ValueError
            ordered = [None] * rows
            for record in data:
                if not isinstance(record, dict):
                    raise ValueError
                index = record.get("index")
                if type(index) is not int or not 0 <= index < rows or ordered[index] is not None:
                    raise ValueError
                embedding = record.get("embedding")
                if (not isinstance(embedding, list) or len(embedding) != DIMENSIONS
                        or any(type(x) not in {int, float} for x in embedding)):
                    raise ValueError
                ordered[index] = embedding
            with np.errstate(over="ignore", invalid="ignore"):
                vectors = np.asarray(ordered, dtype=np.float32)
                norms = np.linalg.norm(vectors, axis=1)
            if (vectors.shape != (rows, DIMENSIONS) or not np.isfinite(vectors).all()
                    or not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=0.05)):
                raise ValueError
            boot = value.get("boot_id")
            if not isinstance(boot, str) or not 1 <= len(boot) <= 128 or any(not (c.isalnum() or c in "_.:-") for c in boot):
                raise ValueError
        except (ValueError, TypeError, KeyError, OverflowError):
            raise RemoteEncoderError("远程向量校验失败（模型、版本、维度或单位向量异常）") from None
        # Pin the first response revision for this client; silently mixing
        # model revisions in one index would make similarity scores unreliable.
        if self.revision is None:
            self.revision = revision
        self.boot_id = boot
        return vectors

    @staticmethod
    def _batch_size(bs):
        if type(bs) is not int or bs <= 0:
            raise ValueError("batch size must be a positive integer")
        return min(bs, MAX_TEXT_BATCH)

    def _text(self, texts, bs, titles=None):
        size = self._batch_size(bs)
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must contain strings")
        if titles is not None:
            titles = list(titles)
            if len(titles) != len(texts) or any(not isinstance(title, str) for title in titles):
                raise ValueError("titles and texts must contain matching strings")
        # Validate every item before issuing any requests, so malformed later
        # items cannot leave a partially completed remote batch.
        costs = []
        for index, text in enumerate(texts):
            title = "" if titles is None else titles[index]
            if not (text.strip() or title.strip()) or len(text) > MAX_TEXT_CHARS or len(title) > MAX_TITLE_CHARS:
                raise ValueError("text/title must contain at most 8192/1024 characters and cannot both be blank")
            costs.append(len(text) + len(title))
        batches, begin, cost = [], 0, 0
        for index, item_cost in enumerate(costs):
            if index > begin and (index - begin >= size or cost + item_cost > MAX_TOTAL_TEXT_CHARS):
                batches.append((begin, index))
                begin, cost = index, 0
            cost += item_cost
        if begin < len(texts):
            batches.append((begin, len(texts)))
        outputs = []
        for begin, end in batches:
            payload = {"input": texts[begin:end], "input_type": "query" if titles is None else "document",
                       "dimensions": DIMENSIONS}
            if titles is not None:
                payload["titles"] = titles[begin:end]
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            outputs.append(self._request("/v1/embeddings", body, "application/json", end - begin))
        return np.concatenate(outputs) if outputs else np.empty((0, DIMENSIONS), dtype=np.float32)

    def query(self, texts, bs=64):
        return self._text(texts, bs)

    def docs(self, titles, texts, bs=64):
        return self._text(texts, bs, titles=titles)

    @staticmethod
    def _png(image, pixel_limit):
        if not isinstance(image, Image.Image):
            raise ValueError("images must contain PIL images")
        if image.width <= 0 or image.height <= 0 or image.width * image.height > pixel_limit:
            raise ValueError("image exceeds remote inference pixel limit")
        if getattr(image, "is_animated", False):
            raise ValueError("animated images are not supported")
        buffer = io.BytesIO()
        with image.convert("RGB") as rgb:
            rgb.save(buffer, format="PNG")
        return buffer.getvalue()

    def images(self, imgs, bs=16):
        self._batch_size(bs)
        outputs = []
        for image in imgs:
            content = self._png(image, MAX_IMAGE_PIXELS)
            if len(content) > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds remote inference 10 MiB limit")
            body, content_type = _multipart([content], "file")
            outputs.append(self._request("/v1/embeddings/image?dimensions=768", body, content_type, 1))
        return np.concatenate(outputs) if outputs else np.empty((0, DIMENSIONS), dtype=np.float32)

    def videos(self, frame_arrays, bs=4):
        """Send each video's ordered RGB frames to the native video encoder."""
        self._batch_size(bs)
        outputs = []
        for frames in frame_arrays:
            if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.ndim != 4
                    or frames.shape[3] != 3 or not 1 <= len(frames) <= MAX_VIDEO_FRAMES
                    or min(frames.shape[1:3]) <= 0):
                raise ValueError("video requires 1 to 8 RGB uint8 frames shaped [T,H,W,3]")
            if int(np.prod(frames.shape[:3])) > MAX_VIDEO_PIXELS:
                raise ValueError("video exceeds remote inference 32 million pixel limit; reduce resolution or frame count")
            files, total = [], 0
            for frame in frames:
                with Image.fromarray(frame) as image:
                    content = self._png(image, MAX_VIDEO_PIXELS)
                total += len(content)
                if total > MAX_VIDEO_BYTES:
                    raise ValueError("video exceeds remote inference 32 MiB limit; reduce resolution or frame count")
                files.append(content)
            body, content_type = _multipart(files, "files")
            outputs.append(self._request("/v1/embeddings/video?dimensions=768", body, content_type, 1))
        return np.concatenate(outputs) if outputs else np.empty((0, DIMENSIONS), dtype=np.float32)
