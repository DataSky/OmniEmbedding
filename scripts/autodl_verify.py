#!/usr/bin/env python3
"""Verify a deployed three-modality service without printing keys or vectors.

Uses only the Python standard library. Pass a key file, never a key argument.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import secrets
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class VerificationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


class Service:
    def __init__(self, base_url, key, timeout):
        parsed = urllib.parse.urlsplit(base_url)
        require(parsed.scheme in {"https", "http"} and parsed.hostname,
                "base URL must be HTTPS or localhost HTTP")
        require(not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
                "base URL must not contain credentials, query parameters or a fragment")
        local = parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
        require(parsed.scheme == "https" or local, "HTTP is allowed only for localhost")
        try:
            parsed.port
        except ValueError:
            raise VerificationError("invalid base URL port") from None
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout
        handlers = [NoRedirect()]
        if local:
            handlers.append(urllib.request.ProxyHandler({}))
        self.opener = urllib.request.build_opener(*handlers)

    def request(self, method, path, data=None, content_type=None, auth=True):
        headers = {"Accept": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + (self.key if auth is True else auth)
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(self.base_url + path, data=data,
                                         headers=headers, method=method)
        try:
            try:
                response = self.opener.open(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                status = response.code
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.URLError) as error:
            # Do not print exception messages, request headers or response bodies.
            raise VerificationError("transport failure: " + type(error).__name__) from None
        require(len(body) <= MAX_RESPONSE_BYTES, "response exceeded size limit")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            payload = None
        return status, payload

    def post_json(self, path, payload):
        return self.request("POST", path, json.dumps(payload, ensure_ascii=False).encode(),
                            "application/json")


def validate_identity(body):
    require(isinstance(body, dict), "expected a JSON object")
    require(body.get("model") == MODEL, "unexpected model")
    require(body.get("model_revision") == REVISION, "unexpected model revision")


def validate_embeddings(status, body, rows, dimensions):
    require(status == 200, "embedding request returned HTTP " + str(status))
    validate_identity(body)
    require(body.get("dimensions") == dimensions, "incorrect response dimensions")
    vectors = body.get("data")
    require(isinstance(vectors, list) and len(vectors) == rows, "incorrect embedding row count")
    for index, item in enumerate(vectors):
        require(isinstance(item, dict) and item.get("index") == index,
                "incorrect embedding row order")
        vector = item.get("embedding")
        require(isinstance(vector, list) and len(vector) == dimensions,
                "incorrect embedding shape")
        require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in vector), "non-finite embedding values")
        norm = math.hypot(*vector)
        require(abs(norm - 1.0) <= 0.001, "embedding is not a unit vector")


def png_frame(phase):
    """Generate two different, valid, equally sized RGB PNG frames."""
    width, height = 96, 64
    pixels = bytearray()
    for y in range(height):
        pixels.append(0)  # PNG scanline filter: None.
        for x in range(width):
            color = (85, 175, 235) if y < 40 else (30, 120, 185)
            if 16 + phase * 16 <= x < 48 + phase * 16 and 35 <= y < 48:
                color = (240, 100, 45)
            pixels.extend(color)

    def chunk(kind, contents):
        return (struct.pack("!I", len(contents)) + kind + contents
                + struct.pack("!I", zlib.crc32(kind + contents) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(pixels))) + chunk(b"IEND", b""))


def multipart(field, frames):
    boundary = "omni-verify-" + secrets.token_hex(16)
    parts = []
    for index, frame in enumerate(frames):
        parts.extend([
            ("--" + boundary + "\r\n").encode(),
            (f'Content-Disposition: form-data; name="{field}"; filename="frame{index}.png"\r\n'
             'Content-Type: image/png\r\n\r\n').encode(),
            frame, b"\r\n",
        ])
    parts.append(("--" + boundary + "--\r\n").encode())
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


def run(args):
    require(math.isfinite(args.ready_timeout) and args.ready_timeout >= 0,
            "ready timeout must be finite and nonnegative")
    require(math.isfinite(args.request_timeout) and args.request_timeout > 0,
            "request timeout must be positive and finite")
    try:
        with args.key_file.open("rb") as source:
            raw_key = source.read(8193)
        require(len(raw_key) <= 8192, "key file is too large")
        key = raw_key.decode("ascii").strip()
    except (OSError, UnicodeError):
        raise VerificationError("key file could not be read as ASCII") from None
    require(key and not any(character.isspace() for character in key), "invalid key file")
    service = Service(args.base_url, key, args.request_timeout)
    checks = []
    started = time.monotonic()
    current = "readiness"

    def record(name, beginning, **details):
        checks.append({"check": name, "ok": True,
                       "elapsed_s": round(time.monotonic() - beginning, 3), **details})

    try:
        beginning = time.monotonic()
        deadline = beginning + args.ready_timeout
        while True:
            status, body = service.request("GET", "/health/ready")
            if status == 200:
                validate_identity(body)
                require(body.get("ready") is True and body.get("status") == "ok",
                        "model is not ready")
                break
            require(status == 503 and time.monotonic() < deadline,
                    "readiness failed with HTTP " + str(status))
            time.sleep(min(5, max(0, deadline - time.monotonic())))
        record(current, beginning)

        current, beginning = "authentication", time.monotonic()
        protected = [("GET", "/health/ready"), ("POST", "/health/inference"),
                     ("POST", "/v1/embeddings"), ("POST", "/v1/embeddings/image"),
                     ("POST", "/v1/embeddings/video")]
        for method, path in protected:
            status, _ = service.request(method, path, auth=False)
            require(status == 401, "missing authentication did not return HTTP 401")
        status, _ = service.request("GET", "/health/ready", auth="invalid-verification-token")
        require(status == 401, "incorrect authentication did not return HTTP 401")
        record(current, beginning, requests=6)

        current, beginning = "deep_probe", time.monotonic()
        status, body = service.request("POST", "/health/inference")
        require(status == 200, "deep probe returned HTTP " + str(status))
        validate_identity(body)
        require(body.get("ready") is True and body.get("status") == "ok", "deep probe is not ready")
        require(body.get("checks") == {"shape": [1, 768], "finite": True,
                                      "nonzero": True, "unit_norm": True}, "deep probe checks failed")
        record(current, beginning, shape=[1, 768])

        texts = ["海边的日落", "厨房里正在做饭"]
        for input_type in ("query", "document"):
            for dimensions in (768, 256):
                current, beginning = f"text_{input_type}_{dimensions}", time.monotonic()
                payload = {"input": texts, "input_type": input_type, "dimensions": dimensions}
                if input_type == "document":
                    payload["titles"] = ["海边", "烹饪"]
                status, body = service.post_json("/v1/embeddings", payload)
                validate_embeddings(status, body, len(texts), dimensions)
                record(current, beginning, shape=[len(texts), dimensions])

        frames = [png_frame(0), png_frame(1)]
        for modality, field, images in (("image", "file", frames[:1]), ("video", "files", frames)):
            current, beginning = modality, time.monotonic()
            data, content_type = multipart(field, images)
            status, body = service.request("POST", f"/v1/embeddings/{modality}?dimensions=768",
                                           data, content_type)
            validate_embeddings(status, body, 1, 768)
            record(current, beginning, shape=[1, 768], frames=len(images))
    except VerificationError as error:
        checks.append({"check": current, "ok": False, "error": str(error)})
        print(json.dumps({"ok": False, "model": MODEL, "revision": REVISION,
                          "elapsed_s": round(time.monotonic() - started, 3), "checks": checks},
                         ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, "model": MODEL, "revision": REVISION,
                      "elapsed_s": round(time.monotonic() - started, 3), "checks": checks},
                     ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="HTTPS endpoint or localhost HTTP base URL")
    parser.add_argument("--key-file", required=True, type=Path, help="file containing the service Bearer key")
    parser.add_argument("--ready-timeout", type=float, default=900, help="seconds to wait for readiness (default: 900)")
    parser.add_argument("--request-timeout", type=float, default=120, help="HTTP request timeout in seconds (default: 120)")
    args = parser.parse_args()
    try:
        return run(args)
    except VerificationError as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    except Exception as error:
        # Avoid tracebacks containing local variables, credentials or vectors.
        print(json.dumps({"ok": False, "error": "verification failed: " + type(error).__name__}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
