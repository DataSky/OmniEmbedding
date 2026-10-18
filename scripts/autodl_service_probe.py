#!/usr/bin/env python3
"""Dependency-free, sequential contract probe for the pinned embedding service.

This is a functional smoke test, not a quality, throughput, or availability test.
No shell is executed when reading the two-variable client .env file. Only HTTP
429 responses are replayed; the explicit wait-ready operation may poll a valid
503 loading/degraded readiness response. Other 5xx and transport errors stop.
"""
from __future__ import annotations

import argparse
import email.utils
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import socket
import ssl
import stat
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
ENV_NAMES = {"OMNI_INFERENCE_URL", "OMNI_INFERENCE_API_KEY"}


class ProbeError(Exception):
    """Only fixed, non-sensitive error codes may leave the probe."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _read_file(path, *, secret=False, maximum=65536):
    """Open regular files without following the final-component symlink."""
    try:
        path = Path(path)
        if path.is_symlink():
            raise ProbeError("configuration_symlink_rejected")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ProbeError("configuration_not_regular_file")
            if secret and (stat.S_IMODE(info.st_mode) != 0o600
                           or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())):
                raise ProbeError("env_requires_owner_and_mode_0600")
            content = handle.read(maximum + 1)
        if len(content) > maximum:
            raise ProbeError("configuration_too_large")
        return content.decode("utf-8")
    except ProbeError:
        raise
    except (OSError, UnicodeError):
        raise ProbeError("configuration_read_failed") from None


def validate_url(value):
    """Require verified HTTPS, or explicit loopback HTTP for local testing."""
    if not isinstance(value, str) or not value or not value.isascii():
        raise ProbeError("endpoint_url_invalid")
    if any(char.isspace() or ord(char) < 32 for char in value) or "\\" in value:
        raise ProbeError("endpoint_url_invalid")
    try:
        parts = urllib.parse.urlsplit(value)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise ProbeError("endpoint_url_invalid") from None
    if (parts.scheme not in {"https", "http"} or not host or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment
            or "?" in value or "#" in value or parts.path not in {"", "/"}
            or (port is not None and not 1 <= port <= 65535)):
        raise ProbeError("endpoint_url_invalid")
    loopback = host.lower() == "localhost"
    try:
        address = ipaddress.ip_address(host)
        loopback = address.is_loopback
        if not address.is_global and not loopback:
            raise ProbeError("endpoint_private_address_rejected")
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", host):
            raise ProbeError("endpoint_hostname_invalid") from None
        if host.endswith(".") or any(not label or len(label) > 63 for label in host.split(".")):
            raise ProbeError("endpoint_hostname_invalid")
    if parts.scheme == "http" and not loopback:
        raise ProbeError("endpoint_requires_https")
    return value.rstrip("/")


def load_config(env_path, endpoint_path=None):
    values = {}
    for line in _read_file(env_path, secret=True).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export[ \t]+)?([A-Z_]+)=(.*)", stripped)
        if not match or match[1] not in ENV_NAMES or match[1] in values:
            raise ProbeError("env_only_two_unique_assignments_allowed")
        value = match[2]
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0] or value[0] in value[1:-1]:
                raise ProbeError("env_literal_value_required")
            value = value[1:-1]
        # Deliberately do not implement shell expansion, escaping, or commands.
        if (not value or len(value) > 4096 or not value.isascii()
                or re.search(r"[\s$`\\;|&<>\"']", value)):
            raise ProbeError("env_literal_value_required")
        values[match[1]] = value
    if values.keys() != ENV_NAMES:
        raise ProbeError("env_both_variables_required")
    url = values["OMNI_INFERENCE_URL"]
    if endpoint_path is not None:
        try:
            endpoint = json.loads(_read_file(endpoint_path))
        except (ValueError, TypeError):
            raise ProbeError("endpoint_file_invalid_json") from None
        if not isinstance(endpoint, dict) or not isinstance(endpoint.get("url"), str):
            raise ProbeError("endpoint_file_requires_url")
        url = endpoint["url"]
    key = values["OMNI_INFERENCE_API_KEY"]
    if not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", key):
        raise ProbeError("env_key_invalid")
    return validate_url(url), key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


class HTTPTransport:
    def __init__(self):
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def __call__(self, request, timeout):
        # urllib's socket timeout alone is not a total request deadline: DNS or
        # a peer slowly sending bytes can otherwise outlive it. A daemon worker
        # bounds how long the caller waits. Timeout is terminal, never replayed;
        # the remote execution outcome remains unknown even after we stop waiting.
        results = queue.Queue(maxsize=1)

        def exchange():
            try:
                result = self._exchange(request, timeout)
            except ProbeError as error:
                result = error
            except (TimeoutError, socket.timeout):
                result = ProbeError("transport_timeout_unknown_execution")
            except urllib.error.URLError as error:
                result = ProbeError("transport_timeout_unknown_execution"
                                    if isinstance(error.reason, (TimeoutError, socket.timeout))
                                    else "transport_failure_unknown_execution")
            except Exception:
                # Do not propagate URLs, response content, or exception messages.
                result = ProbeError("transport_failure_unknown_execution")
            results.put(result)

        threading.Thread(target=exchange, name="embedding-probe-http", daemon=True).start()
        try:
            result = results.get(timeout=timeout)
        except queue.Empty:
            raise ProbeError("transport_timeout_unknown_execution") from None
        if isinstance(result, ProbeError):
            raise result
        return result

    def _exchange(self, request, timeout):
        try:
            response = self.opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        try:
            with response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ProbeError("response_too_large")
                return response.code, response.headers, body
        except (TimeoutError, socket.timeout):
            raise ProbeError("transport_timeout_unknown_execution") from None
        except OSError:
            raise ProbeError("transport_failure_unknown_execution") from None


def synthetic_png(frame_index=0):
    """Generate two small ordered 64x64 RGB frames, without Pillow or downloads."""
    if frame_index not in {0, 1}:
        raise ValueError("frame_index must be 0 or 1")
    pixels = bytearray()
    for y in range(64):
        pixels.append(0)  # PNG filter: None.
        for x in range(64):
            bar = 8 + frame_index * 24 <= x < 24 + frame_index * 24 and 12 <= y < 52
            pixels.extend((230, 180, 30) if bar else (20 + x, 30 + y, 100))

    def chunk(kind, payload):
        return struct.pack("!I", len(payload)) + kind + payload + struct.pack(
            "!I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(pixels))) + chunk(b"IEND", b""))


def multipart(parts):
    boundary = "omni-probe-" + uuid.uuid4().hex
    data = bytearray()
    for field, filename, content in parts:
        data.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
                     f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode("ascii"))
        data.extend(content)
        data.extend(b"\r\n")
    data.extend(f"--{boundary}--\r\n".encode("ascii"))
    return bytes(data), "multipart/form-data; boundary=" + boundary


def retry_delay(value, now_wall):
    """Interpret Retry-After seconds or HTTP-date; never retry before it."""
    if value is None:
        return 2.0
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        result = float(text)
    else:
        try:
            parsed = email.utils.parsedate_to_datetime(text)
            if parsed.tzinfo is None:
                raise ValueError
            result = max(0.0, parsed.timestamp() - now_wall)
        except (ValueError, TypeError, OverflowError):
            return 2.0
    # An unrepresentably large, otherwise numeric delay must stop the request,
    # rather than falling back to a shorter delay and violating Retry-After.
    return result if math.isfinite(result) else float("inf")


def finite_number(number):
    if type(number) not in {int, float}:
        return False
    try:
        return math.isfinite(number)
    except OverflowError:
        return False


class Probe:
    def __init__(self, url, key, *, timeout=300, http_timeout=60, interval=5,
                 retries=2, dimensions=768, transport=None, monotonic=time.monotonic,
                 wall=time.time, sleep=time.sleep, jitter=None):
        self.url, self.key = url, key
        self.timeout, self.http_timeout, self.interval = timeout, http_timeout, interval
        self.retries, self.dimensions = retries, dimensions
        self.transport = transport or HTTPTransport()
        self.clock, self.wall, self.sleep = monotonic, wall, sleep
        self.jitter = jitter or (lambda: random.uniform(0.05, 0.25))
        self.started = self.clock()
        self.deadline = self.started + timeout
        self.boot_id = None
        self.report = {"schema_version": 1, "status": "not_run", "model": MODEL,
                       "model_revision": REVISION, "dimensions": dimensions,
                       "scope": "functional_contract_only_not_quality_or_load_test",
                       "concurrency": 1, "requests": [], "steps": [], "checks": [],
                       "warnings": ["This process cannot coordinate other callers or machines.",
                                    "Synthetic 64x64 PNGs test transport and encoding only.",
                                    "HTTP waiting, polling, and 429 delays share the total deadline.",
                                    "A transport timeout leaves remote execution unknown and is never replayed."]}

    def remaining(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise ProbeError("total_deadline_exceeded")
        return remaining

    def check(self, name, condition):
        self.report["checks"].append({"name": name, "passed": bool(condition)})
        if not condition:
            raise ProbeError("validation_" + name)

    def request(self, name, path, *, body=None, content_type=None):
        for attempt in range(1, self.retries + 2):
            timeout = min(self.http_timeout, self.remaining())
            headers = {"Authorization": "Bearer " + self.key, "Accept": "application/json"}
            if content_type:
                headers["Content-Type"] = content_type
            request = urllib.request.Request(self.url + path, data=body, headers=headers,
                                             method="GET" if body is None else "POST")
            began = self.clock()
            event = {"step": name, "attempt": attempt, "http_status": None}
            self.report["requests"].append(event)
            try:
                status, response_headers, content = self.transport(request, timeout)
            except (TimeoutError, socket.timeout):
                event["outcome"] = "transport_timeout_unknown_execution"
                raise ProbeError(event["outcome"]) from None
            except (urllib.error.URLError, OSError):
                event["outcome"] = "transport_failure_unknown_execution"
                raise ProbeError(event["outcome"]) from None
            except ProbeError as error:
                event["outcome"] = error.code
                raise
            finally:
                event["elapsed_seconds"] = round(self.clock() - began, 6)
            event["http_status"] = status
            self.remaining()
            if status == 429:
                event["outcome"] = "busy"
                if attempt > self.retries:
                    raise ProbeError("http_429_retry_limit")
                retry_after = response_headers.get("Retry-After")
                if retry_after is None:
                    retry_after = response_headers.get("retry-after")
                delay = retry_delay(retry_after, self.wall()) + self.jitter()
                if not math.isfinite(delay) or delay < 0:
                    raise ProbeError("retry_delay_invalid")
                if delay >= self.remaining():
                    raise ProbeError("retry_after_exceeds_deadline")
                event["retry_delay_seconds"] = round(delay, 6)
                self.sleep(delay)
                continue
            if status != 200 and not (name == "ready" and status == 503):
                event["outcome"] = "http_error"
                raise ProbeError("http_" + str(status) if isinstance(status, int) else "http_invalid_status")
            try:
                value = json.loads(content)
            except (ValueError, UnicodeError, TypeError):
                event["outcome"] = "invalid_json"
                raise ProbeError("response_invalid_json") from None
            if not isinstance(value, dict):
                raise ProbeError("response_expected_object")
            event["outcome"] = "received"
            return status, value
        raise ProbeError("request_incomplete")

    def identity(self, value):
        self.check("model_identity", value.get("model") == MODEL)
        self.check("revision_identity", value.get("model_revision") == REVISION)
        boot = value.get("boot_id")
        try:
            valid_boot = isinstance(boot, str) and str(uuid.UUID(boot)) == boot
        except (ValueError, TypeError, AttributeError):
            valid_boot = False
        self.check("boot_id_valid", valid_boot)
        self.check("boot_id_stable", self.boot_id is None or boot == self.boot_id)
        self.boot_id = boot

    def readiness(self, *, wait=False):
        began = self.clock()
        while True:
            status, value = self.request("ready", "/health/ready")
            self.identity(value)
            ready = status == 200 and value.get("ready") is True and value.get("status") == "ok"
            if ready:
                self.check("ready_true", True)
                self.report["steps"].append({"name": "ready", "status": "pass",
                                              "elapsed_seconds": round(self.clock() - began, 6)})
                return
            if not wait:
                self.check("ready_true", False)
            # The explicit readiness poll is not a generic 5xx inference retry.
            self.check("loading_response", value.get("ready") is False
                       and value.get("status") in {"loading", "degraded", "stalled"})
            if self.interval >= self.remaining():
                raise ProbeError("readiness_deadline_exceeded")
            self.sleep(self.interval)

    def vectors(self, value, rows):
        self.identity(value)
        self.check("response_object", value.get("object") == "list")
        self.check("dimensions", type(value.get("dimensions")) is int
                   and value["dimensions"] == self.dimensions)
        data = value.get("data")
        self.check("row_count", isinstance(data, list) and len(data) == rows)
        for index, row in enumerate(data):
            self.check("row_object", isinstance(row, dict) and row.get("object") == "embedding")
            self.check("row_index", type(row.get("index")) is int and row["index"] == index)
            vector = row.get("embedding")
            self.check("vector_length", isinstance(vector, list) and len(vector) == self.dimensions)
            self.check("vector_finite", all(finite_number(number) for number in vector))
            try:
                norm = math.sqrt(math.fsum(number * number for number in vector))
            except (OverflowError, ValueError):
                norm = float("inf")
            self.check("vector_unit_norm", math.isfinite(norm) and abs(norm - 1) <= 1e-3)

    def embedding_step(self, name, path, body, content_type, rows=1):
        began = self.clock()
        _, value = self.request(name, path, body=body, content_type=content_type)
        self.vectors(value, rows)
        self.report["steps"].append({"name": name, "status": "pass", "rows": rows,
                                     "elapsed_seconds": round(self.clock() - began, 6)})

    def run(self, command):
        self.report["command"] = command
        try:
            self.readiness(wait=command == "wait-ready")
            if command == "test":
                for role in ("query", "document"):
                    payload = {"input": ["聚光灯下的舞台表演"], "input_type": role,
                               "dimensions": self.dimensions}
                    if role == "document":
                        payload["input"] = ["舞台上有聚光灯和表演者", "厨房内正在烹饪食物"]
                        payload["titles"] = ["舞台素材", "烹饪素材"]
                    self.embedding_step(role, "/v1/embeddings",
                                        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                        "application/json", len(payload["input"]))
                frames = [synthetic_png(0), synthetic_png(1)]
                body, kind = multipart([("file", "image.png", frames[0])])
                self.embedding_step("image", f"/v1/embeddings/image?dimensions={self.dimensions}", body, kind)
                body, kind = multipart([("files", f"frame{index:02d}.png", frame)
                                        for index, frame in enumerate(frames)])
                self.embedding_step("video", f"/v1/embeddings/video?dimensions={self.dimensions}", body, kind)
            self.remaining()
            self.report["status"] = "pass"
        except ProbeError as error:
            self.report["status"] = "fail"
            self.report["error"] = error.code
        self.report["boot_id"] = self.boot_id
        self.report["elapsed_seconds"] = round(self.clock() - self.started, 6)
        self.report["check_count"] = len(self.report["checks"])
        self.report["passed_check_count"] = sum(item["passed"] for item in self.report["checks"])
        return self.report


def write_report(path, report):
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ProbeError("output_requires_regular_file")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".probe-", delete=False) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    except (OSError, ValueError):
        raise ProbeError("output_write_failed") from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def positive_seconds(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("seconds must be a positive finite number") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("seconds must be a positive finite number")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ready", "wait-ready", "test"))
    parser.add_argument("--env", required=True, type=Path, help="owner-only mode 0600 client .env")
    parser.add_argument("--endpoint-file", type=Path, help="JSON url overrides .env URL after migration")
    parser.add_argument("--output", type=Path, help="write sanitized JSON atomically with mode 0600")
    parser.add_argument("--timeout-seconds", type=positive_seconds, default=300,
                        help="overall time budget including polling and 429 delays (default: 300)")
    parser.add_argument("--http-timeout-seconds", type=positive_seconds, default=60,
                        help="per-operation socket timeout, capped by remaining budget (default: 60)")
    parser.add_argument("--interval-seconds", type=positive_seconds, default=5)
    parser.add_argument("--429-retries", dest="retries", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--dimensions", type=int, choices=(128, 256, 512, 768), default=768)
    args = parser.parse_args(argv)
    try:
        if args.output and (args.output.resolve() == args.env.resolve()
                            or (args.endpoint_file and args.output.resolve() == args.endpoint_file.resolve())):
            raise ProbeError("output_must_not_overwrite_configuration")
        url, key = load_config(args.env, args.endpoint_file)
        report = Probe(url, key, timeout=args.timeout_seconds,
                       http_timeout=args.http_timeout_seconds, interval=args.interval_seconds,
                       retries=args.retries, dimensions=args.dimensions).run(args.command)
        if args.output:
            write_report(args.output, report)
    except ProbeError as error:
        report = {"schema_version": 1, "command": args.command, "status": "fail",
                  "error": error.code, "check_count": 0}
    # A configuration error contains no response, header, exception text, or Key.
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
