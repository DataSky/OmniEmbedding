#!/usr/bin/env python3
"""Measure the public inference API directly; never use RemoteEncoder retries.

Examples (the key itself is never an argument or an output):
  python scripts/autodl_benchmark_service.py latency --key-file PRIVATE_FILE --output-dir RESULTS
  python scripts/autodl_benchmark_service.py load --key-file PRIVATE_FILE --output-dir RESULTS --case-seconds 10
  python scripts/autodl_benchmark_service.py load --key-file PRIVATE_FILE --target-rps 5 --concurrencies 4
  python scripts/autodl_benchmark_service.py load --key-file PRIVATE_FILE --input-type document --include-media
  python scripts/autodl_benchmark_service.py soak --key-file PRIVATE_FILE --soak-seconds 600
  python scripts/autodl_benchmark_service.py sanity --key-file PRIVATE_FILE
  python scripts/autodl_benchmark_service.py summary --input RESULTS

Every case runs for its full configured arrival window. min-success is a sample
validity flag, never a reason to stop early or run forever. Successful throughput
uses elapsed wall time including draining in-flight requests. HTTP latency ends
when the complete response body arrives; scheduled load additionally records
client queue delay and planned-arrival-to-completion latency. No HTTP retries,
environment proxies, redirects, local encoder locks, or Retry-After sleeps apply.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import time
from urllib.parse import urlsplit
import uuid

import httpx

ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = 768
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
TEXT_SEED = "人工智能向量检索系统理解中文文档图像视频内容帮助研究人员比较模型性能稳定性以及实际服务响应速度"
ANCHOR_TEXT = "固定健康锚点用于验证同一模型重复推理的向量一致性"
DOCUMENT_TITLE = "性能测试文档"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)


def distribution(values):
    return {"count": len(values), "p50": percentile(values, .50),
            "p95": percentile(values, .95), "p99": percentile(values, .99),
            "max": round(max(values), 3) if values else None}


def text_of_length(length, offset=0):
    repeated = TEXT_SEED * ((length + offset) // len(TEXT_SEED) + 2)
    return repeated[offset:offset + length]


def json_body(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def read_connection(endpoint_file, key_file):
    """Read only the explicitly selected endpoint and a private regular key file."""
    endpoint = json.loads(endpoint_file.read_text(encoding="utf-8"))
    url = endpoint.get("url", "")
    parsed = urlsplit(url)
    try:
        local = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        local = parsed.hostname == "localhost"
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or any(character.isspace() for character in url)
            or (parsed.scheme == "http" and not local)):
        raise ValueError("endpoint must be HTTPS, without credentials, query or fragment")
    descriptor = os.open(key_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
            raise ValueError("key file must belong to the current user and have mode 600")
        raw = source.read(8193)
    secret = raw.decode("ascii").strip()
    if not secret or len(raw) > 8192 or any(not 33 <= ord(character) <= 126 for character in secret):
        raise ValueError("invalid private key file")
    return url.rstrip("/"), secret, endpoint


@dataclass(frozen=True)
class Case:
    name: str
    path: str
    payload: bytes = b""
    content_type: str = "application/json"
    items: int = 1
    length: int | None = None
    batch: int = 1
    modality: str = "text"
    frames: int = 0
    method: str = "POST"


def text_payload(texts, input_type):
    value = {"input": texts, "input_type": input_type, "dimensions": DIMENSIONS}
    if input_type == "document":
        value["titles"] = [DOCUMENT_TITLE] * len(texts)
    return json_body(value)


def text_case(length, batch, input_type="query"):
    prefix = "document" if input_type == "document" else "text"
    return Case(f"{prefix}_{length}chars_bs{batch}", "/v1/embeddings",
                text_payload([text_of_length(length, item) for item in range(batch)], input_type),
                items=batch, length=length, batch=batch)


def health_case(name):
    return Case("health_" + name, "/health/" + name, items=0, batch=0,
                modality="health", method="GET")


def anchor_case():
    return Case("anchor", "/v1/embeddings",
                json_body({"input": [ANCHOR_TEXT], "input_type": "query", "dimensions": DIMENSIONS}))


def media_cases(dataset=None):
    # Synthetic inputs characterize cost and stability, not semantic accuracy.
    from PIL import Image, ImageDraw

    frames = []
    if dataset is not None:
        first = json.loads((dataset / "corpus.jsonl").read_text().splitlines()[0])
        paths = [Path(value) if Path(value).is_absolute() else dataset / value
                 for value in first["frame_paths"]]
        if len(paths) != 8:
            raise ValueError("real media benchmark requires exactly eight ordered frames")
        frames = [path.read_bytes() for path in paths]
        with Image.open(io.BytesIO(frames[0])) as image:
            width, height = image.size
        label = f"{width}x{height}_real_jpeg"
        extension, media_type = "jpg", "image/jpeg"
    else:
        for index in range(8):
            image = Image.new("RGB", (256, 256), (20 + index * 20, 90, 160))
            draw = ImageDraw.Draw(image)
            draw.rectangle((16 + index * 10, 30, 120 + index * 10, 180), fill=(220, 180, 40))
            output = io.BytesIO()
            image.save(output, format="PNG")
            frames.append(output.getvalue())
            image.close()
        label, extension, media_type = "256px", "png", "image/png"

    def multipart(contents, field):
        boundary = "omni-benchmark-" + uuid.uuid4().hex
        body = bytearray()
        for index, content in enumerate(contents):
            body.extend((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
                         f"filename=\"frame-{index}.{extension}\"\r\nContent-Type: {media_type}\r\n\r\n").encode())
            body.extend(content)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())
        return bytes(body), "multipart/form-data; boundary=" + boundary

    image, image_type = multipart(frames[:1], "file")
    result = [Case("image_" + label, "/v1/embeddings/image", image, image_type, modality="image")]
    for count in (2, 8):
        body, content_type = multipart(frames[:count], "files")
        result.append(Case(f"video_{count}frames_" + label, "/v1/embeddings/video", body,
                           content_type, modality="video", frames=count))
    return result


class ResponseValidationError(ValueError):
    pass


def validated_vectors(value, rows, expected_model=None, expected_revision=None):
    if (not isinstance(value, dict) or type(value.get("dimensions")) is not int
            or value["dimensions"] != DIMENSIONS):
        raise ResponseValidationError("dimensions")
    if expected_model and value.get("model") != expected_model:
        raise ResponseValidationError("model")
    if expected_revision and value.get("model_revision") != expected_revision:
        raise ResponseValidationError("revision")
    records = value.get("data")
    if not isinstance(records, list) or len(records) != rows:
        raise ResponseValidationError("rows")
    vectors = [None] * rows
    norms = []
    for record in records:
        if not isinstance(record, dict):
            raise ResponseValidationError("record")
        index, vector = record.get("index"), record.get("embedding")
        if (type(index) is not int or not 0 <= index < rows or vectors[index] is not None
                or not isinstance(vector, list) or len(vector) != DIMENSIONS):
            raise ResponseValidationError("shape")
        if any(type(number) not in (int, float) or not math.isfinite(number) for number in vector):
            raise ResponseValidationError("finite")
        norm = math.sqrt(math.fsum(number * number for number in vector))
        if not math.isclose(norm, 1.0, abs_tol=.02):
            raise ResponseValidationError("unit_norm")
        vectors[index] = vector
        norms.append(norm)
    if not isinstance(value.get("boot_id"), str) or not value["boot_id"]:
        raise ResponseValidationError("boot_id")
    return vectors, {"shape": [rows, DIMENSIONS], "finite": True, "unit_norm": True,
                     "norm_min": round(min(norms), 8), "norm_max": round(max(norms), 8)}


class Recorder:
    def __init__(self, path, run_id):
        self.path, self.run_id = path, run_id
        self.handle = path.open("x", encoding="utf-8")
        path.chmod(0o600)
        self.boot_ids = set()
        self.last_boot_id = None

    def emit(self, record):
        value = {"run_id": self.run_id, "at": utc_now(), **record}
        self.handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()

    def request(self, record):
        boot = record.get("boot_id")
        if boot and boot not in self.boot_ids:
            self.emit({"record_type": "boot_observed", "boot_id": boot,
                       "previous_boot_id": self.last_boot_id, "changed": bool(self.boot_ids)})
            self.boot_ids.add(boot)
        if boot:
            self.last_boot_id = boot
        self.emit({"record_type": "request", **record})

    def close(self):
        self.handle.close()


class Benchmark:
    def __init__(self, client, url, secret, endpoint, recorder, args):
        self.client, self.url, self.recorder, self.args = client, url, recorder, args
        self.headers = {"Authorization": "Bearer " + secret, "Accept": "application/json",
                        "User-Agent": "omniembedding-benchmark/1"}
        self.expected_model = endpoint.get("model", "google/embeddinggemma-2")
        self.expected_revision = endpoint.get("model_revision")
        self.origin = time.perf_counter()
        self.stopping = asyncio.Event()

    async def request(self, case, phase, case_run_id, concurrency=1, scheduled_at=None, reference=None):
        started = time.perf_counter()
        planned = started if scheduled_at is None else scheduled_at
        record = {"phase": phase, "case": case.name, "case_run_id": case_run_id,
                  "request_id": str(uuid.uuid4()), "status": None, "latency_ms": None,
                  "items": case.items, "payload_bytes": len(case.payload), "response_bytes": 0,
                  "boot_id": None, "errorclass": None, "success": False,
                  "concurrency": concurrency, "text_chars": case.length, "batch": case.batch,
                  "modality": case.modality, "frames": case.frames,
                  "started_offset_s": round(started - self.origin, 6),
                  "scheduled_offset_s": round(planned - self.origin, 6),
                  "client_queue_ms": round(max(0.0, started - planned) * 1000, 3)}
        raw = bytearray()
        vectors = None
        finished = None
        try:
            headers = {**self.headers, "Content-Type": case.content_type,
                       "X-Request-ID": record["request_id"]}
            async with self.client.stream(case.method, self.url + case.path,
                                          content=case.payload, headers=headers) as response:
                record["status"] = response.status_code
                async for part in response.aiter_bytes():
                    raw.extend(part)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise ResponseValidationError("response_size")
            finished = time.perf_counter()
            value = None
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError):
                if response.status_code == 200:
                    raise
            if isinstance(value, dict):
                boot = value.get("boot_id")
                record["boot_id"] = boot if isinstance(boot, str) and len(boot) <= 256 else None
                if case.modality == "health":
                    record["health_status"] = value.get("status")
            if response.status_code != 200:
                record["errorclass"] = "HTTP" + str(response.status_code)
            else:
                if case.modality == "health":
                    if (not isinstance(value, dict) or not record["boot_id"]
                            or (case.name == "health_ready" and value.get("ready") is not True)):
                        raise ResponseValidationError("health")
                    record["health_status"] = value.get("status")
                else:
                    vectors, checks = validated_vectors(value, case.items, self.expected_model,
                                                        self.expected_revision)
                    record["checks"] = checks
                    if reference is not None:
                        vector = vectors[0]
                        cosine = math.fsum(a * b for a, b in zip(reference, vector)) / (
                            math.sqrt(math.fsum(a * a for a in reference))
                            * math.sqrt(math.fsum(a * a for a in vector)))
                        record["anchor_cosine"] = max(-1.0, min(1.0, cosine))
                        if cosine < self.args.anchor_cosine:
                            raise ResponseValidationError("anchor_drift")
                record["success"] = True
        except asyncio.CancelledError:
            record["errorclass"] = "ClientCancelled"
            raise
        except ResponseValidationError as error:
            record["errorclass"] = "ResponseValidationError:" + str(error)
        except Exception as error:
            # Exception messages, response bodies and headers may contain secrets.
            record["errorclass"] = type(error).__name__
        finally:
            finished = finished or time.perf_counter()
            record["latency_ms"] = round((finished - started) * 1000, 3)
            record["end_to_end_ms"] = round((finished - planned) * 1000, 3)
            record["finished_offset_s"] = round(finished - self.origin, 6)
            record["response_bytes"] = len(raw)
            self.recorder.request(record)
        return record, vectors

    def dropped(self, case, phase, case_id, concurrency, planned, errorclass):
        now = time.perf_counter()
        self.recorder.request({"phase": phase, "case": case.name, "case_run_id": case_id,
            "request_id": str(uuid.uuid4()), "status": None, "latency_ms": None,
            "items": case.items, "payload_bytes": len(case.payload), "response_bytes": 0,
            "boot_id": None, "errorclass": errorclass, "success": False,
            "concurrency": concurrency, "text_chars": case.length, "batch": case.batch,
            "modality": case.modality, "frames": case.frames,
            "scheduled_offset_s": round(planned - self.origin, 6),
            "started_offset_s": None, "finished_offset_s": round(now - self.origin, 6),
            "client_queue_ms": round((now - planned) * 1000, 3),
            "end_to_end_ms": round((now - planned) * 1000, 3)})

    async def paced_request(self, case, phase, case_id, concurrency, planned=None):
        started = time.perf_counter()
        result = await self.request(case, phase, case_id, concurrency, planned)
        await asyncio.sleep(max(0.0, self.args.request_gap - (time.perf_counter() - started)))
        return result

    async def run_case(self, case, phase, concurrency, target_rps=None):
        case_id = str(uuid.uuid4())
        started = time.perf_counter()
        deadline = started + self.args.case_seconds
        event = {"case_run_id": case_id, "phase": phase, "case": case.name,
                 "concurrency": concurrency, "target_rps": target_rps,
                 "arrival_window_s": self.args.case_seconds, "min_success": self.args.min_success,
                 "started_offset_s": started - self.origin}
        self.recorder.emit({"record_type": "case_start", **event})
        workers = []
        queue = asyncio.Queue(maxsize=self.args.max_pending)
        drain_timeout = False
        interrupted = False

        async def closed_worker():
            while not self.stopping.is_set() and time.perf_counter() < deadline:
                await self.paced_request(case, phase, case_id, concurrency)

        async def scheduled_worker():
            while True:
                planned = await queue.get()
                try:
                    await self.paced_request(case, phase, case_id, concurrency, planned)
                finally:
                    queue.task_done()

        try:
            if target_rps is None:
                workers = [asyncio.create_task(closed_worker()) for _ in range(concurrency)]
                await asyncio.sleep(self.args.case_seconds)
            else:
                workers = [asyncio.create_task(scheduled_worker()) for _ in range(concurrency)]
                index = 0
                while not self.stopping.is_set():
                    planned = started + index / target_rps
                    if planned >= deadline:
                        break
                    await asyncio.sleep(max(0.0, planned - time.perf_counter()))
                    if self.stopping.is_set():
                        break
                    try:
                        queue.put_nowait(planned)
                    except asyncio.QueueFull:
                        self.dropped(case, phase, case_id, concurrency, planned, "ClientQueueFull")
                    index += 1
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            if target_rps is None:
                await asyncio.wait_for(asyncio.gather(*workers), timeout=self.args.drain_seconds)
            else:
                await asyncio.wait_for(queue.join(), timeout=self.args.drain_seconds)
        except asyncio.TimeoutError:
            drain_timeout = True
        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            while not queue.empty():
                planned = queue.get_nowait()
                self.dropped(case, phase, case_id, concurrency, planned,
                             "ClientInterrupted" if interrupted else "ClientDrainTimeout")
                queue.task_done()
            elapsed = time.perf_counter() - started
            self.recorder.emit({"record_type": "case_end", **event, "elapsed_s": elapsed,
                                "drain_timeout": drain_timeout, "interrupted": interrupted})
        print(f"完成 {phase} {case.name} C={concurrency}，原始记录已保存", flush=True)
        if drain_timeout:
            # A cancelled HTTP client does not stop the server's GPU worker.
            # Continuing would contaminate later cases with unfinished work.
            raise RuntimeError("case drain timed out; later cases were not started")

    async def baseline(self):
        for case in (health_case("live"), health_case("ready")):
            record, _ = await self.request(case, "baseline", "baseline")
            if not record["success"]:
                raise RuntimeError("baseline health check failed; inspect sanitized raw records")
        for _ in range(self.args.warmup):
            record, _ = await self.request(text_case(16, 1), "baseline", "baseline")
            if not record["success"]:
                raise RuntimeError("baseline embedding check failed; inspect sanitized raw records")

    async def latency(self):
        for length in self.args.lengths:
            for batch in self.args.batches:
                if self.stopping.is_set():
                    return
                await self.run_case(text_case(length, batch, self.args.input_type), "latency", 1)
        if self.args.include_media:
            for case in media_cases(self.args.media_dataset):
                await self.run_case(case, "latency", 1)

    async def load(self):
        for length in self.args.load_lengths:
            for batch in self.args.load_batches:
                # Payload is reused byte-for-byte at every concurrency level.
                case = text_case(length, batch, self.args.input_type)
                for concurrency in self.args.concurrencies:
                    if self.stopping.is_set():
                        return
                    await self.run_case(case, "load", concurrency, self.args.target_rps)
        if self.args.include_media:
            for case in media_cases(self.args.media_dataset):
                for concurrency in self.args.concurrencies:
                    if self.stopping.is_set():
                        return
                    await self.run_case(case, "load", concurrency, self.args.target_rps)

    async def soak(self):
        cases = [text_case(64, 1, self.args.input_type),
                 text_case(256, 4, self.args.input_type), *media_cases(self.args.media_dataset)]
        started = time.perf_counter()
        deadline = started + self.args.soak_seconds
        case_id = str(uuid.uuid4())
        self.recorder.emit({"record_type": "case_start", "phase": "soak", "case": "mixed",
            "case_run_id": case_id, "arrival_window_s": self.args.soak_seconds,
            "min_success": self.args.min_success, "concurrency": 1,
            "started_offset_s": started - self.origin})
        next_probe = next_anchor = started
        reference = None
        index = 0
        interrupted = False
        try:
            while not self.stopping.is_set() and time.perf_counter() < deadline:
                now = time.perf_counter()
                if now >= next_probe:
                    for case in (health_case("live"), health_case("ready")):
                        await self.request(case, "soak_probe", case_id)
                    next_probe = now + self.args.probe_interval
                if now >= next_anchor:
                    record, vectors = await self.request(anchor_case(), "soak_anchor", case_id,
                                                         reference=reference)
                    if reference is None and record["success"]:
                        reference = vectors[0]
                        self.recorder.emit({"record_type": "anchor_reference", "case_run_id": case_id,
                                           "boot_id": record["boot_id"]})
                    next_anchor = now + self.args.anchor_interval
                await self.paced_request(cases[index % len(cases)], "soak", case_id, 1)
                index += 1
        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            self.recorder.emit({"record_type": "case_end", "phase": "soak", "case": "mixed",
                "case_run_id": case_id, "arrival_window_s": self.args.soak_seconds,
                "min_success": self.args.min_success, "concurrency": 1,
                "elapsed_s": time.perf_counter() - started, "interrupted": interrupted})

    async def sanity(self):
        """Small bounded negative requests and real single/batch comparisons."""
        case_id = str(uuid.uuid4())
        negatives = [
            ("empty_batch", {"input": []}, 422),
            ("empty_text", {"input": ""}, 422),
            ("too_many_items", {"input": ["测试"] * 17}, 422),
            ("too_long_text", {"input": "测" * 8193}, 422),
            ("total_text_limit", {"input": ["测" * 2049] * 16}, 422),
            ("unsupported_dimension", {"input": "测试", "dimensions": 123}, 422),
            ("unsupported_input_type", {"input": "测试", "input_type": "unsupported"}, 422),
            ("query_with_titles", {"input": "测试", "titles": [DOCUMENT_TITLE]}, 422),
            ("title_count_mismatch", {"input": ["测试", "文档"], "input_type": "document",
                                      "titles": [DOCUMENT_TITLE]}, 422),
        ]
        for name, payload, expected in negatives:
            case = Case(name, "/v1/embeddings", json_body(payload), items=0, batch=0)
            record, _ = await self.request(case, "sanity_negative", case_id)
            self.recorder.emit({"record_type": "sanity_check", "case": name,
                               "passed": record["status"] == expected,
                               "expected_status": expected, "actual_status": record["status"]})
        for name, body, content_type, expected in [
                ("malformed_json", b"{", "application/json", 422),
                ("unsupported_content_type", b"{}", "text/plain", 415)]:
            record, _ = await self.request(Case(name, "/v1/embeddings", body, content_type,
                                                items=0, batch=0), "sanity_negative", case_id)
            self.recorder.emit({"record_type": "sanity_check", "case": name,
                               "passed": record["status"] == expected,
                               "expected_status": expected, "actual_status": record["status"]})
        texts = [text_of_length(length) for length in (16, 64, 256)]
        singles = []
        for index, text in enumerate(texts):
            case = Case(f"single_{index}", "/v1/embeddings", text_payload([text], self.args.input_type))
            record, vectors = await self.request(case, "sanity_consistency", case_id)
            singles.append(vectors[0] if record["success"] else None)
        for name, order in [("single_vs_batch", [0, 1, 2]), ("batch_order_invariance", [2, 0, 1])]:
            case = Case(name, "/v1/embeddings", text_payload([texts[index] for index in order],
                         self.args.input_type), items=3, batch=3)
            record, vectors = await self.request(case, "sanity_consistency", case_id)
            cosines = []
            if record["success"] and all(vector is not None for vector in singles):
                for reference_index, vector in zip(order, vectors):
                    reference = singles[reference_index]
                    cosine = math.fsum(a * b for a, b in zip(reference, vector)) / (
                        math.sqrt(math.fsum(a * a for a in reference))
                        * math.sqrt(math.fsum(a * a for a in vector)))
                    cosines.append(max(-1.0, min(1.0, cosine)))
            self.recorder.emit({"record_type": "sanity_check", "case": name,
                               "passed": len(cosines) == 3 and min(cosines) >= self.args.anchor_cosine,
                               "min_cosine": min(cosines, default=None),
                               "required_cosine": self.args.anchor_cosine})


def summarize(paths, output_dir):
    groups = defaultdict(list)
    events, boot_ids = {}, set()
    invalid_lines = 0
    sanity_checks = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError):
                    invalid_lines += 1
                    continue
                if not isinstance(record, dict):
                    invalid_lines += 1
                    continue
                key = (record.get("run_id"), record.get("case_run_id"))
                if record.get("record_type") == "request":
                    groups[(key, record.get("phase"), record.get("case"))].append(record)
                    if record.get("boot_id"):
                        boot_ids.add(record["boot_id"])
                elif record.get("record_type") in {"case_start", "case_end"}:
                    events[key] = {**events.get(key, {}), **record}
                elif record.get("record_type") == "sanity_check":
                    sanity_checks.append(record)
    cases = []
    for (key, phase, name), records in groups.items():
        event = events.get(key, {})
        successful = [record for record in records if record.get("success")]
        http = [record for record in records if record.get("started_offset_s") is not None]
        busy = [record for record in records if record.get("status") == 429]
        start = min(record.get("scheduled_offset_s", 0.0) for record in records)
        finish = max(record.get("finished_offset_s", start) for record in records)
        elapsed = event.get("elapsed_s", max(0.0, finish - start))
        denominator = elapsed if elapsed > 0 else 1.0
        minimum = event.get("min_success", 0)
        cases.append({"run_id": key[0], "case_run_id": key[1], "phase": phase, "case": name,
            "concurrency": records[0].get("concurrency"), "text_chars": records[0].get("text_chars"),
            "batch": records[0].get("batch"), "target_rps": event.get("target_rps"),
            "arrival_window_s": event.get("arrival_window_s"), "elapsed_s": round(elapsed, 6),
            "offered_requests": len(records), "http_attempts": len(http),
            "success_count": len(successful), "success_items": sum(record["items"] for record in successful),
            "success_rps": round(len(successful) / denominator, 6),
            "success_items_per_second": round(sum(record["items"] for record in successful) / denominator, 6),
            "http_429_count": len(busy), "http_429_rate": round(len(busy) / len(http), 6) if http else None,
            "http_status_counts": dict(Counter(str(record.get("status")) for record in records)),
            "error_counts": dict(Counter(record["errorclass"] for record in records if record.get("errorclass"))),
            "success_http_latency_ms": distribution([record["latency_ms"] for record in successful]),
            "success_end_to_end_ms": distribution([record["end_to_end_ms"] for record in successful]),
            "success_client_queue_ms": distribution([record["client_queue_ms"] for record in successful]),
            "http_429_latency_ms": distribution([record["latency_ms"] for record in busy]),
            "min_success": minimum, "sample_valid": bool(successful) and len(successful) >= minimum,
            "interrupted": event.get("interrupted", event.get("record_type") == "case_start"),
            "drain_timeout": event.get("drain_timeout", False),
            "boot_ids": sorted({record["boot_id"] for record in records if record.get("boot_id")}),
            "anchor_min_cosine": min((record["anchor_cosine"] for record in records
                                       if "anchor_cosine" in record), default=None)})
    value = {"generated_at": utc_now(), "raw_files": [str(path.resolve()) for path in paths],
             "invalid_jsonl_lines": invalid_lines, "unique_boot_ids": sorted(boot_ids),
             "boot_id_changed": len(boot_ids) > 1, "cases": cases,
             "sanity_checks": sanity_checks,
             "sanity_passed": all(check["passed"] for check in sanity_checks) if sanity_checks else None,
             "notes": ["No HTTP retries, redirects or environment proxies.",
                       "All measured cases use the full arrival window; min_success only marks sample validity.",
                       "Throughput includes drain time; HTTP 429 is counted separately from validated successes.",
                       "Synthetic images/video measure performance and stability, not semantic quality."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "summary.json"
    with destination.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
    destination.chmod(0o600)
    lines = ["# 推理服务性能测评", "", "吞吐按完整测量窗口及收尾时间计算；仅统计通过向量校验的成功响应。", "",
             "| 阶段 | 用例 | 并发 | 成功数 | 429 | 成功请求/秒 | 成功条目/秒 | HTTP p50/p95/p99 ms | 样本有效 |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for case in cases:
        latency = case["success_http_latency_ms"]
        lines.append(f"| {case['phase']} | {case['case']} | {case['concurrency']} | {case['success_count']} | "
                     f"{case['http_429_count']} | {case['success_rps']:.3f} | {case['success_items_per_second']:.3f} | "
                     f"{latency['p50']}/{latency['p95']}/{latency['p99']} | {'是' if case['sample_valid'] else '否'} |")
    lines.extend(["", "同一目录可能含多个运行批次；请结合 run_id、case_run_id 读取详细 JSON。",
                  "客户端排队与计划到达至完成延迟位于 summary.json；原始 JSONL 不含密钥、输入正文或完整向量。", ""])
    if sanity_checks:
        lines.extend(["边界与批量一致性检查：" + ("全部通过" if value["sanity_passed"] else "存在失败"), ""])
    markdown = output_dir / "summary.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    markdown.chmod(0o600)
    return value


def csv_ints(value):
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("use comma-separated integers") from None
    if not values or any(number <= 0 for number in values):
        raise argparse.ArgumentTypeError("integers must be positive")
    return list(dict.fromkeys(values))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("latency", "load", "soak", "sanity", "summary"))
    parser.add_argument("--endpoint-file", type=Path, default=ROOT / "data/autodl/endpoint.json")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input", type=Path, help="summary only: raw JSONL file or directory")
    parser.add_argument("--lengths", type=csv_ints, default=[16, 64, 256, 1024])
    parser.add_argument("--batches", type=csv_ints, default=[1, 4, 8, 16])
    parser.add_argument("--load-lengths", type=csv_ints, default=[16, 256, 1024])
    parser.add_argument("--load-batches", type=csv_ints, default=[1, 16])
    parser.add_argument("--input-type", choices=("query", "document"), default="query")
    parser.add_argument("--include-media", action="store_true", help="also measure image, 2-frame and 8-frame video cases")
    parser.add_argument("--media-dataset", type=Path,
                        help="use real ordered JPEG frames from a prepared video dataset instead of synthetic PNGs")
    parser.add_argument("--concurrencies", type=csv_ints, default=[1, 2, 4, 8, 16])
    parser.add_argument("--case-seconds", type=float, default=10.0)
    parser.add_argument("--min-success", type=int, default=50)
    parser.add_argument("--target-rps", type=float, help="load only: evenly scheduled total arrivals/second")
    parser.add_argument("--max-pending", type=int, default=1000)
    parser.add_argument("--drain-seconds", type=float, default=30.0)
    parser.add_argument("--request-gap", type=float, default=.01, help="minimum per-worker request start interval, >=0.01s")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--soak-seconds", type=float, default=600.0)
    parser.add_argument("--probe-interval", type=float, default=30.0)
    parser.add_argument("--anchor-interval", type=float, default=30.0)
    parser.add_argument("--anchor-cosine", type=float, default=.999)
    args = parser.parse_args(argv)
    if args.mode != "summary" and args.key_file is None:
        parser.error("--key-file is required; never pass the key itself")
    for name in ("case_seconds", "drain_seconds", "request_gap", "timeout", "soak_seconds", "probe_interval", "anchor_interval"):
        number = getattr(args, name)
        if not math.isfinite(number) or number <= 0:
            parser.error(name.replace("_", "-") + " must be positive and finite")
    if args.request_gap < .01 or args.min_success < 1 or args.warmup < 0 or args.max_pending < 1:
        parser.error("request-gap >=0.01, min-success >=1, warmup >=0, max-pending >=1 are required")
    if any(batch > 16 for batch in args.batches + args.load_batches):
        parser.error("the service accepts batches of at most 16")
    if any(length > 8192 for length in args.lengths + args.load_lengths):
        parser.error("the service accepts at most 8192 characters per text")
    title_chars = len(DOCUMENT_TITLE) if args.input_type == "document" else 0
    if any((length + title_chars) * batch > 32768 for length in args.lengths for batch in args.batches):
        parser.error("a latency batch cannot exceed 32768 text and title characters")
    if any((length + title_chars) * batch > 32768 for length in args.load_lengths for batch in args.load_batches):
        parser.error("a load batch cannot exceed 32768 text and title characters")
    if args.target_rps is not None and (not math.isfinite(args.target_rps) or args.target_rps <= 0):
        parser.error("target-rps must be positive and finite")
    if args.target_rps is not None and args.mode != "load":
        parser.error("target-rps applies only to load mode")
    if args.mode == "load" and args.case_seconds < 10:
        parser.error("load cases must run for at least 10 seconds")
    if not math.isfinite(args.anchor_cosine) or not -1 <= args.anchor_cosine <= 1:
        parser.error("anchor-cosine must be between -1 and 1")
    return args


async def run_live(args, output_dir, recorder):
    url, secret, endpoint = read_connection(args.endpoint_file, args.key_file)
    maximum = max(args.concurrencies)
    transport = httpx.AsyncHTTPTransport(retries=0, trust_env=False,
        limits=httpx.Limits(max_connections=max(64, maximum * 2), max_keepalive_connections=max(32, maximum)))
    async with httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                                timeout=httpx.Timeout(args.timeout, connect=min(15.0, args.timeout))) as client:
        benchmark = Benchmark(client, url, secret, endpoint, recorder, args)
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, task.cancel)
            except (NotImplementedError, RuntimeError):
                pass
        recorder.emit({"record_type": "run_start", "mode": args.mode, "model": endpoint.get("model"),
                       "model_revision": endpoint.get("model_revision"), "no_retries": True,
                       "trust_env": False, "follow_redirects": False,
                       "request_gap_s": args.request_gap, "raw_url_omitted": True,
                       "input_type": args.input_type, "include_media": args.include_media,
                       "timeout_s": args.timeout, "drain_seconds": args.drain_seconds,
                       "probe_interval_s": args.probe_interval, "anchor_interval_s": args.anchor_interval,
                       "anchor_cosine": args.anchor_cosine})
        try:
            await benchmark.baseline()
            await getattr(benchmark, args.mode)()
        finally:
            recorder.emit({"record_type": "run_end", "mode": args.mode})


def main(argv=None):
    os.umask(0o077)
    args = parse_args(argv)
    if args.mode == "summary":
        source = args.input or args.output_dir
        if source is None:
            raise SystemExit("summary requires --input or --output-dir")
        paths = sorted(source.glob("raw-*.jsonl")) if source.is_dir() else [source]
        if not paths:
            raise SystemExit("no raw JSONL files found")
        output_dir = args.output_dir or (source if source.is_dir() else source.parent)
        summarize(paths, output_dir)
        print("汇总已保存：" + str((output_dir / "summary.json").resolve()))
        return 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output_dir = args.output_dir or ROOT / "data/autodl/benchmarks" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(output_dir / f"raw-{args.mode}-{run_id}.jsonl", run_id)
    result = 0
    try:
        asyncio.run(run_live(args, output_dir, recorder))
    except (KeyboardInterrupt, asyncio.CancelledError):
        result = 130
        recorder.emit({"record_type": "interrupted", "mode": args.mode})
        print("测评已中断，已完成的原始记录会保留。", file=sys.stderr)
    except Exception as error:
        result = 1
        recorder.emit({"record_type": "run_error", "errorclass": type(error).__name__})
        print("测评失败：" + type(error).__name__ + "，请查看原始记录。", file=sys.stderr)
    finally:
        recorder.close()
        summary = summarize(sorted(output_dir.glob("raw-*.jsonl")), output_dir)
        if args.mode == "sanity" and summary["sanity_passed"] is False and result == 0:
            result = 1
    print("结果目录：" + str(output_dir.resolve()))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
