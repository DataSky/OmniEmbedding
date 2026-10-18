#!/usr/bin/env python3
"""Observe a Colab inference service from an independent host.

This performs real health checks; it cannot extend a Colab runtime lease.
No tokens, prompts, response bodies, or URL queries are logged. The process
defaults to a finite two-hour session. The first complete successful observation
is healthy; recovery after an actual fault or instance change needs the configured
success threshold. A successful --once exits 0 during that recovery period too.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import http.client
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable


MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_LOG_LINE_BYTES = 64 * 1024
LOG_RETENTION_SECONDS = 24 * 60 * 60
MAX_SAVED_STATE_BYTES = 64 * 1024
SAVED_STATE_MAX_AGE_SECONDS = 180.0
LOG_STATE_FIELDS = frozenset({
    "schema_version", "updated_at", "status", "outcome", "reason", "url", "boot_id",
    "endpoint_updated_at", "consecutive_failures", "consecutive_successes", "failure_threshold",
    "recovery_threshold", "outage_seconds", "last_success_at", "last_inference_success_at",
    "inference_probe_status", "checks", "error", "model", "model_revision",
    "recovery_required",
})


class ConfigurationError(ValueError):
    """A deliberately sanitized configuration error, safe to print."""


class CheckCancelled(Exception):
    """The owner disabled this monitor; discard the unfinished observation."""


def validate_url(value: object, allowed_http_hosts: tuple[str, ...] = ()) -> str:
    if not isinstance(value, str) or not value or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        raise ConfigurationError("endpoint must be a nonempty HTTP(S) URL")
    try:
        parsed = urllib.parse.urlsplit(value)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        raise ConfigurationError("endpoint has an invalid host or port") from None
    if parsed.scheme not in {"http", "https"} or not host:
        raise ConfigurationError("endpoint must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ConfigurationError("endpoint must not contain credentials, query, or fragment")
    if port == 0:
        raise ConfigurationError("endpoint port must be between 1 and 65535")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    # Plain HTTP requires an explicit host AND port, never a private-network
    # wildcard. This exception is configured by the owner for one LAN service.
    authority = f"[{host}]:{port or 80}" if ":" in host else f"{host}:{port or 80}"
    if parsed.scheme == "http" and not loopback and authority not in allowed_http_hosts:
        raise ConfigurationError("remote endpoints must use HTTPS unless the exact HTTP host and port are allowed")
    return value.rstrip("/")


def _boot_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise ConfigurationError("boot_id must contain 1-128 letters, digits, dots, dashes, colons, or underscores")
    return value


@dataclasses.dataclass(frozen=True)
class Endpoint:
    url: str
    boot_id: str | None = None
    updated_at: str | None = None


def load_endpoint(url: str | None, endpoint_file: Path | None,
                  allowed_http_hosts: tuple[str, ...] = ()) -> Endpoint:
    if endpoint_file is None:
        return Endpoint(validate_url(url, allowed_http_hosts))
    try:
        if endpoint_file.stat().st_size > 8192:
            raise ConfigurationError("endpoint file exceeds 8 KiB")
        data = json.loads(endpoint_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ConfigurationError("endpoint file is missing, unreadable, or invalid JSON") from None
    if not isinstance(data, dict):
        raise ConfigurationError("endpoint file must contain a JSON object")
    updated = data.get("updated_at")
    if updated is not None and (not isinstance(updated, str) or len(updated) > 64):
        raise ConfigurationError("endpoint updated_at must be a short timestamp string")
    return Endpoint(validate_url(data.get("url"), allowed_http_hosts), _boot_id(data.get("boot_id")), updated)


def load_key(key_file: Path | None) -> str:
    if key_file is None:
        key = os.environ.get("OMNI_INFERENCE_API_KEY", "").strip()
    else:
        try:
            with key_file.open("r", encoding="utf-8") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                    raise ConfigurationError("key file must be a regular private file (chmod 600)")
                if hasattr(os, "getuid") and info.st_uid != os.getuid():
                    raise ConfigurationError("key file must belong to the current user")
                key = handle.read(8193).strip()
        except (OSError, UnicodeError):
            raise ConfigurationError("key file is missing or unreadable") from None
    if not key or len(key) > 8192 or any(c.isspace() for c in key) or not key.isascii():
        raise ConfigurationError("set OMNI_INFERENCE_API_KEY or provide a valid private --key-file")
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclasses.dataclass
class HTTPResult:
    code: int
    body: dict
    elapsed_ms: float
    error: str | None = None


class HealthHTTP:
    def __init__(self, timeout: float, *, probe_timeout: float | None = None,
                 cancelled: Callable[[], bool] | None = None):
        self.timeout = timeout
        self.probe_timeout = timeout if probe_timeout is None else probe_timeout
        self.cancelled = cancelled or (lambda: False)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, url: str, method: str, key: str | None = None) -> HTTPResult:
        if self.cancelled():
            raise CheckCancelled()
        headers = {"Accept": "application/json", "User-Agent": "omniembedding-health-monitor/1"}
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(url, method=method, headers=headers)
        timeout = self.probe_timeout if urllib.parse.urlsplit(url).path.endswith("/health/inference") else self.timeout
        start = time.monotonic()
        try:
            try:
                if self.cancelled():
                    raise CheckCancelled()
                response = self.opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                code = response.code
                raw = response.read(65537)
            if self.cancelled():
                raise CheckCancelled()
            if len(raw) > 65536:
                return HTTPResult(code, {}, (time.monotonic() - start) * 1000, "response_too_large")
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeError):
                body = {}
            return HTTPResult(code, body if isinstance(body, dict) else {}, (time.monotonic() - start) * 1000)
        except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError):
            # Exceptions can contain a URL or headers. Never put their message in output.
            return HTTPResult(0, {}, (time.monotonic() - start) * 1000, "connection_failed")


@dataclasses.dataclass
class Settings:
    url: str | None = None
    endpoint_file: Path | None = None
    key_file: Path | None = None
    state_file: Path = Path("colab-health.json")
    interval: float = 60.0
    probe_interval: float = 300.0
    timeout: float = 10.0
    failure_threshold: int = 3
    recovery_threshold: int = 2
    startup_grace: float = 120.0
    max_runtime: float = 7200.0
    once: bool = False
    verbose: bool = False
    log_file: Path | None = None
    allowed_http_hosts: tuple[str, ...] = ()
    probe_timeout: float = 45.0


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _log_line(state: dict) -> bytes:
    """Persist the monitor schema only, never arbitrary response/log fields."""
    record = {key: value for key, value in state.items() if key in LOG_STATE_FIELDS}
    checks = record.get("checks")
    if isinstance(checks, dict):
        record["checks"] = {
            kind: {key: value for key, value in check.items() if key in {"http_status", "latency_ms"}}
            for kind, check in checks.items() if kind in {"live", "ready", "inference"} and isinstance(check, dict)
        }
    try:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    except (ValueError, TypeError, UnicodeError):
        raise OSError("invalid probe log record") from None
    if len(line) > MAX_LOG_LINE_BYTES:
        raise OSError("probe log record exceeds size limit")
    return line


def _compact_probe_log(path: Path, new_line: bytes, now: dt.datetime) -> None:
    """Read at most 8 MiB from the tail, retaining complete recent records."""
    budget = MAX_LOG_BYTES - len(new_line)
    source_fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(source_fd, "rb") as source:
        size = os.fstat(source.fileno()).st_size
        start = max(0, size - budget)
        source.seek(start)
        raw_tail = source.read(budget)
    if start:
        # A bounded suffix can begin inside a record. Discard that fragment.
        _, separator, raw_tail = raw_tail.partition(b"\n")
        if not separator:
            raw_tail = b""
    cutoff = now - dt.timedelta(seconds=LOG_RETENTION_SECONDS)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            written = 0
            for raw in io.BytesIO(raw_tail):
                if len(raw) > MAX_LOG_LINE_BYTES or not raw.endswith(b"\n"):
                    continue
                try:
                    record = json.loads(raw)
                    if not isinstance(record, dict):
                        continue
                    timestamp = record.get("updated_at")
                    if not isinstance(timestamp, str) or len(timestamp) > 64:
                        continue
                    observed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if observed.tzinfo is None or observed < cutoff or observed > now + dt.timedelta(minutes=5):
                        continue
                    line = _log_line(record)
                except (ValueError, UnicodeError, OSError):
                    continue
                if written + len(line) <= budget:
                    target.write(line)
                    written += len(line)
            target.write(new_line)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def append_probe_log(path: Path, state: dict, *, prune: bool = False) -> None:
    """Append one sanitized observation, or raise OSError if it cannot persist.

    Files are private, persist across process restarts, and have an 8 MiB cap.
    The owner calls prune every 60 rounds (and on its first round) so expired
    records disappear even when the size cap has not yet been reached.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _log_line(state)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "ab") as target:
        if not stat.S_ISREG(os.fstat(target.fileno()).st_mode):
            raise OSError("probe log must be a regular file")
        os.fchmod(target.fileno(), 0o600)
        compact = prune or os.fstat(target.fileno()).st_size + len(line) > MAX_LOG_BYTES
        if not compact:
            target.write(line)
            target.flush()
            os.fsync(target.fileno())
    if compact:
        _compact_probe_log(path, line, dt.datetime.now(dt.timezone.utc))


class Monitor:
    def __init__(self, settings: Settings, *, clock: Callable[[], float] = time.monotonic,
                 http: HealthHTTP | None = None, emit: Callable[[str], None] = print,
                 cancelled: Callable[[], bool] | None = None):
        self.settings, self.clock, self.emit = settings, clock, emit
        self.cancelled = cancelled or (lambda: False)
        self.log_file = settings.log_file or settings.state_file.parent / "probe-log.jsonl"
        if self.log_file.resolve() == settings.state_file.resolve():
            raise ConfigurationError("log-file and state-file must be different files")
        self._log_rounds = 0
        self.http = http or HealthHTTP(settings.timeout, probe_timeout=settings.probe_timeout,
                                       cancelled=self.cancelled)
        self.endpoint: Endpoint | None = None
        self.boot_id: str | None = None
        self.status = "STARTING"
        self.failures = self.successes = 0
        self.outage_since: float | None = None
        self.last_probe_at: float | None = None
        self.last_probe_ok: float | None = None
        self.last_success_at: str | None = None
        self.last_deep_success_at: str | None = None
        self.recovering = False
        self.probe_status = "NOT_CHECKED"
        self._probe_auth_key: str | None = None
        self._last_emitted_status: str | None = None
        self.last_state: dict = {}
        self._restore_attempted = False
        self._pending_state: dict | None = None

    @staticmethod
    def _saved_time(value: object, latest: dt.datetime) -> dt.datetime | None:
        if not isinstance(value, str) or len(value) > 64:
            return None
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed > latest:
                return None
            return parsed.astimezone(dt.timezone.utc)
        except (ValueError, OverflowError):
            return None

    def _load_saved_state(self, endpoint: Endpoint, timestamp: str) -> None:
        """Read one bounded candidate, without trusting it before live boot validation."""
        if self._restore_attempted:
            return
        self._restore_attempted = True
        try:
            descriptor = os.open(self.settings.state_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                                 | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(descriptor, "rb") as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SAVED_STATE_BYTES:
                    return
                raw = source.read(MAX_SAVED_STATE_BYTES + 1)
            if len(raw) > MAX_SAVED_STATE_BYTES:
                return
            saved = json.loads(raw)
            latest = dt.datetime.fromisoformat(timestamp)
            if not isinstance(saved, dict) or type(saved.get("schema_version")) is not int or saved["schema_version"] != 1:
                return
            updated = self._saved_time(saved.get("updated_at"), latest)
            if (saved.get("url") != endpoint.url or updated is None
                    or (latest - updated).total_seconds() > SAVED_STATE_MAX_AGE_SECONDS):
                return
            boot = _boot_id(saved.get("boot_id"))
            if boot is None:
                return
            expected_outcomes = {"HEALTHY": {"ok"}, "RECOVERING": {"ok"}, "BUSY": {"busy"},
                                 "DOWN": {"failure", "starting"}, "DEGRADED": {"failure", "starting"},
                                 "AUTH_ERROR": {"auth"}, "CONFIG_ERROR": {"config"}, "STARTING": {"starting"}}
            if saved.get("outcome") not in expected_outcomes.get(saved.get("status"), set()):
                return
            for field, minimum in (("consecutive_failures", 0), ("consecutive_successes", 0),
                                   ("failure_threshold", 1), ("recovery_threshold", 1)):
                value = saved.get(field)
                if type(value) is not int or not minimum <= value <= 1_000_000:
                    return
            if "recovery_required" in saved and type(saved["recovery_required"]) is not bool:
                return
            probe_status = saved.get("inference_probe_status")
            if probe_status is not None and (not isinstance(probe_status, str)
                    or probe_status not in {"OK", "FAILURE", "AUTH", "BUSY", "NOT_CHECKED"}):
                return
            if saved["outcome"] == "ok" and saved["consecutive_failures"] != 0:
                return
            # A known change in endpoint metadata is a restart even if the
            # observer process also restarted. Never revive its old counters.
            metadata_updated = self._saved_time(endpoint.updated_at, latest)
            if (endpoint.boot_id is not None and endpoint.boot_id != boot
                    and metadata_updated is not None and metadata_updated > updated):
                self.boot_id = boot
                return
            self._pending_state = saved
        except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
            return  # Persisted files are untrusted observations, never instructions.

    def _resume_after_live(self, response: HTTPResult, now: float, timestamp: str) -> None:
        saved, self._pending_state = self._pending_state, None
        if saved is None:
            return
        latest = dt.datetime.fromisoformat(timestamp)
        updated = self._saved_time(saved["updated_at"], latest)
        if updated is None or (latest - updated).total_seconds() > SAVED_STATE_MAX_AGE_SECONDS:
            return
        try:
            boot = _boot_id(response.body.get("boot_id"))
        except ConfigurationError:
            boot = None
        needs_recovery = (saved["status"] not in {"HEALTHY", "BUSY"}
                          or saved.get("recovery_required") is True
                          or saved["status"] == "BUSY" and saved.get("inference_probe_status") in {"FAILURE", "AUTH"})
        if boot is None or self._result(response, "live")[0] != "ok":
            # A restart of the observer cannot erase an ongoing outage merely
            # because the unreachable service cannot repeat its boot_id. Keep
            # only conservative fault evidence; successful probes and success
            # streaks still require a current matching boot before reuse.
            if needs_recovery:
                self.boot_id = saved["boot_id"]
                self.failures = saved["consecutive_failures"]
                self.successes = 0
                self.recovering = True
                self.status = saved["status"]
                self._last_emitted_status = self.status
                self._restore_outage(saved, now, latest, updated)
            return
        # Seed the saved identity so _observe_boot still detects a new instance,
        # even when static endpoint metadata already names that new instance.
        if boot != saved["boot_id"]:
            self.boot_id = saved["boot_id"]
            return
        self.boot_id = boot  # Fresh live evidence supersedes static/stale metadata.
        self.status = saved["status"]
        self.failures = saved["consecutive_failures"]
        self.successes = saved["consecutive_successes"]
        self.recovering = needs_recovery
        self._last_emitted_status = self.status
        for field, attr in (("last_success_at", "last_success_at"),
                            ("last_inference_success_at", "last_deep_success_at")):
            observed = self._saved_time(saved.get(field), updated)
            if observed is not None:
                setattr(self, attr, observed.isoformat(timespec="seconds"))
        # Only an explicitly successful real probe can be reused. Failed,
        # busy or unknown probes are re-run now, rather than cleared by ready.
        deep = self._saved_time(saved.get("last_inference_success_at"), updated)
        if saved.get("inference_probe_status") == "OK" and deep is not None:
            elapsed = (latest - deep).total_seconds()
            self.last_probe_at = self.last_probe_ok = now - elapsed
            self.probe_status = "OK"
        self._restore_outage(saved, now, latest, updated)

    def _restore_outage(self, saved: dict, now: float, latest: dt.datetime, updated: dt.datetime) -> None:
        duration = saved.get("outage_seconds")
        if (self.recovering and isinstance(duration, (int, float)) and not isinstance(duration, bool)
                and 0 <= duration <= 31 * 24 * 60 * 60 and math.isfinite(duration)):
            self.outage_since = now - duration - (latest - updated).total_seconds()

    def _check_cancelled(self) -> None:
        if self.cancelled():
            raise CheckCancelled()

    def _event(self, event: str, now: str, **fields) -> None:
        self.emit(json.dumps({"event": event, "updated_at": now, **fields}, ensure_ascii=False))

    def _identity(self, endpoint: Endpoint, timestamp: str) -> None:
        changed = self.endpoint is not None and endpoint.url != self.endpoint.url
        metadata_changed = self.endpoint is None or endpoint.boot_id != self.endpoint.boot_id
        restarted = metadata_changed and endpoint.boot_id is not None and self.boot_id is not None and endpoint.boot_id != self.boot_id
        if changed or restarted:
            self.failures = self.successes = 0
            self.last_probe_at = self.last_probe_ok = None
            self.last_deep_success_at = None
            self.probe_status = "NOT_CHECKED"
            self._probe_auth_key = None
            self.recovering = True
            # Do not reset outage_since: endpoint churn must not indefinitely hide an outage.
            self._event("RESTARTED" if restarted else "ENDPOINT_CHANGED", timestamp,
                        url=endpoint.url, boot_id=endpoint.boot_id)
        if endpoint.boot_id is not None and metadata_changed:
            self.boot_id = endpoint.boot_id
        self.endpoint = endpoint

    def _observe_boot(self, response: HTTPResult, timestamp: str) -> None:
        raw = response.body.get("boot_id")
        if raw is None:
            return
        try:
            value = _boot_id(raw)
        except ConfigurationError:
            return  # Untrusted response data is neither a log message nor a configuration change.
        if self.boot_id is not None and value != self.boot_id:
            self.failures = self.successes = 0
            self.last_probe_at = self.last_probe_ok = None
            self.last_deep_success_at = None
            self.probe_status = "NOT_CHECKED"
            self._probe_auth_key = None
            self.recovering = True
            self._event("RESTARTED", timestamp, url=self.endpoint.url, boot_id=value)
        self.boot_id = value

    @staticmethod
    def _result(response: HTTPResult, kind: str) -> tuple[str, str]:
        if response.code in {401, 403}:
            return "auth", f"{kind}_unauthorized"
        if response.code == 429:
            return "busy", f"{kind}_busy"
        if (response.code == 503 and kind == "inference" and response.error is None
                and response.body.get("status") == "busy"
                and response.body.get("error") == "InferenceBusy"):
            return "busy", "inference_busy"
        if response.code == 503 and kind == "ready":
            phase = response.body.get("status")
            if phase == "loading":
                return "starting", "model_not_ready"
            if phase == "stalled":
                return "failure", "ready_inference_stalled"
            if phase == "degraded":
                return "failure", "model_degraded"
        if response.code == 200 and response.error is None and response.body.get("status") == "ok":
            if kind == "ready" and response.body.get("ready") is not True:
                return "failure", f"{kind}_invalid_readiness"
            if kind == "inference":
                body, checks = response.body, response.body.get("checks")
                # Colab reports ready + the actual shape; the project API reports
                # ok + a boolean shape check. Accept either complete contract,
                # while explicit negative/invalid flags always fail closed.
                flags_valid = all(body[flag] is True for flag in ("ready", "ok") if flag in body)
                shape = checks.get("shape") if isinstance(checks, dict) else None
                legacy_shape = (isinstance(shape, list) and len(shape) == 2
                                and all(type(value) is int for value in shape) and shape == [1, 768])
                legacy_valid = body.get("ready") is True and legacy_shape
                project_valid = (body.get("ok") is True and shape is True
                                 and type(body.get("dimensions")) is int and body["dimensions"] == 768
                                 and body.get("task") == "query")
                if (not flags_valid or not isinstance(checks, dict) or not (legacy_valid or project_valid)
                        or any(checks.get(field) is not True for field in ("finite", "nonzero", "unit_norm"))):
                    return "failure", "inference_invalid_self_test"
            return "ok", "checks_passed"
        if 300 <= response.code < 400:
            return "failure", f"{kind}_redirect_rejected"
        return "failure", f"{kind}_{response.error or 'check_failed'}"

    def _checks(self, endpoint: Endpoint, key: str, now: float, timestamp: str) -> tuple[str, str, dict]:
        self._check_cancelled()
        checks = {}
        degraded_readiness = False
        if self._probe_auth_key is not None and key != self._probe_auth_key:
            self._probe_auth_key = None
            self.last_probe_at = None
            self.probe_status = "NOT_CHECKED"
        for kind, method, authenticated in [("live", "GET", False), ("ready", "GET", True)]:
            self._check_cancelled()
            response = self.http.request(f"{endpoint.url}/health/{kind}", method, key if authenticated else None)
            self._check_cancelled()
            checks[kind] = {"http_status": response.code, "latency_ms": round(response.elapsed_ms, 2)}
            if kind == "live":
                self._resume_after_live(response, now, timestamp)
            self._observe_boot(response, timestamp)
            outcome, reason = self._result(response, kind)
            if kind == "ready" and response.code == 503 and response.body.get("status") == "degraded":
                # A transient model error clears API readiness. Its deep probe
                # is the recovery operation, so readiness must not gate it out.
                degraded_readiness = True
                continue
            if outcome != "ok":
                return outcome, reason, checks
        if (self._probe_auth_key is not None and self.last_probe_at is not None
                and now - self.last_probe_at < self.settings.probe_interval):
            return "auth", "inference_probe_unauthorized", checks
        if self.last_probe_at is None or now - self.last_probe_at >= self.settings.probe_interval:
            self._check_cancelled()
            # A temporary gateway 403 can recover without token rotation. Retry
            # only after the deep-check cooldown, then observe the new result.
            self._probe_auth_key = None
            response = self.http.request(f"{endpoint.url}/health/inference", "POST", key)
            self._check_cancelled()
            self.last_probe_at = now
            checks["inference"] = {"http_status": response.code, "latency_ms": round(response.elapsed_ms, 2)}
            self._observe_boot(response, timestamp)
            outcome, reason = self._result(response, "inference")
            self.probe_status = outcome.upper()
            if outcome != "ok":
                if outcome == "auth":
                    self._probe_auth_key = key
                return outcome, reason, checks
            self.last_probe_ok = now
            self.last_deep_success_at = timestamp
        elif degraded_readiness:
            # A previous successful probe cannot override current degradation.
            # Retry on the normal deep-check cadence instead of hammering a GPU.
            return "failure", "model_degraded_pending_probe", checks
        elif self.probe_status == "FAILURE":
            # Lightweight success cannot clear an actual inference failure.
            return "failure", "inference_probe_pending_retry", checks
        elif self.probe_status == "BUSY" and self.last_probe_ok is None:
            return "busy", "inference_probe_pending_retry", checks
        return "ok", "checks_passed", checks

    def tick(self) -> dict:
        if self.cancelled():
            return self.last_state
        now = self.clock()
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        checks, error = {}, None
        try:
            endpoint = load_endpoint(self.settings.url, self.settings.endpoint_file, self.settings.allowed_http_hosts)
            self._check_cancelled()
            self._load_saved_state(endpoint, timestamp)
            self._identity(endpoint, timestamp)
            self._check_cancelled()
            key = load_key(self.settings.key_file)
            self._check_cancelled()
            outcome, reason, checks = self._checks(endpoint, key, now, timestamp)
            self._check_cancelled()
        except CheckCancelled:
            return self.last_state
        except ConfigurationError as exc:
            self._pending_state = None
            outcome, reason, error = "config", "configuration_error", str(exc)
        if self.cancelled():
            return self.last_state
        if outcome == "ok":
            self.failures = 0
            self.successes += 1
            self.outage_since = None
            self.last_success_at = timestamp
            self.status = ("HEALTHY" if not self.recovering or self.successes >= self.settings.recovery_threshold
                           else "RECOVERING")
        elif outcome == "busy":
            self.failures = self.successes = 0
            self.outage_since = None
            self.status = "BUSY"
        else:
            self.successes = 0
            self.recovering = True
            if self.outage_since is None:
                self.outage_since = now
            unavailable_for = max(0, now - self.outage_since)
            if outcome == "auth":
                self.status = "AUTH_ERROR"
            elif outcome == "config":
                self.status = "CONFIG_ERROR"
            elif outcome == "starting" and unavailable_for < self.settings.startup_grace:
                self.status = "STARTING"
                self.failures = 0
            else:
                self.failures += 1
                # A time bound supplements consecutive counts if a runtime/URL changes each round.
                grace = self.settings.startup_grace if outcome == "starting" else 0.0
                deadline = grace + self.settings.interval * max(0, self.settings.failure_threshold - 1)
                down = self.failures >= self.settings.failure_threshold or unavailable_for >= deadline
                self.status = "DOWN" if down else "DEGRADED"
        outage_seconds = None if self.outage_since is None else round(max(0, now - self.outage_since), 2)
        state = {
            "schema_version": 1, "updated_at": timestamp, "status": self.status,
            "outcome": outcome, "reason": reason,
            "url": self.endpoint.url if self.endpoint else None, "boot_id": self.boot_id,
            "endpoint_updated_at": self.endpoint.updated_at if self.endpoint else None,
            "consecutive_failures": self.failures, "consecutive_successes": self.successes,
            "failure_threshold": self.settings.failure_threshold, "recovery_threshold": self.settings.recovery_threshold,
            "outage_seconds": outage_seconds, "last_success_at": self.last_success_at,
            "last_inference_success_at": self.last_deep_success_at, "inference_probe_status": self.probe_status,
            "checks": checks,
            "recovery_required": self.recovering and self.status != "HEALTHY",
        }
        if error is not None:
            state["error"] = error
        if self.cancelled():
            return self.last_state
        try:
            append_probe_log(self.log_file, state, prune=self._log_rounds % 60 == 0)
        except OSError:
            # A successful HTTP sample must not conceal a broken audit trail.
            self.status = "CONFIG_ERROR"
            self.successes = 0
            self.recovering = True
            state.update(status=self.status, outcome="config", reason="cannot_write_probe_log",
                         consecutive_successes=0, error="probe log could not be persisted")
            self.last_state = state
            atomic_json(self.settings.state_file, state)
            if self.status != self._last_emitted_status:
                self._event(self.status, timestamp, status=self.status, reason=state["reason"], boot_id=self.boot_id)
                self._last_emitted_status = self.status
            raise OSError("cannot persist health probe log") from None
        self._log_rounds += 1
        atomic_json(self.settings.state_file, state)
        if self.status != self._last_emitted_status:
            event = "RECOVERED" if self.status == "HEALTHY" and self.recovering else self.status
            self._event(event, timestamp, status=self.status, reason=reason, boot_id=self.boot_id)
            self._last_emitted_status = self.status
            if self.status == "HEALTHY":
                self.recovering = False
        elif self.settings.verbose:
            self._event("CHECK", timestamp, status=self.status, reason=reason)
        self.last_state = state
        return state


def _positive(value: str) -> float:
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive number") from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="HTTPS service base URL (HTTP allowed on loopback only)")
    source.add_argument("--endpoint-file", type=Path, help="reloaded each round; JSON {url,boot_id?,updated_at?}")
    parser.add_argument("--key-file", type=Path, help="private key file (chmod 600); otherwise use OMNI_INFERENCE_API_KEY")
    parser.add_argument("--state-file", type=Path, default=Path("colab-health.json"))
    parser.add_argument("--log-file", type=Path, help="private per-round JSONL; defaults to state-file directory/probe-log.jsonl")
    parser.add_argument("--interval", type=_positive, default=60.0, help="light check interval in seconds (default: 60)")
    parser.add_argument("--probe-interval", type=_positive, default=300.0, help="real inference probe interval (default: 300)")
    parser.add_argument("--timeout", type=_positive, default=10.0, help="per-request network timeout in seconds")
    parser.add_argument("--probe-timeout", type=_positive, default=45.0, help="real inference request timeout in seconds (default: 45)")
    parser.add_argument("--failure-threshold", type=_positive_int, default=3)
    parser.add_argument("--recovery-threshold", type=_positive_int, default=2)
    parser.add_argument("--startup-grace", type=_positive, default=120.0, help="bounded 503 startup grace in seconds")
    parser.add_argument("--max-runtime", type=_positive, default=7200.0, help="finite session duration in seconds (default: 7200)")
    parser.add_argument("--once", action="store_true", help="one complete check; exit 0 on success, 2 on busy/loading, 3 on auth, 4 on config")
    parser.add_argument("--verbose", action="store_true", help="print every round, without request/response bodies")
    settings = Settings(**vars(parser.parse_args(argv)))
    if settings.log_file is None:
        settings.log_file = settings.state_file.parent / "probe-log.jsonl"
    if settings.log_file.resolve() == settings.state_file.resolve():
        parser.error("log-file and state-file must be different files")
    return settings


def main(argv: list[str] | None = None) -> int:
    settings = parse_args(argv)
    stop = threading.Event()
    monitor = Monitor(settings, cancelled=stop.is_set)
    interrupted = 0

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = signum
        stop.set()

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request_stop)
    started = time.monotonic()
    try:
        while not stop.is_set() and time.monotonic() - started < settings.max_runtime:
            round_started = time.monotonic()
            try:
                state = monitor.tick()
            except OSError:
                # A stale/missing state file must never silently masquerade as successful monitoring.
                print(json.dumps({"event": "MONITOR_ERROR", "reason": "cannot_write_monitor_artifact"}), file=sys.stderr)
                return 4
            if stop.is_set():
                break
            if settings.once:
                return {"ok": 0, "busy": 2, "starting": 2, "auth": 3, "config": 4}.get(state["outcome"], 1)
            wait = min(max(0, settings.interval - (time.monotonic() - round_started)),
                       max(0, settings.max_runtime - (time.monotonic() - started)))
            stop.wait(wait)
        if interrupted:
            return 128 + interrupted
        return 0 if monitor.last_state.get("outcome") in {"ok", "busy"} else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
