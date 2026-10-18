"""Observe an existing Colab endpoint and summarize bounded persistent history.

Importing this module has no side effects. ``summary`` never probes a service:
only the explicitly started background worker performs network health checks.
It neither allocates nor renews nor releases Colab sessions.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

MAX_TAIL_BYTES = 8 * 1024 * 1024
MAX_HISTORY_ROWS = 20_000
MAX_ENTRIES = 1000
STALE_AFTER_S = 180
OUTCOMES = {"ok", "busy", "failure", "starting", "auth", "config"}
STATUSES = {"STARTING", "HEALTHY", "RECOVERING", "BUSY", "DEGRADED", "DOWN",
            "AUTH_ERROR", "CONFIG_ERROR", "MONITOR_ERROR", "EXPIRED", "STOPPED", "RELEASE_UNCONFIRMED", "UNKNOWN", "STALE"}


def _timestamp(value) -> datetime | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            return None
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _utc(value=None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return datetime.fromtimestamp(value, timezone.utc)
    raise ValueError("now must be a datetime or finite UTC timestamp")


def _iso(value: datetime | None):
    return value.isoformat(timespec="seconds") if value is not None else None


def _small_json(path: Path) -> dict | None:
    try:
        with path.open("rb") as handle:
            content = handle.read(64 * 1024 + 1)
        if len(content) > 64 * 1024:
            return None
        value = json.loads(content)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError, RecursionError):
        return None


def _label(value, maximum=256):
    # Explicit allowlists prevent prompt/credential/path fields in a damaged
    # record from being copied into an HTTP response.
    if (isinstance(value, str) and len(value) <= maximum
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*(?:/[A-Za-z0-9][A-Za-z0-9_.:-]*)?", value)):
        return value
    return None


def _observation(value, until: datetime) -> tuple[dict, datetime] | None:
    if not isinstance(value, dict):
        return None
    when = _timestamp(value.get("updated_at"))
    if when is None or when > until:
        return None
    status, outcome = value.get("status"), value.get("outcome")
    if not isinstance(status, str) or not isinstance(outcome, str) or status not in STATUSES or outcome not in OUTCOMES:
        return None
    # A malformed success label must not conceal an actual failed observation.
    if status in {"HEALTHY", "RECOVERING"} and outcome != "ok":
        status = "BUSY" if outcome == "busy" else "DEGRADED"
    result = {"updated_at": _iso(when), "status": status, "outcome": outcome, "checks": {}}
    reason = value.get("reason")
    if isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]{1,100}", reason):
        result["reason"] = reason
    for field in ("boot_id", "model", "model_revision"):
        label = _label(value.get(field))
        if label is not None:
            result[field] = label
    for field in ("consecutive_successes", "consecutive_failures", "recovery_threshold", "failure_threshold"):
        count = value.get(field)
        minimum = 1 if field.endswith("threshold") else 0
        if type(count) is int and minimum <= count <= 1_000_000:
            result[field] = count
    deep_time = _timestamp(value.get("last_inference_success_at"))
    if deep_time is not None and deep_time <= until:
        result["last_inference_success_at"] = _iso(deep_time)
    checks = value.get("checks")
    if isinstance(checks, dict):
        for kind in ("live", "ready", "inference"):
            raw = checks.get(kind)
            if not isinstance(raw, dict):
                continue
            check = {}
            code = raw.get("http_status")
            if isinstance(code, int) and not isinstance(code, bool) and 0 <= code <= 599:
                check["http_status"] = code
            latency = raw.get("latency_ms")
            if (isinstance(latency, (int, float)) and not isinstance(latency, bool)
                    and 0 <= latency <= 3_600_000 and math.isfinite(latency)):
                check["latency_ms"] = round(latency, 3)
            if check:
                result["checks"][kind] = check
    return result, when


def _matching_endpoint(value, endpoint: dict | None):
    """Reject known old targets without mistaking static metadata for a reboot.

    URL-less legacy records remain readable. A configured boot is only newer
    than an observed boot when its explicit timestamp proves that ordering;
    local endpoint files may retain their initial boot after a real restart.
    """
    if not isinstance(value, dict) or not isinstance(endpoint, dict):
        return isinstance(value, dict)
    configured_url, observed_url = endpoint.get("url"), value.get("url")
    if isinstance(configured_url, str) and observed_url is not None:
        if not isinstance(observed_url, str) or observed_url.rstrip("/") != configured_url.rstrip("/"):
            return False
    configured_boot, observed_boot = _label(endpoint.get("boot_id")), _label(value.get("boot_id"))
    configured_at, observed_at = _timestamp(endpoint.get("updated_at")), _timestamp(value.get("updated_at"))
    if (configured_boot is not None and observed_boot is not None and configured_boot != observed_boot
            and configured_at is not None and observed_at is not None and observed_at < configured_at):
        return False
    return True


def _percentile(samples: list[float], percentile: float):
    if not samples:
        return None
    values = sorted(samples)
    position = (len(values) - 1) * percentile
    low, high = math.floor(position), math.ceil(position)
    return round(values[low] + (values[high] - values[low]) * (position - low), 2)


class ServiceHealth:
    def __init__(self, data_dir: Path, interval: float = 60, probe_interval: float = 300,
                 *, now: Callable[[], datetime] | None = None, directory: Path | None = None,
                 lease_managed: bool = True, allowed_http_hosts: tuple[str, ...] = ()):
        if not math.isfinite(interval) or interval <= 0 or not math.isfinite(probe_interval) or probe_interval <= 0:
            raise ValueError("health intervals must be positive finite seconds")
        self.directory = Path(directory) if directory is not None else Path(data_dir) / "colab"
        self.endpoint_file = self.directory / "endpoint.json"
        self.key_file = self.directory / "inference.key"
        self.state_file = self.directory / "health.json"
        self.log_file = self.directory / "probe-log.jsonl"
        self.lease_file = self.directory / "lease.json"
        self.interval, self.probe_interval = interval, probe_interval
        self.lease_managed = lease_managed
        self.allowed_http_hosts = tuple(allowed_http_hosts)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._restart_requested = False
        self._last_background_error: datetime | None = None

    @property
    def enabled(self):
        return self.endpoint_file.is_file() and self.key_file.is_file()

    @property
    def monitor_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _lease(self, endpoint: dict | None = None):
        if not self.lease_managed:
            return None, True, None
        lease_exists = self.lease_file.exists()
        lease = _small_json(self.lease_file) if lease_exists else None
        if lease_exists and lease is None:
            return None, False, None
        terminal = lease.get("status") if lease else None
        if not isinstance(terminal, str) or terminal not in {"RELEASED", "RELEASE_UNCONFIRMED", "STOPPED", "EXPIRED"}:
            terminal = None
        record = lease if lease and lease.get("expires_at") is not None else endpoint
        raw = record.get("expires_at") if isinstance(record, dict) else None
        if raw is None:
            return None, True, terminal
        parsed = _timestamp(raw)
        return parsed, parsed is not None, terminal

    def _create_monitor(self):
        # Keep initialization independent of the optional background monitor.
        from scripts.monitor_colab import Monitor, Settings

        return Monitor(Settings(endpoint_file=self.endpoint_file, key_file=self.key_file,
                                state_file=self.state_file, log_file=self.log_file,
                                interval=self.interval, probe_interval=self.probe_interval,
                                allowed_http_hosts=self.allowed_http_hosts),
                       emit=lambda event: None, cancelled=self._stop.is_set)

    def _may_start(self):
        if not self.enabled:
            return False
        expires, valid, terminal = self._lease(_small_json(self.endpoint_file))
        return not terminal and valid and (expires is None or _utc(self._now()) < expires)

    def _spawn_locked(self):
        self._stop.clear()
        self._restart_requested = False
        self._thread = threading.Thread(target=self._run, name="omni-service-health", daemon=True)
        self._thread.start()

    def start(self):
        with self._lock:
            if self.monitor_running:
                if self._stop.is_set() and self._may_start():
                    # A cancelled HTTP call may still be draining. Its worker
                    # starts the replacement only after it has fully stopped.
                    self._restart_requested = True
                return not self._stop.is_set()
            if not self._may_start():
                return False
            self._spawn_locked()
            return True

    def stop(self, timeout: float = 5):
        with self._lock:
            self._restart_requested = False
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0, timeout))

    def _record_error(self):
        self._last_background_error = _utc(self._now())
        observation = {"updated_at": _iso(self._last_background_error), "status": "MONITOR_ERROR",
                       "outcome": "failure", "reason": "monitor_tick_failed", "checks": {}}
        try:
            from scripts.monitor_colab import append_probe_log, atomic_json

            atomic_json(self.state_file, observation)
            append_probe_log(self.log_file, observation)
        except Exception:
            # Unwritable disks are still visible through the in-memory status;
            # exception messages can contain paths or credentials, so omit them.
            pass

    def _run(self):
        try:
            self._worker()
        finally:
            with self._lock:
                if self._restart_requested and self._may_start():
                    self._spawn_locked()

    def _worker(self):
        try:
            monitor = self._create_monitor()
        except Exception:
            if not self._stop.is_set():
                self._record_error()
            return
        while not self._stop.is_set() and self.enabled:
            expires, valid, terminal = self._lease(_small_json(self.endpoint_file))
            current = _utc(self._now())
            if terminal or (expires is not None and current >= expires):
                break
            if valid:
                try:
                    monitor.tick()
                    if not self._stop.is_set():
                        self._last_background_error = None
                except Exception:
                    if not self._stop.is_set():
                        self._record_error()
            else:
                self._record_error()
            wait_s = self.interval
            if expires is not None:
                wait_s = min(wait_s, max(0, (expires - _utc(self._now())).total_seconds()))
            self._stop.wait(wait_s)

    def _history(self, since: datetime, until: datetime, endpoint: dict | None = None):
        try:
            with self.log_file.open("rb") as handle:
                handle.seek(0, 2)
                length = handle.tell()
                offset = max(0, length - MAX_TAIL_BYTES)
                handle.seek(offset)
                if offset:
                    handle.readline(MAX_TAIL_BYTES)
                content = handle.read(MAX_TAIL_BYTES)
        except OSError:
            return [], None, False, []
        rows, current_rows, latest, limited = [], [], None, offset > 0
        for raw in reversed(content.splitlines()):
            try:
                value = json.loads(raw)
                observation = _observation(value, until)
            except (ValueError, UnicodeError, RecursionError):
                continue
            if observation is None:
                continue
            entry, when = observation
            matches = _matching_endpoint(value, endpoint)
            if matches and (latest is None or when > latest[1]):
                latest = observation
            if when < since:
                continue
            rows.append(observation)
            if matches:
                current_rows.append(observation)
            if len(rows) >= MAX_HISTORY_ROWS:
                limited = True
                break
        rows.sort(key=lambda item: item[1], reverse=True)
        current_rows.sort(key=lambda item: item[1], reverse=True)
        return rows, latest, limited, current_rows

    def summary(self, hours: float = 3, limit: int = 180, *, now=None):
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= 24 or not math.isfinite(hours):
            raise ValueError("hours must be between 0 and 24")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_ENTRIES:
            raise ValueError(f"limit must be between 1 and {MAX_ENTRIES}")
        until = _utc(self._now() if now is None else now)
        since = until - timedelta(hours=hours)
        endpoint = _small_json(self.endpoint_file)
        rows, historical_latest, limited, current_rows = self._history(since, until, endpoint)
        expires, valid_lease, terminal = self._lease(endpoint)
        raw_state = _small_json(self.state_file)
        state = _observation(raw_state, until) if _matching_endpoint(raw_state, endpoint) else None
        latest = historical_latest
        if state is not None and (latest is None or state[1] >= latest[1]):
            latest = state
        metadata = latest[0] if latest else {}
        current_boot = metadata.get("boot_id")
        if current_boot is not None:
            current_rows = [(entry, when) for entry, when in current_rows if entry.get("boot_id") == current_boot]
            if historical_latest is not None and historical_latest[0].get("boot_id") != current_boot:
                historical_latest = None
        last_probe = latest[1] if latest else None
        age = round(max(0, (until - last_probe).total_seconds()), 3) if last_probe else None
        history_age = (until - current_rows[0][1]).total_seconds() if current_rows else None
        stale = history_age is None or history_age > STALE_AFTER_S
        if not self.enabled:
            status = "NOT_CONFIGURED"
        elif terminal:
            status = terminal if terminal in {"EXPIRED", "RELEASE_UNCONFIRMED"} else "STOPPED"
        elif not valid_lease:
            status = "CONFIG_ERROR"
        elif expires is not None and until >= expires:
            status = "EXPIRED"
        elif stale:
            status = "STALE" if historical_latest is not None or (age is not None and age > STALE_AFTER_S) else "UNKNOWN"
        else:
            status = latest[0]["status"] if latest else "UNKNOWN"
        if (self._last_background_error is not None and self._last_background_error <= until
                and (last_probe is None or self._last_background_error >= last_probe)
                and status not in {"EXPIRED", "STOPPED", "RELEASE_UNCONFIRMED", "NOT_CONFIGURED"}):
            status = "MONITOR_ERROR"
        observations = [entry for entry, _ in rows]
        successful = sum(entry["outcome"] == "ok" for entry in observations)
        busy = sum(entry["outcome"] == "busy" for entry in observations)
        latencies = [entry["checks"]["ready"]["latency_ms"] for entry in observations
                     if "latency_ms" in entry["checks"].get("ready", {})]
        deep_successes = sum(entry["outcome"] == "ok" and entry["checks"].get("inference", {}).get("http_status") == 200
                             for entry in observations)
        deep_times = [_timestamp(entry.get("last_inference_success_at")) for entry, _ in current_rows]
        if latest:
            deep_times.append(_timestamp(latest[0].get("last_inference_success_at")))
        endpoint = endpoint or {}
        return {
            "window_hours": hours, "since": _iso(since), "until": _iso(until),
            "observed_since": _iso(rows[-1][1]) if rows else None,
            "observed_until": _iso(rows[0][1]) if rows else None,
            "latest_status": status, "last_probe_at": _iso(last_probe), "last_probe_age_s": age,
            "stale": stale, "observed_count": len(observations), "success_count": successful,
            "failure_count": len(observations) - successful - busy, "busy_count": busy,
            "deep_success_count": deep_successes,
            "availability_pct": round(successful / len(observations) * 100, 2) if observations else None,
            "availability_basis": "successful observations / all observations; not a business SLA",
            "latency_p50_ms": _percentile(latencies, 0.5), "latency_p95_ms": _percentile(latencies, 0.95),
            "last_inference_success_at": _iso(max((value for value in deep_times if value is not None), default=None)),
            "boot_id": metadata.get("boot_id") or _label(endpoint.get("boot_id")),
            "model": _label(endpoint.get("model")) or metadata.get("model"),
            "model_revision": _label(endpoint.get("model_revision")) or metadata.get("model_revision"),
            "consecutive_successes": metadata.get("consecutive_successes"),
            "consecutive_failures": metadata.get("consecutive_failures"),
            "recovery_threshold": metadata.get("recovery_threshold"),
            "failure_threshold": metadata.get("failure_threshold"),
            "lease_expires_at": _iso(expires), "monitor_running": self.monitor_running,
            "entries": observations[:limit], "status_counts": dict(Counter(entry["status"] for entry in observations)),
            "history_truncated": limited,
        }


class ServiceHealthManager:
    """Persist monitoring switches while keeping each endpoint's history separate.

    Selecting a tab only selects the history to read. An explicit enable switch
    controls its worker; neither operation provisions or extends an instance.
    """

    PROFILE_LABELS = {"local": "本地电脑", "colab": "Google Colab"}

    def __init__(self, data_dir: Path, interval: float = 60, probe_interval: float = 300,
                 *, now: Callable[[], datetime] | None = None):
        self.data_dir = Path(data_dir)
        self.config_file = self.data_dir / "services" / "profiles.json"
        self.services = {
            "local": ServiceHealth(self.data_dir, interval, probe_interval, now=now,
                                   directory=self.data_dir / "services" / "local", lease_managed=False,
                                   allowed_http_hosts=("192.168.31.204:8770",)),
            "colab": ServiceHealth(self.data_dir, interval, probe_interval, now=now),
        }
        self._lock = threading.RLock()
        self._started = False
        self._config_error = False
        self._enabled = {"local": True, "colab": False}
        self.selected_profile = "local"
        self._load_config()

    def _load_config(self):
        if not self.config_file.exists():
            return
        value = _small_json(self.config_file)
        try:
            if not isinstance(value, dict) or set(value) - {"schema_version", "selected", "profiles"}:
                raise ValueError
            if "schema_version" in value and (type(value["schema_version"]) is not int or value["schema_version"] != 1):
                raise ValueError
            selected = value["selected"]
            profiles = value["profiles"]
            if not isinstance(selected, str) or selected not in self.services or not isinstance(profiles, dict):
                raise ValueError
            if set(profiles) != set(self.services):
                raise ValueError
            enabled = {}
            for name in self.services:
                item = profiles[name]
                if not isinstance(item, dict) or set(item) != {"enabled"} or type(item["enabled"]) is not bool:
                    raise ValueError
                enabled[name] = item["enabled"]
            self.selected_profile, self._enabled = selected, enabled
        except (KeyError, TypeError, ValueError):
            # Damaged switches must not silently turn monitoring back on.
            self._enabled = {name: False for name in self.services}
            self._config_error = True

    def _persist(self, selected: str, enabled: dict[str, bool]):
        value = {"schema_version": 1, "selected": selected,
                 "profiles": {name: {"enabled": enabled[name]} for name in self.services}}
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.config_file.parent,
                                             prefix=".profiles-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.config_file)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _url(service: ServiceHealth, endpoint: dict | None):
        value = endpoint.get("url") if endpoint else None
        if not isinstance(value, str) or not value or len(value) > 2048:
            return None
        try:
            # Shared validation keeps the UI and the actual transport in sync.
            # It never opens a connection or reads the credential file.
            from scripts.monitor_colab import validate_url

            return validate_url(value, service.allowed_http_hosts)
        except (ValueError, TypeError):
            return None

    def _availability(self, name: str, *, now=None):
        service = self.services[name]
        endpoint = _small_json(service.endpoint_file)
        url = self._url(service, endpoint)
        configured = url is not None and service.key_file.is_file()
        if not configured:
            reason = "config_error" if service.endpoint_file.is_file() and (endpoint is None or url is None) else "not_configured"
            return configured, url, reason
        expires, valid, terminal = service._lease(endpoint)
        if not valid:
            reason = "config_error"
        elif terminal == "RELEASE_UNCONFIRMED":
            reason = "release_unconfirmed"
        elif terminal in {"RELEASED", "STOPPED"}:
            reason = "stopped"
        elif terminal == "EXPIRED" or (expires is not None and _utc(service._now() if now is None else now) >= expires):
            reason = "expired"
        else:
            reason = None
        return configured, url, reason

    def _summary(self, name: str, hours: float, limit: int, *, now=None):
        service = self.services[name]
        result = service.summary(hours=hours, limit=limit, now=now)
        if not self._enabled[name]:
            result["latest_status"] = "DISABLED"
        configured, url, reason = self._availability(name, now=now)
        if self._enabled[name] and reason == "config_error":
            result["latest_status"] = "CONFIG_ERROR"
        result["profile"] = {
            "id": name, "label": self.PROFILE_LABELS[name], "enabled": self._enabled[name],
            "configured": configured, "url": url, "lease_managed": service.lease_managed,
            "last_status": result["latest_status"], "monitor_running": service.monitor_running,
            "stopping": service.monitor_running and service._stop.is_set(),
            "can_enable": configured and reason is None, "unavailable_reason": reason,
        }
        return result

    def start(self):
        with self._lock:
            self._started = True
            for name, service in self.services.items():
                _, _, reason = self._availability(name)
                if self._enabled[name] and reason is None:
                    service.start()

    def stop(self, timeout: float = 5):
        with self._lock:
            self._started = False
            services = list(self.services.values())
            # Cancel every profile before waiting for any in-flight request.
            for service in services:
                service.stop(timeout=0)
        for service in services:
            service.stop(timeout=timeout)

    def summary(self, profile: str | None = None, hours: float = 3, limit: int = 180, *, now=None):
        with self._lock:
            name = self.selected_profile if profile is None else profile
            if not isinstance(name, str) or name not in self.services:
                raise ValueError("profile must be local or colab")
            summaries = {item: self._summary(item, hours, limit, now=now) for item in self.services}
            result = summaries[name]
            result["profiles"] = [summaries[item]["profile"] for item in self.services]
            result["selected_profile"] = self.selected_profile
            result["settings_error"] = self._config_error
            return result

    def update(self, profile: str, enabled: bool | None = None, selected: bool | None = None):
        with self._lock:
            if not isinstance(profile, str) or profile not in self.services:
                raise ValueError("profile must be local or colab")
            if enabled is not None and type(enabled) is not bool:
                raise ValueError("enabled must be a boolean")
            if selected is not None and type(selected) is not bool:
                raise ValueError("selected must be a boolean")
            if enabled is None and selected is None:
                raise ValueError("provide enabled or selected")
            if enabled:
                _, _, reason = self._availability(profile)
                if reason is not None:
                    messages = {"not_configured": "服务地址或访问密钥尚未配置", "config_error": "服务配置无效",
                                "expired": "Colab 租约已到期，请先手动部署实例", "stopped": "Colab 实例已释放，请先手动部署实例",
                                "release_unconfirmed": "Colab 实例释放状态尚未确认，请先检查租约"}
                    raise ValueError(messages[reason])
            next_enabled = dict(self._enabled)
            if enabled is not None:
                next_enabled[profile] = enabled
            next_selected = profile if selected else self.selected_profile
            self._persist(next_selected, next_enabled)
            self._enabled, self.selected_profile = next_enabled, next_selected
            self._config_error = False
            service = self.services[profile]
            if enabled is False:
                service.stop(timeout=0)
            elif enabled is True and self._started:
                service.start()
            return self.summary(profile)
