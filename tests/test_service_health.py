"""Read-only history summaries and one bounded worker for the existing service."""
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from omniembed.service_health import MAX_TAIL_BYTES, ServiceHealth

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def configure(tmp_path, *, expires=NOW + timedelta(hours=1)):
    service = ServiceHealth(tmp_path, now=lambda: NOW)
    service.directory.mkdir(parents=True)
    service.endpoint_file.write_text(json.dumps({"url": "https://example.invalid", "boot_id": "boot-a",
                                               "model": "google/embeddinggemma-2", "model_revision": "fixed-sha"}))
    service.key_file.write_text("private-inference-key")
    service.key_file.chmod(0o600)
    if expires is not None:
        service.lease_file.write_text(json.dumps({"expires_at": expires.isoformat()}))
    return service


def row(seconds=0, *, outcome="ok", status=None, latency=10, deep=False, **extra):
    if status is None:
        status = {"ok": "HEALTHY", "busy": "BUSY", "failure": "DOWN", "auth": "AUTH_ERROR",
                  "config": "CONFIG_ERROR", "starting": "STARTING"}[outcome]
    result = {"updated_at": (NOW - timedelta(seconds=seconds)).isoformat(), "status": status,
              "outcome": outcome, "boot_id": "boot-a", "checks": {"ready": {"http_status": 200, "latency_ms": latency}}}
    if deep:
        result["checks"]["inference"] = {"http_status": 200, "latency_ms": 999}
        result["last_inference_success_at"] = result["updated_at"]
    result.update(extra)
    return result


def history(service, rows):
    service.log_file.write_text("\n".join(json.dumps(value) if isinstance(value, dict) else value for value in rows) + "\n")


def wait_for(condition, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("background worker did not reach expected condition")


def test_unconfigured_is_read_only_and_does_not_start(tmp_path):
    service = ServiceHealth(tmp_path, now=lambda: NOW)
    assert not service.start()
    result = service.summary()
    assert result["latest_status"] == "NOT_CONFIGURED"
    assert result["observed_count"] == 0 and result["stale"]
    assert result["availability_pct"] is None
    assert result["latency_p50_ms"] is None and result["latency_p95_ms"] is None
    assert result["last_probe_at"] is None and not result["monitor_running"]
    assert not service.directory.exists()


def test_fresh_state_without_history_never_claims_observed_window(tmp_path):
    service = configure(tmp_path)
    service.state_file.write_text(json.dumps(row(1, deep=True)))
    result = service.summary()
    assert result["latest_status"] == "UNKNOWN" and result["stale"]
    assert result["observed_count"] == 0 and result["entries"] == []
    assert result["observed_since"] is None and result["observed_until"] is None
    assert result["availability_pct"] is None
    assert result["last_probe_age_s"] == 1


def test_three_hour_filter_statistics_and_newest_first_limit(tmp_path):
    service = configure(tmp_path)
    history(service, [row(10801), row(40, outcome="busy", latency=40), "broken json", row(30, latency=30),
                      row(-1), row(20, outcome="failure", latency=20), row(10, status="RECOVERING", latency=10, deep=True),
                      row(10800, latency=50)])
    result = service.summary(hours=3, limit=2)
    assert result["observed_count"] == 5
    assert result["success_count"] == 3 and result["failure_count"] == 1 and result["busy_count"] == 1
    assert result["deep_success_count"] == 1 and result["availability_pct"] == 60
    assert result["latency_p50_ms"] == 30 and result["latency_p95_ms"] == 48
    assert result["observed_since"] == (NOW - timedelta(hours=3)).isoformat(timespec="seconds")
    assert result["observed_until"] == (NOW - timedelta(seconds=10)).isoformat(timespec="seconds")
    assert [entry["outcome"] for entry in result["entries"]] == ["ok", "failure"]
    assert result["latest_status"] == "RECOVERING" and not result["stale"]
    assert result["last_inference_success_at"] == result["observed_until"]
    assert result["status_counts"] == {"HEALTHY": 2, "BUSY": 1, "DOWN": 1, "RECOVERING": 1}
    assert result["model"] == "google/embeddinggemma-2" and result["model_revision"] == "fixed-sha"


def test_staleness_and_empty_expired_window(tmp_path):
    service = configure(tmp_path)
    history(service, [row(180)])
    assert not service.summary()["stale"]
    history(service, [row(181)])
    assert service.summary()["latest_status"] == "STALE"
    history(service, [row(14400)])
    service.state_file.write_text(json.dumps(row(1)))
    result = service.summary()
    assert result["observed_count"] == 0 and result["latest_status"] == "STALE" and result["stale"]
    history(service, [])
    assert service.summary()["latest_status"] == "UNKNOWN"


def test_corrupt_records_and_secret_fields_are_ignored(tmp_path):
    service = configure(tmp_path)
    history(service, [row(1, prompt="secret-prompt", key_file=str(service.key_file), token="private-inference-key",
                          checks={"ready": {"http_status": 200, "latency_ms": float("nan"), "token": "secret-token"}}),
                      row(2, status=[]), row(2, status="DOWN", outcome=[]), "[1,2,3]", "[" * 2000 + "]" * 2000])
    result = service.summary()
    assert result["observed_count"] == 1
    assert result["latency_p50_ms"] is None and result["latency_p95_ms"] is None
    serialized = json.dumps(result)
    for secret in ("secret-prompt", "private-inference-key", "secret-token", str(service.key_file)):
        assert secret not in serialized


def test_summary_never_runs_network_checks_and_survives_restart(tmp_path, monkeypatch):
    service = configure(tmp_path)
    history(service, [row(1, deep=True), row(60, outcome="busy")])

    def unexpected():
        raise AssertionError("summary must never construct a monitor or run a probe")

    monkeypatch.setattr(service, "_create_monitor", unexpected)
    expected = service.summary()
    assert service.summary() == expected
    restarted = ServiceHealth(tmp_path, now=lambda: NOW)
    assert restarted.summary() == expected
    assert not service.monitor_running and not restarted.monitor_running


def test_tail_read_is_bounded_and_damaged_partial_lines_are_ignored(tmp_path):
    service = configure(tmp_path)
    with service.log_file.open("wb") as handle:
        handle.write(b"x" * (MAX_TAIL_BYTES + 100) + b"\n")
        handle.write(json.dumps(row(1)).encode() + b"\n")
    result = service.summary()
    assert result["observed_count"] == 1 and result["history_truncated"]


def test_expired_lease_prevents_start_and_overrides_recent_healthy_log(tmp_path, monkeypatch):
    service = configure(tmp_path, expires=NOW - timedelta(seconds=1))
    history(service, [row(2)])
    monkeypatch.setattr(service, "_create_monitor", lambda: pytest.fail("expired lease must not probe"))
    assert not service.start()
    assert service.summary()["latest_status"] == "EXPIRED"
    assert not service.summary()["monitor_running"]


def test_invalid_lease_is_fail_closed(tmp_path, monkeypatch):
    service = configure(tmp_path)
    service.lease_file.write_text("invalid JSON")
    monkeypatch.setattr(service, "_create_monitor", lambda: pytest.fail("invalid lease must not probe"))
    assert not service.start()
    assert service.summary()["latest_status"] == "CONFIG_ERROR"


def test_early_release_stops_existing_monitor_before_expiry(tmp_path, monkeypatch):
    service = configure(tmp_path)
    service.interval = 0.01
    ticks = []

    class FakeMonitor:
        def tick(self):
            ticks.append(1)
            service.lease_file.write_text(json.dumps({"status": "RELEASED", "expires_at": (NOW + timedelta(hours=1)).isoformat()}))

    monkeypatch.setattr(service, "_create_monitor", lambda: FakeMonitor())
    assert service.start()
    wait_for(lambda: not service.monitor_running)
    assert len(ticks) == 1 and service.summary()["latest_status"] == "STOPPED"
    assert not service.start()


def test_unconfirmed_release_never_claims_the_vm_has_stopped(tmp_path):
    service = configure(tmp_path)
    history(service, [row(1)])
    service.lease_file.write_text(json.dumps({"status": "RELEASE_UNCONFIRMED", "release_verified": False,
                                              "expires_at": (NOW + timedelta(hours=1)).isoformat()}))
    service._last_background_error = NOW
    result = service.summary()
    assert result["latest_status"] == "RELEASE_UNCONFIRMED"
    assert not service.start() and not result["monitor_running"]


def test_worker_has_one_thread_and_stops_when_lease_expires(tmp_path, monkeypatch):
    service = configure(tmp_path)
    service.interval = 0.01
    current, ticks, constructions = [NOW], [], []
    service._now = lambda: current[0]

    class FakeMonitor:
        def tick(self):
            ticks.append(1)
            current[0] = NOW + timedelta(hours=2)

    def make_monitor():
        constructions.append(1)
        return FakeMonitor()

    monkeypatch.setattr(service, "_create_monitor", make_monitor)
    assert service.start()
    service.start()
    wait_for(lambda: not service.monitor_running)
    assert len(constructions) == 1 and len(ticks) == 1
    assert service.summary()["latest_status"] == "EXPIRED"
    service.stop()


def test_worker_rereads_endpoint_and_stop_only_owns_its_thread(tmp_path, monkeypatch):
    service = configure(tmp_path)
    service.interval = 0.01
    seen = []
    other_stop = threading.Event()
    other = threading.Thread(target=lambda: other_stop.wait(2), daemon=True)
    other.start()

    class FakeMonitor:
        def tick(self):
            seen.append(json.loads(service.endpoint_file.read_text())["url"])

    monkeypatch.setattr(service, "_create_monitor", lambda: FakeMonitor())
    try:
        assert service.start() and service.start()
        wait_for(lambda: bool(seen))
        value = json.loads(service.endpoint_file.read_text())
        value["url"] = "https://replacement.invalid"
        service.endpoint_file.write_text(json.dumps(value))
        wait_for(lambda: "https://replacement.invalid" in seen)
        service.stop(timeout=1)
        assert not service.monitor_running and other.is_alive()
    finally:
        service.stop(timeout=1)
        other_stop.set()
        other.join(timeout=1)


def test_background_monitor_reuses_tick_history_and_sanitizes_errors(tmp_path, monkeypatch):
    from scripts import monitor_colab

    service = configure(tmp_path)
    service.interval = 0.02
    requests = []

    def healthy_request(self, url, method, key=None):
        requests.append(url)
        kind = url.rsplit("/", 1)[-1]
        body = {"status": "ok", "ready": True, "boot_id": "boot-a"}
        if kind == "inference":
            body["checks"] = {"shape": [1, 768], "finite": True, "nonzero": True, "unit_norm": True}
        return monitor_colab.HTTPResult(200, body, 12)

    monkeypatch.setattr(monitor_colab.HealthHTTP, "request", healthy_request)
    # The monitor writes wall-clock timestamps, so use real UTC for this integration.
    current = datetime.now(timezone.utc)
    service._now = lambda: current
    service.lease_file.write_text(json.dumps({"expires_at": (current + timedelta(hours=1)).isoformat()}))
    try:
        assert service.start()
        wait_for(lambda: service.log_file.exists() and len(service.log_file.read_text().splitlines()) >= 2)
    finally:
        service.stop(timeout=1)
    result = service.summary(now=datetime.now(timezone.utc))
    assert result["observed_count"] >= 2 and result["success_count"] >= 2
    assert result["latest_status"] == "HEALTHY" and requests
    count = len(requests)
    service.summary(now=datetime.now(timezone.utc))
    assert len(requests) == count

    class BrokenMonitor:
        def tick(self):
            raise RuntimeError("secret credential and /private/key/path")

    monkeypatch.setattr(service, "_create_monitor", lambda: BrokenMonitor())
    try:
        service.start()
        wait_for(lambda: service._last_background_error is not None)
    finally:
        service.stop(timeout=1)
    result = service.summary(now=datetime.now(timezone.utc))
    assert result["latest_status"] == "MONITOR_ERROR"
    assert "secret credential" not in json.dumps(result) and "/private/key/path" not in json.dumps(result)


@pytest.mark.parametrize("kwargs", [{"hours": 0}, {"hours": 25}, {"hours": float("nan")}, {"limit": 0}, {"limit": 1001}])
def test_invalid_summary_bounds_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ServiceHealth(tmp_path).summary(**kwargs)
