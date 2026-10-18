"""Independent durable switches for local and leased Colab observations."""
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from omniembed.service_health import ServiceHealthManager


NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def configure(tmp_path, *, expires=NOW + timedelta(hours=1)):
    manager = ServiceHealthManager(tmp_path, interval=0.01, now=lambda: NOW)
    for name, service in manager.services.items():
        service.directory.mkdir(parents=True)
        endpoint = {"url": "http://192.168.31.204:8770" if name == "local" else "https://colab.invalid",
                    "boot_id": f"{name}-boot", "expires_at": expires.isoformat()}
        service.endpoint_file.write_text(json.dumps(endpoint))
        service.key_file.write_text(f"private-{name}-credential")
        service.key_file.chmod(0o600)
    return manager


def history(service, *, seconds=1, status="HEALTHY"):
    service.log_file.write_text(json.dumps({"updated_at": (NOW - timedelta(seconds=seconds)).isoformat(),
                                            "status": status, "outcome": "ok" if status == "HEALTHY" else "failure",
                                            "checks": {"ready": {"http_status": 200}}}) + "\n")


def wait_for(condition, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("worker did not reach the expected state")


def test_defaults_are_local_on_colab_off_without_creating_files(tmp_path):
    manager = ServiceHealthManager(tmp_path, now=lambda: NOW)
    result = manager.summary()
    assert result["selected_profile"] == "local" and result["profile"]["id"] == "local"
    assert result["latest_status"] == "NOT_CONFIGURED"
    profiles = {value["id"]: value for value in result["profiles"]}
    assert profiles["local"]["enabled"] and not profiles["local"]["can_enable"]
    assert not profiles["colab"]["enabled"] and profiles["colab"]["last_status"] == "DISABLED"
    assert not manager.config_file.exists() and not (tmp_path / "colab").exists()


def test_local_has_its_own_history_and_ignores_every_colab_lease(tmp_path):
    manager = configure(tmp_path, expires=NOW - timedelta(hours=1))
    local, colab = manager.services["local"], manager.services["colab"]
    local.lease_file.write_text("malformed lease that must not apply to local")
    colab.lease_file.write_text(json.dumps({"status": "RELEASE_UNCONFIRMED"}))
    history(local)
    history(colab, seconds=300, status="DOWN")
    result = manager.summary("local")
    assert result["latest_status"] == "HEALTHY" and result["profile"]["can_enable"]
    assert result["lease_expires_at"] is None and not result["profile"]["lease_managed"]
    assert result["observed_count"] == 1 and result["failure_count"] == 0
    colab_result = manager.summary("colab")
    assert colab_result["latest_status"] == "DISABLED"
    assert colab_result["profile"]["unavailable_reason"] == "release_unconfirmed"
    assert colab_result["failure_count"] == 1 and colab_result["stale"]


def test_switches_and_selected_tab_persist_independently(tmp_path):
    manager = configure(tmp_path)
    manager.update("colab", enabled=True)
    manager.update("local", enabled=False)
    result = manager.update("colab", selected=True)
    assert result["selected_profile"] == "colab"
    assert not result["profiles"][0]["enabled"] and result["profiles"][1]["enabled"]
    result = manager.update("local", selected=True)
    assert result["latest_status"] == "DISABLED" and result["selected_profile"] == "local"
    restarted = ServiceHealthManager(tmp_path, now=lambda: NOW)
    result = restarted.summary()
    assert result["profile"]["id"] == "local" and result["latest_status"] == "DISABLED"
    assert result["profiles"][1]["enabled"]
    stored = json.loads(manager.config_file.read_text())
    assert stored == {"schema_version": 1, "selected": "local", "profiles": {
        "local": {"enabled": False}, "colab": {"enabled": True}}}
    assert "credential" not in json.dumps(stored) and "http" not in json.dumps(stored)


@pytest.mark.parametrize("kwargs", [{"enabled": 1}, {"enabled": "false"}, {"selected": 1}, {"selected": "colab"}, {}])
def test_update_rejects_invalid_flags_without_persisting(tmp_path, kwargs):
    manager = configure(tmp_path)
    with pytest.raises(ValueError):
        manager.update("local", **kwargs)
    assert not manager.config_file.exists() and manager.summary()["profile"]["enabled"]


@pytest.mark.parametrize("profile", ["../../colab", "unknown", 1, [], None])
def test_profile_names_are_strict_allowlist(tmp_path, profile):
    manager = configure(tmp_path)
    with pytest.raises(ValueError):
        manager.update(profile, enabled=False)
    if profile is not None:
        with pytest.raises(ValueError):
            manager.summary(profile)


@pytest.mark.parametrize("content", ["bad json", "[]", '{"selected":"local","profiles":{"local":{"enabled":1}}}',
                                     '{"selected":"colab","profiles":{"local":{"enabled":true},"colab":{"enabled":true}},"url":"https://injected.invalid"}'])
def test_damaged_configuration_fails_closed_without_silently_reenabling(tmp_path, content):
    manager = configure(tmp_path)
    manager.config_file.parent.mkdir(parents=True, exist_ok=True)
    manager.config_file.write_text(content)
    restarted = ServiceHealthManager(tmp_path, now=lambda: NOW)
    result = restarted.summary()
    assert result["settings_error"]
    assert all(not profile["enabled"] for profile in result["profiles"])
    assert result["latest_status"] == "DISABLED"
    restarted.update("local", enabled=True)
    assert not restarted.summary()["settings_error"]


def test_expired_colab_cannot_be_enabled_but_history_tab_can_be_selected(tmp_path):
    manager = configure(tmp_path, expires=NOW - timedelta(seconds=1))
    with pytest.raises(ValueError, match="租约已到期"):
        manager.update("colab", enabled=True)
    assert not manager.config_file.exists()
    result = manager.update("colab", selected=True)
    assert result["latest_status"] == "DISABLED" and not result["profile"]["can_enable"]
    assert result["profile"]["unavailable_reason"] == "expired"
    assert result["profile"]["lease_managed"]


@pytest.mark.parametrize("endpoint", ["http://192.168.31.205:8770", "http://192.168.31.204:8771",
                                      "https://user:private@host.invalid", "https://host.invalid?token=private"])
def test_invalid_endpoint_cannot_be_enabled_or_exposed(tmp_path, endpoint):
    manager = configure(tmp_path)
    manager.services["local"].endpoint_file.write_text(json.dumps({"url": endpoint}))
    with pytest.raises(ValueError, match="配置无效"):
        manager.update("local", enabled=True)
    result = manager.summary("local")
    assert result["profile"]["url"] is None and result["latest_status"] == "CONFIG_ERROR"
    assert "private" not in json.dumps(result)


def test_summary_never_reads_credentials_or_constructs_monitor(tmp_path, monkeypatch):
    manager = configure(tmp_path)
    for service in manager.services.values():
        history(service)
        monkeypatch.setattr(service, "_create_monitor", lambda: pytest.fail("summary must not probe"))
    from pathlib import Path
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name == "inference.key":
            pytest.fail("history summaries must not read credentials")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    before = {name: service.log_file.read_bytes() for name, service in manager.services.items()}
    for _ in range(2):
        result = manager.summary()
        assert result["latest_status"] == "HEALTHY"
        assert "private-local-credential" not in json.dumps(result)
        assert "inference.key" not in json.dumps(result)
    assert before == {name: service.log_file.read_bytes() for name, service in manager.services.items()}


def test_failed_save_does_not_change_running_flags(tmp_path, monkeypatch):
    manager = configure(tmp_path)

    def failure(*args):
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_persist", failure)
    with pytest.raises(OSError):
        manager.update("local", enabled=False)
    assert manager.summary()["profile"]["enabled"] and manager.selected_profile == "local"


def test_disabling_inflight_probe_cancels_followup_requests_and_writes_no_fake_failure(tmp_path, monkeypatch):
    from scripts import monitor_colab

    manager = configure(tmp_path)
    entered, release = threading.Event(), threading.Event()
    requests = []

    def blocked_request(self, url, method, key=None):
        requests.append(url)
        entered.set()
        assert release.wait(2)
        return monitor_colab.HTTPResult(200, {"status": "ok", "boot_id": "local-boot"}, 1)

    monkeypatch.setattr(monitor_colab.HealthHTTP, "request", blocked_request)
    local = manager.services["local"]
    try:
        manager.start()
        assert entered.wait(2)
        result = manager.update("local", enabled=False)
        assert result["latest_status"] == "DISABLED"
        assert result["profile"]["stopping"] and result["profile"]["monitor_running"]
        release.set()
        wait_for(lambda: not local.monitor_running)
        assert requests == ["http://192.168.31.204:8770/health/live"]
        assert not local.log_file.exists() and not local.state_file.exists()
        assert not manager.services["colab"].monitor_running
    finally:
        release.set()
        manager.stop(timeout=1)


def test_fast_disable_enable_waits_for_old_worker_before_restarting(tmp_path, monkeypatch):
    manager = configure(tmp_path)
    local = manager.services["local"]
    entered, release, replacement = threading.Event(), threading.Event(), threading.Event()
    constructions, active, maximum = [], [0], [0]

    class FakeMonitor:
        def __init__(self, number):
            self.number = number

        def tick(self):
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
            try:
                if self.number == 0:
                    entered.set()
                    assert release.wait(2)
                else:
                    replacement.set()
            finally:
                active[0] -= 1

    def create():
        number = len(constructions)
        constructions.append(number)
        return FakeMonitor(number)

    monkeypatch.setattr(local, "_create_monitor", create)
    try:
        manager.start()
        assert entered.wait(2)
        manager.update("local", enabled=False)
        manager.update("local", enabled=True)
        assert constructions == [0]
        release.set()
        assert replacement.wait(2)
        assert constructions == [0, 1] and maximum[0] == 1
        manager.update("local", enabled=False)
        wait_for(lambda: not local.monitor_running)
        assert manager.summary()["latest_status"] == "DISABLED"
    finally:
        release.set()
        manager.stop(timeout=1)


def test_recovery_progress_comes_from_latest_observation_not_window_success_rate(tmp_path):
    manager = configure(tmp_path)
    local = manager.services["local"]
    row = {"updated_at": (NOW - timedelta(seconds=1)).isoformat(), "status": "RECOVERING", "outcome": "ok",
           "url": "http://192.168.31.204:8770", "boot_id": "local-boot", "checks": {},
           "consecutive_successes": 1, "recovery_threshold": 2, "consecutive_failures": 0, "failure_threshold": 3}
    local.log_file.write_text(json.dumps(row) + "\n")
    result = manager.summary("local")
    assert result["latest_status"] == "RECOVERING" and result["availability_pct"] == 100
    assert result["consecutive_successes"] == 1 and result["recovery_threshold"] == 2
    assert result["consecutive_failures"] == 0 and result["failure_threshold"] == 3
    manager.update("local", enabled=False)
    result = manager.summary("local")
    assert result["latest_status"] == "DISABLED" and result["consecutive_successes"] == 1


@pytest.mark.parametrize("invalid", [True, False, -1, 1_000_001, 1.5, "2", [], None])
def test_recovery_progress_rejects_invalid_counter_values(tmp_path, invalid):
    manager = configure(tmp_path)
    row = {"updated_at": (NOW - timedelta(seconds=1)).isoformat(), "status": "RECOVERING", "outcome": "ok",
           "checks": {}, "consecutive_successes": invalid, "recovery_threshold": invalid,
           "consecutive_failures": invalid, "failure_threshold": invalid}
    manager.services["local"].log_file.write_text(json.dumps(row) + "\n")
    result = manager.summary("local")
    for field in ("consecutive_successes", "consecutive_failures", "recovery_threshold", "failure_threshold"):
        assert result[field] is None and field not in result["entries"][0]


def test_old_url_history_does_not_make_reconfigured_target_healthy(tmp_path):
    manager = configure(tmp_path)
    local = manager.services["local"]
    row = {"updated_at": (NOW - timedelta(seconds=1)).isoformat(), "status": "HEALTHY", "outcome": "ok",
           "url": "https://old-target.invalid", "boot_id": "old-boot", "checks": {},
           "last_inference_success_at": (NOW - timedelta(seconds=1)).isoformat(),
           "consecutive_successes": 20, "recovery_threshold": 2}
    local.log_file.write_text(json.dumps(row) + "\n")
    local.state_file.write_text(json.dumps(row))
    result = manager.summary("local")
    assert result["latest_status"] == "UNKNOWN" and result["stale"]
    assert result["observed_count"] == 1 and result["success_count"] == 1
    assert result["last_probe_at"] is None and result["last_inference_success_at"] is None
    assert result["boot_id"] == "local-boot" and result["consecutive_successes"] is None


def test_new_configured_boot_does_not_inherit_preexisting_deep_success(tmp_path):
    manager = configure(tmp_path)
    local = manager.services["local"]
    endpoint = json.loads(local.endpoint_file.read_text())
    endpoint["boot_id"] = "next-boot"
    endpoint["updated_at"] = (NOW - timedelta(seconds=5)).isoformat()
    local.endpoint_file.write_text(json.dumps(endpoint))
    local.log_file.write_text(json.dumps({"updated_at": (NOW - timedelta(seconds=10)).isoformat(),
                                         "status": "HEALTHY", "outcome": "ok", "checks": {},
                                         "url": endpoint["url"], "boot_id": "local-boot",
                                         "last_inference_success_at": (NOW - timedelta(seconds=10)).isoformat()}) + "\n")
    result = manager.summary("local")
    assert result["latest_status"] == "UNKNOWN" and result["last_inference_success_at"] is None
    assert result["boot_id"] == "next-boot" and result["observed_count"] == 1


@pytest.mark.parametrize("metadata_time", [None, NOW - timedelta(seconds=60)])
def test_observed_restart_overrides_older_static_boot_metadata_without_old_deep_success(tmp_path, metadata_time):
    manager = configure(tmp_path)
    local = manager.services["local"]
    endpoint = json.loads(local.endpoint_file.read_text())
    if metadata_time is not None:
        endpoint["updated_at"] = metadata_time.isoformat()
    local.endpoint_file.write_text(json.dumps(endpoint))
    rows = [{"updated_at": (NOW - timedelta(seconds=20)).isoformat(), "status": "HEALTHY", "outcome": "ok",
             "url": endpoint["url"], "boot_id": "local-boot", "checks": {},
             "last_inference_success_at": (NOW - timedelta(seconds=20)).isoformat()},
            {"updated_at": (NOW - timedelta(seconds=1)).isoformat(), "status": "RECOVERING", "outcome": "ok",
             "url": endpoint["url"], "boot_id": "next-boot", "checks": {},
             "consecutive_successes": 1, "recovery_threshold": 2}]
    local.log_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = manager.summary("local")
    assert result["latest_status"] == "RECOVERING" and result["boot_id"] == "next-boot"
    assert not result["stale"] and result["last_inference_success_at"] is None
    assert result["observed_count"] == 2 and result["success_count"] == 2


def test_fresh_state_for_new_boot_needs_its_own_history_before_showing_healthy(tmp_path):
    manager = configure(tmp_path)
    local = manager.services["local"]
    old = {"updated_at": (NOW - timedelta(seconds=20)).isoformat(), "status": "HEALTHY", "outcome": "ok",
           "url": "http://192.168.31.204:8770", "boot_id": "local-boot", "checks": {},
           "last_inference_success_at": (NOW - timedelta(seconds=20)).isoformat()}
    local.log_file.write_text(json.dumps(old) + "\n")
    local.state_file.write_text(json.dumps({**old, "updated_at": (NOW - timedelta(seconds=1)).isoformat(),
                                           "boot_id": "next-boot", "last_inference_success_at": None}))
    result = manager.summary("local")
    assert result["latest_status"] == "UNKNOWN" and result["stale"]
    assert result["boot_id"] == "next-boot" and result["last_inference_success_at"] is None
    assert result["observed_count"] == 1
