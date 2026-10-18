"""Black-box HTTP and transition tests for the external Colab monitor."""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "monitor_colab.py"
spec = importlib.util.spec_from_file_location("colab_monitor", SCRIPT)
monitor_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = monitor_module
spec.loader.exec_module(monitor_module)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds=60):
        self.now += seconds


@contextmanager
def health_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            self.server.requests.append((self.command, self.path, self.headers.get("Authorization")))
            kind = self.path.rsplit("/", 1)[-1]
            code = self.server.codes.get(kind, 200)
            if kind != "live" and self.headers.get("Authorization") != "Bearer test-secret":
                code = 401
            if self.server.delay:
                time.sleep(self.server.delay)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            if code == 302:
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/sink")
            self.end_headers()
            body = {"status": "ok" if code == 200 else "loading", "boot_id": self.server.boot_id}
            if kind in {"ready", "inference"}:
                body["ready"] = code == 200
            if kind == "inference":
                body["checks"] = {"shape": [1, 768], "finite": True, "nonzero": True, "unit_norm": True}
            body.update(self.server.bodies.get(kind, {}))
            try:
                self.wfile.write(json.dumps(body).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.requests, server.codes, server.boot_id, server.delay, server.bodies = [], {}, "boot-a", 0.0, {}
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.fixture
def make_monitor(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_INFERENCE_API_KEY", "test-secret")

    def make(url, **kwargs):
        events, clock = [], Clock()
        settings = monitor_module.Settings(url=url, state_file=tmp_path / "health.json", **kwargs)
        monitor = monitor_module.Monitor(settings, clock=clock, emit=events.append)
        return monitor, clock, events

    return make


def test_real_http_failure_threshold_and_recovery(make_monitor):
    with health_server() as (server, url):
        monitor, clock, events = make_monitor(url)
        server.codes["live"] = 500
        states = []
        for _ in range(3):
            states.append(monitor.tick()["status"])
            clock.advance()
        assert states == ["DEGRADED", "DEGRADED", "DOWN"]
        assert len(events) == 2  # An unchanged failure does not generate another notification.
        server.codes.clear()
        assert monitor.tick()["status"] == "RECOVERING"
        clock.advance()
        healthy = monitor.tick()
        assert healthy["status"] == "HEALTHY"
        assert json.loads(events[-1])["event"] == "RECOVERED"
        persisted = json.loads(monitor.settings.state_file.read_text())
        assert persisted == healthy and persisted["last_inference_success_at"]
        assert sum(path.endswith("/inference") for _, path, _ in server.requests) == 1
        assert all(auth is None for _, path, auth in server.requests if path.endswith("/live"))
        assert "test-secret" not in "".join(events) + monitor.settings.state_file.read_text()


@pytest.mark.parametrize("busy_route", ["live", "ready", "inference"])
def test_busy_does_not_become_an_outage(make_monitor, busy_route):
    with health_server() as (server, url):
        monitor, clock, events = make_monitor(url)
        server.codes[busy_route] = 429
        for _ in range(7):
            state = monitor.tick()
            assert state["status"] == "BUSY"
            assert state["consecutive_failures"] == 0
            assert state["outage_seconds"] is None
            clock.advance()
        assert len(events) == 1


def test_auth_failure_stops_inference_then_key_rotation_recovers(make_monitor, monkeypatch):
    with health_server() as (server, url):
        monitor, clock, events = make_monitor(url)
        monkeypatch.setenv("OMNI_INFERENCE_API_KEY", "wrong-private-key")
        for _ in range(3):
            assert monitor.tick()["status"] == "AUTH_ERROR"
            clock.advance()
        assert not any(path.endswith("/inference") for _, path, _ in server.requests)
        monkeypatch.setenv("OMNI_INFERENCE_API_KEY", "test-secret")
        monitor.tick()
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"
        assert json.loads(events[-1])["event"] == "RECOVERED"
        assert "wrong-private-key" not in "".join(events) + monitor.settings.state_file.read_text()


def test_inference_auth_error_retries_after_cooldown_without_key_rotation(make_monitor):
    with health_server() as (server, url):
        monitor, clock, events = make_monitor(url)
        server.codes["inference"] = 403
        assert monitor.tick()["status"] == "AUTH_ERROR"
        for _ in range(4):
            clock.advance()
            assert monitor.tick()["status"] == "AUTH_ERROR"
        assert sum(path.endswith("/inference") for _, path, _ in server.requests) == 1
        assert len(events) == 1
        clock.advance()
        assert monitor.tick()["status"] == "AUTH_ERROR"
        assert sum(path.endswith("/inference") for _, path, _ in server.requests) == 2
        assert len(events) == 1
        # The same valid token recovers once a temporary gateway denial is fixed.
        server.codes["inference"] = 200
        clock.advance(300)
        assert monitor.tick()["status"] == "RECOVERING"
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"
        assert sum(path.endswith("/inference") for _, path, _ in server.requests) == 3
        assert json.loads(events[-1])["event"] == "RECOVERED"


def test_inference_key_rotation_bypasses_auth_cooldown(make_monitor, monkeypatch):
    with health_server() as (server, url):
        monitor, clock, _ = make_monitor(url)
        server.codes["inference"] = 403
        assert monitor.tick()["status"] == "AUTH_ERROR"
        clock.advance()
        server.codes["inference"] = 200
        monkeypatch.setenv("OMNI_INFERENCE_API_KEY", "replacement-key")
        assert monitor.tick()["status"] == "AUTH_ERROR"  # readiness rejects the new invalid key
        monkeypatch.setenv("OMNI_INFERENCE_API_KEY", "test-secret")
        assert monitor.tick()["status"] == "RECOVERING"
        assert sum(path.endswith("/inference") for _, path, _ in server.requests) == 2


def test_light_checks_cannot_clear_failed_model_probe(make_monitor):
    with health_server() as (server, url):
        monitor, clock, _ = make_monitor(url)
        server.codes["inference"] = 500
        assert monitor.tick()["status"] == "DEGRADED"
        server.codes["inference"] = 200
        clock.advance()
        assert monitor.tick()["status"] == "DEGRADED"
        clock.advance()
        assert monitor.tick()["status"] == "DOWN"
        clock.advance(180)
        assert monitor.tick()["status"] == "RECOVERING"
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"


def test_endpoint_reload_and_restart_require_fresh_probe(make_monitor, tmp_path):
    with health_server() as (server, url), health_server() as (new_server, new_url):
        endpoint_file = tmp_path / "endpoint.json"
        endpoint_file.write_text(json.dumps({"url": url, "boot_id": "boot-a"}))
        monitor, clock, events = make_monitor(None, endpoint_file=endpoint_file)
        monitor.tick()
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"
        new_server.boot_id = "boot-b"
        endpoint_file.write_text(json.dumps({"url": new_url, "boot_id": "boot-b"}))
        clock.advance()
        state = monitor.tick()
        assert state["url"] == new_url and state["boot_id"] == "boot-b"
        assert state["status"] == "RECOVERING" and "inference" in state["checks"]
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"
        assert sum(json.loads(event)["event"] == "RESTARTED" for event in events) == 1


def test_stale_boot_metadata_does_not_reset_recovery_every_round(make_monitor, tmp_path):
    with health_server() as (server, url):
        endpoint_file = tmp_path / "endpoint.json"
        endpoint_file.write_text(json.dumps({"url": url, "boot_id": "boot-a"}))
        monitor, clock, events = make_monitor(None, endpoint_file=endpoint_file)
        monitor.tick()
        server.boot_id = "boot-b"
        clock.advance()
        assert monitor.tick()["status"] == "RECOVERING"
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"
        assert sum(json.loads(event)["event"] == "RESTARTED" for event in events) == 1


@pytest.mark.parametrize("route,code,rounds", [("live", 500, 3), ("ready", 503, 5)])
def test_endpoint_churn_cannot_hide_outage_forever(make_monitor, tmp_path, route, code, rounds):
    with health_server() as (server, url):
        endpoint_file = tmp_path / "endpoint.json"
        monitor, clock, _ = make_monitor(None, endpoint_file=endpoint_file)
        server.codes[route] = code
        for i in range(rounds):
            endpoint_file.write_text(json.dumps({"url": f"{url}/worker-{i}", "boot_id": f"boot-{i}"}))
            state = monitor.tick()
            if i == 0 and code == 503:
                assert state["status"] == "STARTING"
            clock.advance()
        assert state["status"] == "DOWN"
        assert state["outage_seconds"] == (rounds - 1) * 60


def test_redirect_is_not_followed_with_bearer_token(make_monitor):
    with health_server() as (server, url):
        server.codes["ready"] = 302
        monitor, _, _ = make_monitor(url)
        state = monitor.tick()
        assert state["status"] == "DEGRADED"
        assert state["reason"] == "ready_redirect_rejected"
        assert [path for _, path, _ in server.requests] == ["/health/live", "/health/ready"]


def test_http_timeout_is_bounded(make_monitor):
    with health_server() as (server, url):
        server.delay = 0.3
        monitor, _, _ = make_monitor(url, timeout=0.03)
        started = time.monotonic()
        state = monitor.tick()
        assert time.monotonic() - started < 0.25
        assert state["outcome"] == "failure" and state["checks"]["live"]["http_status"] == 0


@pytest.mark.parametrize("url", ["http://example.com", "https://host/path?token=SECRET", "https://user:SECRET@host", "ftp://host", "http://host", "https://host/#SECRET", "https://host:0"])
def test_unsafe_urls_are_rejected_without_disclosing_input(url):
    with pytest.raises(monitor_module.ConfigurationError) as error:
        monitor_module.validate_url(url)
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("url", ["http://192.168.31.204:8770", "http://10.0.0.1:8770", "http://172.16.1.2:8770"])
def test_private_http_is_rejected_by_default(url):
    with pytest.raises(monitor_module.ConfigurationError):
        monitor_module.validate_url(url)


def test_http_permission_is_exact_host_and_port_and_applies_to_endpoint_file(tmp_path):
    allowed = ("192.168.31.204:8770",)
    url = "http://192.168.31.204:8770"
    assert monitor_module.validate_url(url, allowed) == url
    endpoint_file = tmp_path / "endpoint.json"
    endpoint_file.write_text(json.dumps({"url": url, "boot_id": "lan-a"}))
    assert monitor_module.load_endpoint(None, endpoint_file, allowed).url == url
    with pytest.raises(monitor_module.ConfigurationError):
        monitor_module.load_endpoint(None, endpoint_file)
    for denied in ["http://192.168.31.205:8770", "http://192.168.31.204:8771",
                   "http://192.168.31.204", "http://192.168.31.204.evil:8770",
                   f"{url}?secret=value", f"{url}#secret", "http://user:secret@192.168.31.204:8770"]:
        with pytest.raises(monitor_module.ConfigurationError):
            monitor_module.validate_url(denied, allowed)


def test_cancelled_monitor_does_not_read_key_or_write_observation(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("disabled monitor must not read its endpoint or credentials")

    monkeypatch.setattr(monitor_module, "load_key", forbidden)
    monkeypatch.setattr(monitor_module, "load_endpoint", forbidden)
    monitor = monitor_module.Monitor(monitor_module.Settings(state_file=tmp_path / "health.json"),
                                     cancelled=lambda: True, emit=forbidden)
    assert monitor.tick() == {}
    assert not monitor.settings.state_file.exists()
    assert not monitor.log_file.exists()


@pytest.mark.parametrize("cancel_after", ["live", "ready", "inference"])
def test_cancel_during_request_stops_remaining_requests_and_leaves_artifacts_unchanged(tmp_path, monkeypatch, cancel_after):
    monkeypatch.setenv("OMNI_INFERENCE_API_KEY", "test-secret")
    stopped = threading.Event()
    state_file, log_file = tmp_path / "health.json", tmp_path / "probe-log.jsonl"
    prior = {"status": "HEALTHY", "updated_at": "2026-10-17T00:00:00Z"}
    state_file.write_text(json.dumps(prior))
    log_file.write_text(json.dumps(prior) + "\n")
    state_before, log_before = state_file.read_bytes(), log_file.read_bytes()

    class CancellingHTTP:
        def __init__(self):
            self.requests = []

        def request(self, url, method, key=None):
            kind = url.rsplit("/", 1)[-1]
            self.requests.append(kind)
            if kind == cancel_after:
                stopped.set()
            body = {"status": "ok", "ready": True, "checks": {
                "shape": [1, 768], "finite": True, "nonzero": True, "unit_norm": True}}
            return monitor_module.HTTPResult(200, body, 1.0)

    http, events = CancellingHTTP(), []
    monitor = monitor_module.Monitor(
        monitor_module.Settings(url="http://127.0.0.1:8000", state_file=state_file, log_file=log_file),
        http=http, cancelled=stopped.is_set, emit=events.append)
    monitor.last_state = prior
    assert monitor.tick() == prior
    assert http.requests == ["live", "ready", "inference"][:["live", "ready", "inference"].index(cancel_after) + 1]
    assert state_file.read_bytes() == state_before
    assert log_file.read_bytes() == log_before
    assert events == []
    # Once disabled, neither another light round nor a new deep check begins.
    assert monitor.tick() == prior
    assert len(http.requests) == ["live", "ready", "inference"].index(cancel_after) + 1


def test_http_transport_honors_cancellation_and_separate_probe_timeout():
    stopped, calls = threading.Event(), []
    http = monitor_module.HealthHTTP(10, probe_timeout=45, cancelled=stopped.is_set)

    class Opener:
        def open(self, request, *, timeout):
            calls.append((request.full_url, timeout))
            response = io.BytesIO(b'{"status":"ok"}')
            response.code = 200
            return response

    http.opener = Opener()
    http.request("http://127.0.0.1/health/ready", "GET", "test-secret")
    http.request("http://127.0.0.1/health/inference", "POST", "test-secret")
    assert [timeout for _, timeout in calls] == [10, 45]
    stopped.set()
    with pytest.raises(monitor_module.CheckCancelled):
        http.request("http://127.0.0.1/health/live", "GET")
    assert len(calls) == 2


def test_private_key_file_permissions_and_rotation(tmp_path, monkeypatch):
    key_file = tmp_path / "key"
    key_file.write_text("private-key")
    key_file.chmod(0o644)
    with pytest.raises(monitor_module.ConfigurationError, match="chmod 600"):
        monitor_module.load_key(key_file)
    key_file.chmod(0o600)
    assert monitor_module.load_key(key_file) == "private-key"
    key_file.write_text("replacement-key")
    assert monitor_module.load_key(key_file) == "replacement-key"


def test_once_cli_real_service_and_private_state_file(tmp_path):
    with health_server() as (server, url):
        state_file = tmp_path / "state.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--url", url, "--state-file", str(state_file), "--once"],
            env={**os.environ, "OMNI_INFERENCE_API_KEY": "test-secret"},
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0, result.stderr
        state = json.loads(state_file.read_text())
        assert state["outcome"] == "ok" and state["checks"]["inference"]["http_status"] == 200
        assert state_file.stat().st_mode & 0o077 == 0
        assert "test-secret" not in result.stdout + result.stderr + state_file.read_text()
        assert len(server.requests) == 3


def test_finite_session_and_bad_numeric_configuration(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--url", "http://127.0.0.1:1", "--state-file", str(tmp_path / "state.json"),
         "--max-runtime", "0.05", "--interval", "0.01", "--timeout", "0.01"],
        env={**os.environ, "OMNI_INFERENCE_API_KEY": "test-secret"}, capture_output=True, text=True, timeout=2,
    )
    assert result.returncode == 1
    for flag, value in [("--interval", "0"), ("--timeout", "nan"), ("--failure-threshold", "0")]:
        with pytest.raises(SystemExit):
            monitor_module.parse_args(["--url", "https://example.com", flag, value])


@pytest.mark.parametrize("ready", [None, False, "true", 1])
def test_http_200_requires_actual_readiness(make_monitor, ready):
    with health_server() as (server, url):
        server.bodies["ready"] = {"ready": ready}
        monitor, _, _ = make_monitor(url)
        state = monitor.tick()
        assert state["outcome"] == "failure"
        assert state["reason"] == "ready_invalid_readiness"
        assert not any(path.endswith("/inference") for _, path, _ in server.requests)


@pytest.mark.parametrize("checks", [
    None, {}, {"shape": [1, 256], "finite": True, "nonzero": True, "unit_norm": True},
    {"shape": [1, 768], "finite": False, "nonzero": True, "unit_norm": True},
    {"shape": [1, 768], "finite": True, "nonzero": False, "unit_norm": True},
    {"shape": [1, 768], "finite": True, "nonzero": True, "unit_norm": "true"},
])
def test_http_200_requires_valid_model_self_test(make_monitor, checks):
    with health_server() as (server, url):
        server.bodies["inference"] = {"checks": checks}
        monitor, _, _ = make_monitor(url)
        state = monitor.tick()
        assert state["outcome"] == "failure"
        assert state["reason"] == "inference_invalid_self_test"


def project_probe_body():
    return {"status": "ok", "ok": True, "dimensions": 768, "task": "query",
            "checks": {"shape": True, "finite": True, "nonzero": True, "unit_norm": True}}


@pytest.mark.parametrize("protocol", ["colab", "project"])
def test_both_inference_success_contracts_are_accepted(protocol):
    body = project_probe_body()
    if protocol == "colab":
        body = {"status": "ok", "ready": True,
                "checks": {**body["checks"], "shape": [1, 768]}}
    result = monitor_module.HTTPResult(200, body, 1.0)
    assert monitor_module.Monitor._result(result, "inference") == ("ok", "checks_passed")


@pytest.mark.parametrize("protocol", ["colab", "project"])
@pytest.mark.parametrize("flag", ["ready", "ok"])
@pytest.mark.parametrize("value", [None, False, "true", 1])
def test_inference_contracts_cannot_bypass_invalid_success_flags(protocol, flag, value):
    body = project_probe_body()
    if protocol == "colab":
        body = {"status": "ok", "ready": True,
                "checks": {**body["checks"], "shape": [1, 768]}}
    body[flag] = value
    result = monitor_module.HTTPResult(200, body, 1.0)
    assert monitor_module.Monitor._result(result, "inference") == ("failure", "inference_invalid_self_test")


@pytest.mark.parametrize("field", ["ok", "dimensions", "task", "checks"])
def test_project_probe_requires_complete_contract(field):
    body = project_probe_body()
    body.pop(field)
    result = monitor_module.HTTPResult(200, body, 1.0)
    assert monitor_module.Monitor._result(result, "inference")[0] == "failure"


@pytest.mark.parametrize("field,value", [
    ("dimensions", None), ("dimensions", 128), ("dimensions", 768.0), ("dimensions", "768"),
    ("dimensions", True), ("task", None), ("task", "document"), ("task", "Query"),
])
def test_project_probe_requires_exact_dimension_and_query_task(field, value):
    body = project_probe_body()
    body[field] = value
    result = monitor_module.HTTPResult(200, body, 1.0)
    assert monitor_module.Monitor._result(result, "inference")[0] == "failure"


@pytest.mark.parametrize("field", ["shape", "finite", "nonzero", "unit_norm"])
@pytest.mark.parametrize("value", [None, False, "true", 1])
def test_project_probe_rejects_nonboolean_or_failed_checks(field, value):
    body = project_probe_body()
    body["checks"][field] = value
    result = monitor_module.HTTPResult(200, body, 1.0)
    assert monitor_module.Monitor._result(result, "inference")[0] == "failure"


@pytest.mark.parametrize("field", ["shape", "finite", "nonzero", "unit_norm"])
def test_project_probe_rejects_missing_checks(field):
    body = project_probe_body()
    body["checks"].pop(field)
    result = monitor_module.HTTPResult(200, body, 1.0)
    assert monitor_module.Monitor._result(result, "inference")[0] == "failure"


@pytest.mark.parametrize("shape", [[1, 768], [True, 768], [1, 768.0], False, "[1, 768]"])
def test_shape_value_cannot_switch_to_an_incomplete_protocol(shape):
    body = project_probe_body()
    body["checks"]["shape"] = shape
    result = monitor_module.HTTPResult(200, body, 1.0)
    assert monitor_module.Monitor._result(result, "inference")[0] == "failure"


@pytest.mark.parametrize("shape", [[True, 768], [1, 768.0], [1, 768, 1], True, False, "[1, 768]"])
def test_legacy_shape_requires_exact_integer_dimensions(shape):
    body = {"status": "ok", "ready": True,
            "checks": {**project_probe_body()["checks"], "shape": shape}}
    result = monitor_module.HTTPResult(200, body, 1.0)
    assert monitor_module.Monitor._result(result, "inference")[0] == "failure"


def test_project_503_busy_does_not_become_an_outage_and_recovers(make_monitor):
    with health_server() as (server, url):
        server.codes["inference"] = 503
        server.bodies["inference"] = {"status": "busy", "ok": False, "error": "InferenceBusy"}
        monitor, clock, _ = make_monitor(url)
        for _ in range(8):
            state = monitor.tick()
            assert state["status"] == "BUSY" and state["consecutive_failures"] == 0
            assert state["outage_seconds"] is None
            clock.advance()
        assert sum(path.endswith("/inference") for _, path, _ in server.requests) == 2
        server.codes.clear()
        server.bodies.clear()
        clock.advance(300)
        assert monitor.tick()["status"] == "HEALTHY"
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"


@pytest.mark.parametrize("kind,status,error,http_error", [
    ("inference", "stalled", "InferenceBusy", None),
    ("inference", "busy", "InferenceTimeout", None),
    ("inference", "busy", None, None),
    ("inference", "loading", "InferenceBusy", None),
    ("inference", "busy", "InferenceBusy", "invalid_json"),
    ("ready", "busy", "InferenceBusy", None),
])
def test_only_explicit_inference_busy_is_accepted(kind, status, error, http_error):
    response = monitor_module.HTTPResult(503, {"status": status, "error": error}, 1.0, http_error)
    assert monitor_module.Monitor._result(response, kind)[0] == "failure"


def test_stalled_is_failure_instead_of_startup_grace(make_monitor):
    with health_server() as (server, url):
        server.codes["ready"] = 503
        server.bodies["ready"] = {"status": "stalled", "ready": False}
        monitor, _, _ = make_monitor(url)
        state = monitor.tick()
        assert state["status"] == "DEGRADED" and state["outcome"] == "failure"
        assert state["reason"] == "ready_inference_stalled"
        assert not any(path.endswith("/inference") for _, path, _ in server.requests)


class APIHTTP:
    """Drive the real inference app through the monitor's network interface."""

    def __init__(self, client):
        self.client = client
        self.requests = []

    def request(self, url, method, key=None):
        from urllib.parse import urlsplit

        path = urlsplit(url).path
        headers = {} if key is None else {"Authorization": f"Bearer {key}"}
        self.requests.append((method, path, key))
        response = self.client.request(method, path, headers=headers)
        return monitor_module.HTTPResult(response.status_code, response.json(), 1.0)


@pytest.fixture
def project_health_client(monkeypatch):
    from conftest import FakeEncoder
    from fastapi.testclient import TestClient
    from omniembed import server
    from omniembed.health import HEALTH
    from omniembed.health_api import warmup

    fake = FakeEncoder()
    monkeypatch.setattr(server.PIPE, "enc", fake)
    monkeypatch.setattr(server.INDEX, "enc", fake)
    monkeypatch.setattr(server.CONFIG, "api_key", "test-secret")
    monkeypatch.setenv("OMNI_SKIP_WARMUP", "1")
    # This integration exercises the monitored API, never the observer's
    # unrelated background target or a real model.
    monkeypatch.setattr(server.SERVICE_HEALTH, "start", lambda: None)
    monkeypatch.setattr(server.SERVICE_HEALTH, "stop", lambda: None)
    HEALTH.reset()
    try:
        with TestClient(server.app) as client:
            assert warmup(lambda: fake, retry_s=0.01)
            yield client, fake, server
    finally:
        HEALTH.reset()


def test_real_project_health_contract_and_vector_failure_recovery(make_monitor, project_health_client):
    import numpy as np

    client, fake, _ = project_health_client
    monitor, clock, _ = make_monitor("http://127.0.0.1:8000")
    monitor.http = APIHTTP(client)
    first = monitor.tick()
    assert first["outcome"] == "ok" and first["checks"]["inference"]["http_status"] == 200
    assert fake.last_prefix == "task: search result | query: "
    clock.advance()
    assert monitor.tick()["status"] == "HEALTHY"

    fake.bad_vector = np.zeros((1, 768), dtype=np.float32)
    clock.advance(300)
    failed = monitor.tick()
    assert failed["outcome"] == "failure" and failed["checks"]["inference"]["http_status"] == 503
    fake.bad_vector = None
    clock.advance(300)
    recovered = monitor.tick()
    assert recovered["status"] == "RECOVERING" and recovered["boot_id"] == first["boot_id"]
    clock.advance()
    assert monitor.tick()["status"] == "HEALTHY"


def test_real_project_probe_busy_is_not_an_outage(make_monitor, project_health_client, monkeypatch):
    from omniembed.health import HEALTH

    client, _, server = project_health_client
    monkeypatch.setattr(server.CONFIG, "probe_wait_s", 0.01)
    monitor, clock, _ = make_monitor("http://127.0.0.1:8000")
    monitor.http = APIHTTP(client)
    HEALTH.lock.acquire()
    try:
        state = monitor.tick()
        assert state["status"] == "BUSY" and state["consecutive_failures"] == 0
        assert state["checks"]["inference"]["http_status"] == 503
    finally:
        HEALTH.lock.release()
    clock.advance(300)
    assert monitor.tick()["status"] == "HEALTHY"


class RecoverableEncoder:
    model_id, revision, device, dtype = "fake-gemma", "test-sha", "cpu", "float32"

    def __init__(self):
        self.fail = False
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def query(self, texts, bs=1):
        import numpy as np

        if self.block:
            self.entered.set()
            assert self.release.wait(5)
        if self.fail:
            raise RuntimeError("transient model error")
        return np.full((len(texts), 768), 768 ** -0.5, dtype=np.float32)


def _wait_api_ready(client):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if client.get("/health/ready", headers={"Authorization": "Bearer test-secret"}).status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError("fake API did not finish startup")


def test_real_api_transient_failure_recovers_without_restart(make_monitor):
    from fastapi.testclient import TestClient
    from omniembed.inference import create_app

    encoder = RecoverableEncoder()
    app = create_app(encoder, api_key="test-secret")
    with TestClient(app) as client:
        _wait_api_ready(client)
        monitor, clock, _ = make_monitor("http://127.0.0.1:8000")
        monitor.http = APIHTTP(client)
        original_boot = monitor.tick()["boot_id"]
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"
        encoder.fail = True
        response = client.post("/v1/embeddings", json={"input": "hello"},
                               headers={"Authorization": "Bearer test-secret"})
        assert response.status_code == 503
        clock.advance()
        state = monitor.tick()
        assert state["outcome"] == "failure" and state["reason"] == "model_degraded_pending_probe"
        encoder.fail = False
        clock.advance(300)
        state = monitor.tick()
        assert state["status"] == "RECOVERING" and state["checks"]["inference"]["http_status"] == 200
        assert state["boot_id"] == original_boot
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"
        assert all(key == "test-secret" for _, path, key in monitor.http.requests if path != "/health/live")


def test_real_api_failed_probe_retries_while_readiness_is_degraded(make_monitor):
    from fastapi.testclient import TestClient
    from omniembed.inference import create_app

    encoder = RecoverableEncoder()
    with TestClient(create_app(encoder, api_key="test-secret")) as client:
        _wait_api_ready(client)
        monitor, clock, _ = make_monitor("http://127.0.0.1:8000")
        monitor.http = APIHTTP(client)
        encoder.fail = True
        first = monitor.tick()
        assert first["outcome"] == "failure" and first["checks"]["inference"]["http_status"] == 503
        assert client.get("/health/ready", headers={"Authorization": "Bearer test-secret"}).status_code == 503
        encoder.fail = False
        clock.advance(300)
        recovered = monitor.tick()
        assert recovered["status"] == "RECOVERING"
        assert recovered["checks"]["inference"]["http_status"] == 200
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"


def test_real_api_loading_and_stalled_never_trigger_recovery_inference(make_monitor):
    from fastapi.testclient import TestClient
    from omniembed.inference import create_app

    encoder = RecoverableEncoder()
    encoder.block = True
    app = create_app(encoder, api_key="test-secret")
    try:
        with TestClient(app) as client:
            assert encoder.entered.wait(2)
            monitor, clock, _ = make_monitor("http://127.0.0.1:8000")
            monitor.http = APIHTTP(client)
            assert monitor.tick()["status"] == "STARTING"
            with app.state.inference.state_lock:
                app.state.inference.busy_since = time.monotonic() - 901
            clock.advance()
            stalled = monitor.tick()
            assert stalled["outcome"] == "failure" and stalled["reason"] == "ready_inference_stalled"
            assert not any(path == "/health/inference" for _, path, _ in monitor.http.requests)
            encoder.release.set()
            _wait_api_ready(client)
            clock.advance()
            assert monitor.tick()["status"] == "RECOVERING"
    finally:
        encoder.release.set()


def test_real_api_busy_is_reported_without_failure(make_monitor):
    from fastapi.testclient import TestClient
    from omniembed.inference import create_app

    encoder = RecoverableEncoder()
    with TestClient(create_app(encoder, api_key="test-secret")) as client:
        _wait_api_ready(client)
        monitor, clock, _ = make_monitor("http://127.0.0.1:8000")
        monitor.http = APIHTTP(client)
        encoder.block = True
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(client.post, "/v1/embeddings", json={"input": "busy"},
                                  headers={"Authorization": "Bearer test-secret"})
            try:
                assert encoder.entered.wait(2)
                state = monitor.tick()
                assert state["status"] == "BUSY" and state["consecutive_failures"] == 0
                assert state["checks"]["inference"]["http_status"] == 429
            finally:
                encoder.release.set()
            assert request.result(timeout=2).status_code == 200
        clock.advance(300)
        assert monitor.tick()["status"] == "HEALTHY"


def test_real_api_new_boot_requires_fresh_probe(make_monitor):
    from fastapi.testclient import TestClient
    from omniembed.inference import create_app

    with TestClient(create_app(RecoverableEncoder(), api_key="test-secret")) as first:
        _wait_api_ready(first)
        monitor, clock, events = make_monitor("http://127.0.0.1:8000")
        monitor.http = APIHTTP(first)
        previous_boot = monitor.tick()["boot_id"]
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"
    with TestClient(create_app(RecoverableEncoder(), api_key="test-secret")) as second:
        _wait_api_ready(second)
        monitor.http = APIHTTP(second)
        clock.advance()
        state = monitor.tick()
        assert state["status"] == "RECOVERING"
        assert state["boot_id"] != previous_boot
        assert state["checks"]["inference"]["http_status"] == 200
        assert sum(json.loads(event)["event"] == "RESTARTED" for event in events) == 1


def test_every_round_is_logged_and_survives_monitor_restart(make_monitor):
    with health_server() as (server, url):
        monitor, clock, events = make_monitor(url)
        states = []
        for _ in range(4):
            states.append(monitor.tick())
            clock.advance()
        assert len(events) == 1  # Initial success is HEALTHY, not a recovered outage.
        log_path = monitor.log_file
        assert [json.loads(line) for line in log_path.read_text().splitlines()] == states
        assert log_path.stat().st_mode & 0o077 == 0
        restarted, _, _ = make_monitor(url)
        states.append(restarted.tick())
        assert [json.loads(line) for line in log_path.read_text().splitlines()] == states


def test_initial_complete_success_is_healthy_without_a_recovery_event(make_monitor):
    with health_server() as (server, url):
        monitor, _, events = make_monitor(url)
        state = monitor.tick()
        assert state["status"] == "HEALTHY"
        assert state["consecutive_successes"] == 1 and not state["recovery_required"]
        assert [path for _, path, _ in server.requests] == ["/health/live", "/health/ready", "/health/inference"]
        assert [json.loads(event)["event"] for event in events] == ["HEALTHY"]


def test_healthy_restart_validates_light_checks_and_keeps_real_probe_schedule(make_monitor):
    with health_server() as (server, url):
        monitor, _, _ = make_monitor(url)
        saved = monitor.tick()
        latest = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        saved.update(updated_at=(latest - dt.timedelta(seconds=60)).isoformat(),
                     last_success_at=(latest - dt.timedelta(seconds=60)).isoformat(),
                     last_inference_success_at=(latest - dt.timedelta(seconds=250)).isoformat())
        monitor.settings.state_file.write_text(json.dumps(saved))
        server.requests.clear()
        restarted, clock, events = make_monitor(url)
        state = restarted.tick()
        assert state["status"] == "HEALTHY" and state["consecutive_successes"] == 2
        assert state["last_inference_success_at"] == saved["last_inference_success_at"]
        assert [path for _, path, _ in server.requests] == ["/health/live", "/health/ready"]
        assert not events
        clock.advance(49)
        assert restarted.tick()["status"] == "HEALTHY"
        assert not any(path.endswith("/inference") for _, path, _ in server.requests)
        clock.advance(1)
        assert restarted.tick()["checks"]["inference"]["http_status"] == 200


def test_failure_streak_and_partial_recovery_survive_monitor_restarts(make_monitor):
    with health_server() as (server, url):
        monitor, clock, _ = make_monitor(url)
        server.codes["live"] = 500
        assert monitor.tick()["consecutive_failures"] == 1
        clock.advance()
        assert monitor.tick()["consecutive_failures"] == 2
        restarted, _, _ = make_monitor(url)
        down = restarted.tick()
        assert down["status"] == "DOWN" and down["consecutive_failures"] == 3
        server.codes.clear()
        first_success = restarted.tick()
        assert first_success["status"] == "RECOVERING" and first_success["consecutive_successes"] == 1
        assert first_success["recovery_required"]
        server.requests.clear()
        restarted_again, _, events = make_monitor(url)
        healthy = restarted_again.tick()
        assert healthy["status"] == "HEALTHY" and healthy["consecutive_successes"] == 2
        assert not healthy["recovery_required"]
        assert [path for _, path, _ in server.requests] == ["/health/live", "/health/ready"]
        assert [json.loads(event)["event"] for event in events] == ["RECOVERED"]


def test_saved_down_stays_down_when_restarted_observer_cannot_connect(make_monitor):
    with health_server() as (server, url):
        monitor, clock, _ = make_monitor(url)
        server.codes["live"] = 500
        for _ in range(3):
            down = monitor.tick()
            clock.advance()
        assert down["status"] == "DOWN"

        class OfflineHTTP:
            def request(self, url, method, key=None):
                return monitor_module.HTTPResult(0, {}, 1.0, "connection_failed")

        restarted, clock, events = make_monitor(url)
        restarted.http = OfflineHTTP()
        still_down = restarted.tick()
        assert still_down["status"] == "DOWN" and still_down["consecutive_failures"] == 4
        assert still_down["last_inference_success_at"] is None
        assert still_down["inference_probe_status"] == "NOT_CHECKED"
        assert not events  # Re-observing the same outage is not a new transition.
        server.codes.clear()
        server.requests.clear()
        restarted.http = monitor_module.HealthHTTP(10, probe_timeout=45)
        clock.advance()
        assert restarted.tick()["status"] == "RECOVERING"
        assert [path for _, path, _ in server.requests] == ["/health/live", "/health/ready", "/health/inference"]


def test_unverified_boot_cannot_restore_healthy_or_cached_probe(make_monitor):
    with health_server() as (_, url):
        monitor, _, _ = make_monitor(url)
        assert monitor.tick()["status"] == "HEALTHY"

        class OfflineHTTP:
            def request(self, url, method, key=None):
                return monitor_module.HTTPResult(0, {}, 1.0, "connection_failed")

        restarted, _, _ = make_monitor(url)
        restarted.http = OfflineHTTP()
        failed = restarted.tick()
        assert failed["status"] == "DEGRADED" and failed["consecutive_failures"] == 1
        assert failed["consecutive_successes"] == 0
        assert failed["last_inference_success_at"] is None
        assert failed["inference_probe_status"] == "NOT_CHECKED"


def test_failed_live_with_matching_boot_does_not_reuse_saved_success(make_monitor):
    with health_server() as (server, url):
        monitor, _, _ = make_monitor(url)
        assert monitor.tick()["status"] == "HEALTHY"
        server.codes["live"] = 500
        restarted, _, _ = make_monitor(url)
        failed = restarted.tick()
        assert failed["status"] == "DEGRADED" and failed["boot_id"] == server.boot_id
        assert failed["last_inference_success_at"] is None
        assert failed["inference_probe_status"] == "NOT_CHECKED"


@pytest.mark.parametrize("change", [
    {"url": "https://another-service.example"}, {"boot_id": None}, {"boot_id": "invalid boot"},
    {"consecutive_successes": True}, {"consecutive_failures": -1}, {"recovery_threshold": 0},
    {"schema_version": True}, {"status": "invented"}, {"recovery_required": "true"},
    {"status": "BUSY", "outcome": "busy", "consecutive_successes": 0, "inference_probe_status": ["AUTH"]},
    {"updated_at": "stale"}, {"updated_at": "future"},
    {"updated_at": "0001-01-01T00:00:00+23:00"},
])
def test_invalid_stale_or_other_target_saved_state_cannot_skip_real_probe(make_monitor, change):
    with health_server() as (server, url):
        monitor, _, _ = make_monitor(url)
        saved = monitor.tick()
        change = dict(change)
        if change.get("updated_at") in {"stale", "future"}:
            delta = -181 if change["updated_at"] == "stale" else 60
            change["updated_at"] = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=delta)).isoformat()
        saved.update(change)
        monitor.settings.state_file.write_text(json.dumps(saved))
        server.requests.clear()
        restarted, _, _ = make_monitor(url)
        assert restarted.tick()["status"] == "HEALTHY"
        assert [path for _, path, _ in server.requests] == ["/health/live", "/health/ready", "/health/inference"]


@pytest.mark.parametrize("body", ["not json", "[" * 2000 + "]" * 2000, "x" * (64 * 1024 + 1)])
def test_corrupt_or_oversized_saved_state_does_not_crash_monitor(make_monitor, body):
    with health_server() as (server, url):
        monitor, _, _ = make_monitor(url)
        monitor.settings.state_file.write_text(body)
        assert monitor.tick()["status"] == "HEALTHY"
        assert len(server.requests) == 3


def test_fifo_saved_state_is_ignored_without_waiting_for_a_writer(make_monitor):
    if not hasattr(os, "mkfifo"):
        pytest.skip("requires FIFO support")
    with health_server() as (server, url):
        monitor, _, _ = make_monitor(url)
        os.mkfifo(monitor.settings.state_file)
        started = time.monotonic()
        assert monitor.tick()["status"] == "HEALTHY"
        assert time.monotonic() - started < 1
        assert len(server.requests) == 3


def test_saved_healthy_boot_change_still_forces_real_probe_and_recovery(make_monitor):
    with health_server() as (server, url):
        monitor, _, _ = make_monitor(url)
        old = monitor.tick()
        server.boot_id = "boot-b"
        server.requests.clear()
        restarted, clock, events = make_monitor(url)
        current = restarted.tick()
        assert current["boot_id"] != old["boot_id"] and current["status"] == "RECOVERING"
        assert current["consecutive_successes"] == 1
        assert current["checks"]["inference"]["http_status"] == 200
        assert [json.loads(event)["event"] for event in events] == ["RESTARTED", "RECOVERING"]
        clock.advance()
        assert restarted.tick()["status"] == "HEALTHY"


def test_static_old_boot_metadata_does_not_repeatedly_restart_a_healthy_monitor(make_monitor, tmp_path):
    with health_server() as (server, url):
        endpoint_file = tmp_path / "endpoint.json"
        endpoint_file.write_text(json.dumps({"url": url, "boot_id": "old-static-boot"}))
        monitor, clock, _ = make_monitor(None, endpoint_file=endpoint_file)
        assert monitor.tick()["status"] == "RECOVERING"
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"
        for _ in range(2):
            server.requests.clear()
            restarted, _, events = make_monitor(None, endpoint_file=endpoint_file)
            assert restarted.tick()["status"] == "HEALTHY"
            assert [path for _, path, _ in server.requests] == ["/health/live", "/health/ready"]
            assert not events


def test_saved_health_never_bypasses_current_readiness(make_monitor):
    with health_server() as (server, url):
        monitor, _, _ = make_monitor(url)
        assert monitor.tick()["status"] == "HEALTHY"
        server.codes["ready"] = 503
        server.requests.clear()
        restarted, _, _ = make_monitor(url)
        assert restarted.tick()["status"] == "STARTING"
        assert [path for _, path, _ in server.requests] == ["/health/live", "/health/ready"]


def test_failed_deep_probe_is_retested_after_restart_without_light_only_recovery(make_monitor):
    with health_server() as (server, url):
        monitor, _, _ = make_monitor(url)
        server.codes["inference"] = 500
        assert monitor.tick()["status"] == "DEGRADED"
        server.codes.clear()
        server.requests.clear()
        restarted, _, _ = make_monitor(url)
        recovered = restarted.tick()
        assert recovered["status"] == "RECOVERING" and recovered["consecutive_successes"] == 1
        assert [path for _, path, _ in server.requests] == ["/health/live", "/health/ready", "/health/inference"]


def test_legacy_busy_state_retains_a_known_deep_failure_recovery_requirement(make_monitor):
    with health_server() as (server, url):
        monitor, _, _ = make_monitor(url)
        server.codes["inference"] = 500
        saved = monitor.tick()
        saved.update(status="BUSY", outcome="busy", consecutive_failures=0)
        saved.pop("recovery_required")
        monitor.settings.state_file.write_text(json.dumps(saved))
        server.codes.clear()
        restarted, _, _ = make_monitor(url)
        recovered = restarted.tick()
        assert recovered["status"] == "RECOVERING" and recovered["consecutive_successes"] == 1
        assert recovered["checks"]["inference"]["http_status"] == 200


def test_round_log_does_not_copy_auth_prompt_or_raw_response(make_monitor):
    with health_server() as (server, url):
        server.bodies["ready"] = {"prompt": "PRIVATE-PROMPT", "body": "RAW-BODY", "token": "test-secret"}
        server.bodies["inference"] = {"authorization": "Bearer test-secret", "prompt": "PRIVATE-PROMPT"}
        monitor, _, _ = make_monitor(url)
        monitor.tick()
        content = monitor.log_file.read_text()
        assert all(secret not in content for secret in ("test-secret", "PRIVATE-PROMPT", "RAW-BODY", "Bearer"))
        assert json.loads(content)["checks"]["inference"]["http_status"] == 200


def test_log_retention_drops_old_malformed_and_partial_rows(tmp_path):
    now = dt.datetime.now(dt.timezone.utc)
    recent = {"schema_version": 1, "updated_at": (now - dt.timedelta(hours=3)).isoformat(),
              "status": "DOWN", "outcome": "failure", "checks": {}}
    old = {**recent, "updated_at": (now - dt.timedelta(hours=25)).isoformat()}
    future = {**recent, "updated_at": (now + dt.timedelta(days=1)).isoformat()}
    invalid_date = {**recent, "updated_at": "definitely not a date"}
    new = {**recent, "updated_at": now.isoformat(), "status": "HEALTHY", "outcome": "ok"}
    log_path = tmp_path / "probe.jsonl"
    lines = [json.dumps(row) for row in (old, recent, future, invalid_date)]
    log_path.write_text("\n".join(lines) + '\nnot-json\n[]\n{"partial":')
    log_path.chmod(0o644)
    monitor_module.append_probe_log(log_path, new, prune=True)
    assert [json.loads(line) for line in log_path.read_text().splitlines()] == [recent, new]
    assert log_path.stat().st_mode & 0o077 == 0


def test_log_size_cap_retains_recent_tail_and_never_grows_unbounded(tmp_path):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    recent = {"schema_version": 1, "updated_at": now, "status": "DOWN", "outcome": "failure", "checks": {}}
    new = {**recent, "status": "HEALTHY", "outcome": "ok"}
    log_path = tmp_path / "probe.jsonl"
    with log_path.open("wb") as handle:
        handle.write(b"x" * (monitor_module.MAX_LOG_BYTES + 1024))
        handle.write(b"\n" + json.dumps(recent).encode() + b"\n")
    monitor_module.append_probe_log(log_path, new)
    assert log_path.stat().st_size <= monitor_module.MAX_LOG_BYTES
    assert [json.loads(line) for line in log_path.read_text().splitlines()] == [recent, new]
    assert not list(tmp_path.glob(".probe.jsonl.*"))


def test_periodic_pruning_happens_without_rewriting_every_round(make_monitor, monkeypatch):
    with health_server() as (_, url):
        monitor, _, _ = make_monitor(url)
        compact = monitor_module._compact_probe_log
        compactions = []

        def observe_compact(path, line, now):
            compactions.append(path)
            return compact(path, line, now)

        monkeypatch.setattr(monitor_module, "_compact_probe_log", observe_compact)
        for _ in range(61):
            monitor.tick()
        assert len(compactions) == 2
        assert len(monitor.log_file.read_text().splitlines()) == 61


def test_log_failure_never_returns_or_persists_healthy(make_monitor, tmp_path):
    with health_server() as (_, url):
        invalid_log = tmp_path / "directory-not-log"
        invalid_log.mkdir()
        monitor, clock, events = make_monitor(url, log_file=invalid_log)
        with pytest.raises(OSError, match="cannot persist"):
            monitor.tick()
        state = json.loads(monitor.settings.state_file.read_text())
        assert state["status"] == "CONFIG_ERROR" and state["reason"] == "cannot_write_probe_log"
        assert monitor.last_state == state and state["consecutive_successes"] == 0
        assert json.loads(events[-1])["event"] == "CONFIG_ERROR"
        invalid_log.rmdir()
        clock.advance()
        assert monitor.tick()["status"] == "RECOVERING"
        clock.advance()
        assert monitor.tick()["status"] == "HEALTHY"


def test_log_and_state_paths_must_differ_and_cli_has_durable_default(tmp_path):
    state_file = tmp_path / "health.json"
    settings = monitor_module.parse_args(["--url", "https://test.example", "--state-file", str(state_file)])
    assert settings.log_file == tmp_path / "probe-log.jsonl"
    with pytest.raises(SystemExit):
        monitor_module.parse_args(["--url", "https://test.example", "--state-file", str(state_file),
                                   "--log-file", str(state_file)])
    with pytest.raises(monitor_module.ConfigurationError, match="different files"):
        monitor_module.Monitor(monitor_module.Settings(url="https://test.example", state_file=state_file, log_file=state_file))


def test_log_compaction_discards_unknown_secret_fields(tmp_path):
    row = {"schema_version": 1, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "HEALTHY",
           "outcome": "ok", "prompt": "SECRET-PROMPT", "body": "SECRET-RESPONSE",
           "checks": {"ready": {"http_status": 200, "latency_ms": 1, "token": "SECRET-TOKEN"}}}
    path = tmp_path / "probe.jsonl"
    path.write_text(json.dumps(row) + "\n")
    monitor_module.append_probe_log(path, row, prune=True)
    assert "SECRET" not in path.read_text()
