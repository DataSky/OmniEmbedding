"""Lease expiry/cleanup behavior without creating or deleting real Colab VMs."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colab_lease_guard.py"
spec = importlib.util.spec_from_file_location("colab_lease_guard", SCRIPT)
lease = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lease
spec.loader.exec_module(lease)
SESSION = "omni-test-20261017"
ACTIVE = f"[{SESSION}] vm-endpoint | Hardware: T4 | Shape: STANDARD | Variant: GPU\n"
OTHER = "[other-session] other-endpoint | Hardware: CPU | Shape: STANDARD | Variant: DEFAULT\n"
READY = {"status": "ready", "url": "https://honest-tunnel.trycloudflare.com", "boot_id": "boot-1"}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds
        return False


class CLI:
    """Boundary double: speaks CLI arguments/files and records their side effects."""
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.downloads = [READY]
        self.sessions = [lease.NO_SESSIONS]
        self.stop_exception = None

    def __call__(self, arguments, **kwargs):
        self.calls.append((self.clock(), arguments, kwargs))
        assert arguments[:3] == ["colab", "--config", "/not-read/private-sessions.json"]
        assert kwargs["capture_output"] and kwargs["text"] and kwargs["stdin"] == subprocess.DEVNULL
        assert 0 < kwargs["timeout"] <= 30
        command = arguments[3]
        assert command in {"download", "stop", "sessions"}
        if command == "download":
            assert arguments[4:7] == ["-s", SESSION, "/content/omniembed-endpoint.json"]
            body = self.downloads.pop(0) if len(self.downloads) > 1 else self.downloads[0]
            if isinstance(body, Exception):
                raise body
            if body is None:
                return subprocess.CompletedProcess(arguments, 1, stdout="token=CLI-SECRET", stderr="CLI-SECRET")
            Path(arguments[7]).write_text(body if isinstance(body, str) else json.dumps(body))
            return subprocess.CompletedProcess(arguments, 0, stdout="CLI-SECRET", stderr="")
        if command == "stop":
            assert arguments[4:] == ["-s", SESSION]
            if self.stop_exception:
                raise self.stop_exception
            return subprocess.CompletedProcess(arguments, 0, stdout="CLI-SECRET", stderr="")
        assert arguments[4:] == []
        output = self.sessions.pop(0) if len(self.sessions) > 1 else self.sessions[0]
        if isinstance(output, Exception):
            raise output
        if isinstance(output, subprocess.CompletedProcess):
            return output
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")


@pytest.fixture
def make_guard(tmp_path):
    def make(**kwargs):
        clock, events = Clock(), []
        cli = CLI(clock)
        settings = lease.Settings(config=Path("/not-read/private-sessions.json"), session=SESSION,
                                  state_file=tmp_path / "lease.json", endpoint_file=tmp_path / "endpoint.json",
                                  **kwargs)
        guard = lease.LeaseGuard(settings, clock=clock, runner=cli, wait=clock.wait, emit=events.append)
        return guard, clock, cli, events
    return make


def test_expiry_only_releases_named_session_and_requires_verification(make_guard):
    guard, clock, cli, events = make_guard(max_runtime=180)
    cli.downloads = [None, {"status": "starting", "url": "http://127.0.0.1:8000"}, READY]
    assert guard.run() == 0
    assert [json.loads(event)["event"] for event in events] == ["ACTIVE", "EXPIRED", "RELEASED"]
    assert guard.state["status"] == "RELEASED" and guard.state["release_verified"]
    assert guard.state["sync_errors"] == 2
    stop = [call for call in cli.calls if call[1][3] == "stop"]
    assert len(stop) == 1 and stop[0][0] == 180  # Missing startup files never expire the lease early.
    assert json.loads(guard.settings.endpoint_file.read_text()) == READY
    assert json.loads(guard.settings.state_file.read_text()) == guard.state
    assert guard.settings.endpoint_file.stat().st_mode & 0o077 == 0
    assert guard.settings.state_file.stat().st_mode & 0o077 == 0
    assert "CLI-SECRET" not in "".join(events) + guard.settings.state_file.read_text()
    assert not list(guard.settings.endpoint_file.parent.glob(".colab-endpoint-download.*"))


def test_download_timeout_cannot_cross_lease_deadline(make_guard):
    guard, clock, cli, _ = make_guard(max_runtime=7)
    original = cli.__call__

    def timeout_download(arguments, **kwargs):
        if arguments[3] == "download":
            assert kwargs["timeout"] == 7
            clock.now += kwargs["timeout"]
            raise subprocess.TimeoutExpired(arguments, kwargs["timeout"], output="CLI-SECRET")
        return original(arguments, **kwargs)

    guard.runner = timeout_download
    assert guard.run() == 0
    assert clock() == 7 and guard.state["sync_errors"] == 1
    assert cli.calls[0][1][3] == "stop" and cli.calls[0][0] == 7


def test_loopback_and_unready_do_not_replace_last_valid_endpoint(make_guard):
    guard, _, cli, _ = make_guard(max_runtime=240)
    cli.downloads = [READY, {"status": "ready", "url": "http://localhost:8000"},
                     {"status": "stopped", "url": READY["url"]}, {"status": [], "url": READY["url"]}]
    assert guard.run() == 0
    assert guard.state["sync_errors"] == 3
    assert json.loads(guard.settings.endpoint_file.read_text()) == READY
    assert guard.state["remote_status"] == "unknown"


@pytest.mark.parametrize("body", ["not-json", "[]", "x" * 8193,
    {"status": "ready", "url": "https://attacker.example"},
    {"status": "ready", "url": "https://fake.trycloudflare.com@evil.example"},
    {"status": "ready", "url": "https://test.trycloudflare.com?token=secret"},
    {"status": "ready", "url": "https://test.trycloudflare.com", "boot_id": {"token": "secret"}}])
def test_invalid_download_is_bounded_and_cannot_cause_early_cleanup(make_guard, body):
    guard, _, cli, _ = make_guard(max_runtime=60)
    cli.downloads = [body]
    assert guard.run() == 0
    assert guard.state["sync_errors"] == 1
    assert not guard.settings.endpoint_file.exists()
    assert next(call[0] for call in cli.calls if call[1][3] == "stop") == 60


def test_endpoint_drops_secret_fields_and_loopback_override_is_explicit():
    cleaned = lease.validate_endpoint({**READY, "token": "secret", "api_key": "secret", "local_url": "http://127.0.0.1:8000"})
    assert cleaned == READY
    local = {"status": "ready", "url": "http://127.0.0.1:8000"}
    with pytest.raises(ValueError):
        lease.validate_endpoint(local)
    assert lease.validate_endpoint(local, allow_localhost=True) == local


def test_stop_success_does_not_claim_release_while_session_remains(make_guard):
    guard, _, cli, events = make_guard(max_runtime=1)
    cli.sessions = [ACTIVE]
    assert guard.run() == 1
    assert guard.state["status"] == "RELEASE_UNCONFIRMED"
    assert not guard.state["release_verified"]
    assert guard.state["stop_attempts"] == 3
    assert sum(call[1][3] == "stop" for call in cli.calls) == 3
    assert not any(json.loads(event)["event"] == "RELEASED" for event in events)


def test_stop_timeout_is_verified_and_retried_boundedly(make_guard):
    guard, _, cli, _ = make_guard(max_runtime=1)
    cli.stop_exception = subprocess.TimeoutExpired("stop", 30, output="CLI-SECRET")
    cli.sessions = [ACTIVE, ACTIVE, lease.NO_SESSIONS]
    assert guard.run() == 0
    assert guard.state["stop_attempts"] == 3 and guard.state["release_verified"]


@pytest.mark.parametrize("output", ["", "login required", "[?] vm | Hardware: T4 | Shape: STANDARD | Variant: GPU", "truncated", subprocess.CompletedProcess([], 1, stdout=lease.NO_SESSIONS)])
def test_unknown_session_output_never_claims_release(make_guard, output):
    guard, _, cli, _ = make_guard(max_runtime=1)
    cli.sessions = [output]
    assert guard.run() == 1
    assert guard.state["status"] == "RELEASE_UNCONFIRMED"


def test_exact_session_name_match_and_other_vms_are_preserved():
    assert not lease.session_absent(ACTIVE + OTHER, SESSION)
    assert lease.session_absent(OTHER, SESSION)
    assert lease.session_absent(ACTIVE.replace(SESSION, SESSION + "-other"), SESSION)
    assert not lease.session_absent("\x1b[31m" + ACTIVE + "\x1b[0m", SESSION)


def test_signal_and_unexpected_error_always_release(make_guard):
    guard, _, cli, _ = make_guard(max_runtime=3600)
    guard.wait = lambda seconds: guard.request_stop(signal.SIGTERM)
    assert guard.run() == 143
    assert guard.state["status"] == "RELEASED" and guard.state["reason"] == "interrupted"
    assert sum(call[1][3] == "stop" for call in cli.calls) == 1
    guard, _, cli, _ = make_guard(max_runtime=3600)

    def broken_sync():
        raise RuntimeError("CLI-SECRET")

    guard.sync_endpoint = broken_sync
    assert guard.run() == 1
    assert guard.state["status"] == "RELEASED" and guard.state["reason"] == "guard_error"
    assert sum(call[1][3] == "stop" for call in cli.calls) == 1


@pytest.fixture
def fake_cli_environment(tmp_path):
    """An actual child executable verifies CLI wiring/signal cleanup, never a VM."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.jsonl"
    program = bin_dir / "colab"
    program.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["LEASE_TEST_CALLS"], "a") as out:
    out.write(json.dumps(args) + "\\n")
if args[2] == "download":
    pathlib.Path(args[-1]).write_text(json.dumps({"status":"ready","url":"https://test.trycloudflare.com","boot_id":"boot-1"}))
elif args[2] == "sessions":
    print("[colab] No active sessions found on server.")
''')
    program.chmod(0o700)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""), "LEASE_TEST_CALLS": str(calls)}
    command = [sys.executable, str(SCRIPT), "--config", "/not-read/private-sessions.json", "--session", SESSION,
               "--state-file", str(tmp_path / "lease.json"), "--endpoint-file", str(tmp_path / "endpoint.json"),
               "--sync-interval", "0.1"]
    return command, env, calls, tmp_path / "lease.json"


def test_real_process_finite_lease_and_cli_arguments(fake_cli_environment):
    command, env, calls, state_file = fake_cli_environment
    result = subprocess.run(command + ["--max-runtime", "0.2"], env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert json.loads(state_file.read_text())["status"] == "RELEASED"
    invoked = [json.loads(line) for line in calls.read_text().splitlines()]
    assert invoked[-2:] == [["--config", "/not-read/private-sessions.json", "stop", "-s", SESSION],
                            ["--config", "/not-read/private-sessions.json", "sessions"]]


def test_real_process_sigterm_cleans_up_named_session(fake_cli_environment):
    command, env, calls, state_file = fake_cli_environment
    process = subprocess.Popen(command + ["--max-runtime", "3600"], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not calls.exists():
            time.sleep(0.01)
        assert calls.exists(), "test CLI did not start"
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 143, stderr
        assert json.loads(state_file.read_text())["status"] == "RELEASED"
        invoked = [json.loads(line) for line in calls.read_text().splitlines()]
        assert any(args[2:] == ["stop", "-s", SESSION] for args in invoked)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_parser_will_not_overwrite_cli_config(tmp_path):
    args = ["--config", str(tmp_path / "private.json"), "--session", SESSION,
            "--state-file", str(tmp_path / "private.json"), "--endpoint-file", str(tmp_path / "endpoint.json")]
    with pytest.raises(SystemExit):
        lease.parse_args(args)
