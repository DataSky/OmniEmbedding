#!/usr/bin/env python3
"""Externally bound one existing Colab session to a finite lease.

Run on the Mac or another reliable host, independently of the Colab VM.
The only Colab commands issued are download, stop, and sessions. This script
never reads the CLI credential/session config, creates a VM, or renews a lease.
SIGINT/SIGTERM and unexpected failures trigger cleanup of the named session.
SIGKILL, host shutdown, and a sleeping host cannot execute a Python finally;
keep the VM-side unassign deadline as an independent second layer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable
from urllib.parse import urlsplit


MAX_ENDPOINT_BYTES = 8192
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SESSION_LINE = re.compile(r"^\[([^\]\r\n]+)\]\s+\S+\s+\| Hardware: .+\| Shape: .+\| Variant: .+$")
NO_SESSIONS = "[colab] No active sessions found on server."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_endpoint(data: object, *, allow_localhost: bool = False) -> dict:
    """Only a ready public tunnel is routable from the external monitor.

    allow_localhost is an explicit test-only escape hatch, not a CLI option.
    Colab's startup placeholder must never redirect a Mac monitor to its Mac.
    """
    if not isinstance(data, dict):
        raise ValueError("endpoint_not_object")
    if data.get("status") != "ready":
        raise ValueError("endpoint_not_ready")
    url = data.get("url")
    if not isinstance(url, str) or not url or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError("invalid_endpoint_url")
    try:
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        raise ValueError("invalid_endpoint_url") from None
    if (not host or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or port == 0):
        raise ValueError("invalid_endpoint_url")
    public = parsed.scheme == "https" and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com", host)
    if not public:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if not allow_localhost or not loopback or parsed.scheme not in {"http", "https"}:
            raise ValueError("endpoint_not_public_tunnel")
    # Only supervisor metadata is copied. Arbitrary/secret-bearing extra fields
    # from a downloaded JSON object are never propagated to the local endpoint.
    keys = ("url", "status", "boot_id", "updated_at", "started_at", "expires_at",
            "model", "model_revision", "device", "dtype", "auth")
    result = {key: data[key] for key in keys if key in data}
    if any(not isinstance(value, (str, type(None))) for value in result.values()):
        raise ValueError("invalid_endpoint_metadata")
    result["url"] = url.rstrip("/")
    return result


def session_absent(output: object, name: str) -> bool:
    """Fail closed on empty, changed, ambiguous, or unrecognized CLI output."""
    if not isinstance(output, str) or len(output) > 1 << 20:
        return False
    lines = [ANSI.sub("", line).strip() for line in output.splitlines() if line.strip()]
    if lines == [NO_SESSIONS]:
        return True
    if not lines:
        return False
    names = []
    for line in lines:
        match = SESSION_LINE.fullmatch(line)
        if not match or match.group(1) == "?":
            return False  # An orphaned assignment could be our unsuccessfully released VM.
        names.append(match.group(1))
    return name not in names


@dataclass
class Settings:
    config: Path
    session: str
    state_file: Path
    endpoint_file: Path
    max_runtime: float = 7200.0
    remote_endpoint: str = "/content/omniembed-endpoint.json"
    sync_interval: float = 60.0


class LeaseGuard:
    def __init__(self, settings: Settings, *, clock: Callable[[], float] = time.monotonic,
                 runner: Callable = subprocess.run, wait: Callable[[float], bool] | None = None,
                 emit: Callable[[str], None] = print):
        self.settings, self.clock, self.runner, self.emit = settings, clock, runner, emit
        self.stop = threading.Event()
        self.wait = wait or self.stop.wait
        self.interrupted = 0
        self.deadline = 0.0
        self.last_emitted = None
        self.state = {
            "schema_version": 1, "status": "ACTIVE", "session": settings.session,
            "expires_at": None, "updated_at": utc_now(), "sync_errors": 0,
            "last_endpoint_sync_at": None, "remote_status": None,
            "stop_attempts": 0, "release_verified": False,
        }

    def request_stop(self, signum=signal.SIGTERM) -> None:
        self.interrupted = signum
        self.stop.set()

    def publish(self) -> None:
        self.state["updated_at"] = utc_now()
        try:
            atomic_json(self.settings.state_file, self.state)
        except OSError:
            # Failure to persist observability must not prevent VM cleanup.
            self.emit(json.dumps({"event": "STATE_WRITE_ERROR", "session": self.settings.session}))
        if self.last_emitted != self.state["status"]:
            self.emit(json.dumps({"event": self.state["status"], "session": self.settings.session,
                                  "updated_at": self.state["updated_at"], "expires_at": self.state["expires_at"]}))
            self.last_emitted = self.state["status"]

    def command(self, arguments: list[str], timeout: float):
        # No shell and no credential parsing. Never print stdout/stderr or
        # exception text: CLI failures may contain authenticated URLs/tokens.
        return self.runner(["colab", "--config", str(self.settings.config), *arguments],
                           timeout=timeout, capture_output=True, text=True, stdin=subprocess.DEVNULL)

    def sync_endpoint(self) -> None:
        temporary = None
        try:
            self.settings.endpoint_file.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".colab-endpoint-download.", dir=self.settings.endpoint_file.parent)
            os.close(descriptor)
            temporary = Path(name)
            remaining = self.deadline - self.clock()
            if remaining <= 0:
                return
            result = self.command(["download", "-s", self.settings.session, self.settings.remote_endpoint, str(temporary)],
                                  min(30.0, remaining))
            if result.returncode != 0:
                raise ValueError("endpoint_download_failed")
            if temporary.stat().st_size > MAX_ENDPOINT_BYTES:
                raise ValueError("endpoint_too_large")
            data = json.loads(temporary.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                remote_status = data.get("status")
                self.state["remote_status"] = remote_status if isinstance(remote_status, str) and remote_status in {"starting", "ready", "failed", "stopped"} else "unknown"
            endpoint = validate_endpoint(data)
            atomic_json(self.settings.endpoint_file, endpoint)
            self.state["last_endpoint_sync_at"] = utc_now()
            self.state["last_sync_error"] = None
        except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
            self.state["sync_errors"] += 1
            self.state["last_sync_error"] = "endpoint_sync_failed"
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            self.publish()

    def release(self) -> bool:
        for attempt in range(1, 4):
            self.state["stop_attempts"] = attempt
            try:
                self.command(["stop", "-s", self.settings.session], 30.0)
            except (OSError, subprocess.SubprocessError):
                pass
            # A successful stop alone is insufficient: the VM might still be
            # assigned, or the CLI could have lost its local session mapping.
            try:
                result = self.command(["sessions"], 30.0)
                absent = result.returncode == 0 and session_absent(result.stdout, self.settings.session)
            except (OSError, subprocess.SubprocessError):
                absent = False
            if absent:
                self.state.update(status="RELEASED", release_verified=True)
                self.publish()
                return True
        self.state.update(status="RELEASE_UNCONFIRMED", release_verified=False,
                          release_error="could_not_verify_named_session_absent")
        self.publish()
        return False

    def run(self) -> int:
        self.deadline = self.clock() + self.settings.max_runtime
        self.state["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=self.settings.max_runtime)).isoformat(timespec="seconds")
        try:
            self.publish()
            while not self.stop.is_set():
                remaining = self.deadline - self.clock()
                if remaining <= 0:
                    self.state.update(status="EXPIRED", reason="runtime_expired")
                    self.publish()
                    break
                self.sync_endpoint()
                remaining = self.deadline - self.clock()
                if remaining > 0:
                    self.wait(min(self.settings.sync_interval, remaining))
            if self.stop.is_set():
                self.state["reason"] = "interrupted"
        except Exception:
            self.state["reason"] = "guard_error"
        finally:
            released = self.release()
        if not released:
            return 1
        if self.state.get("reason") == "guard_error":
            return 1
        return 128 + self.interrupted if self.interrupted else 0


def positive_seconds(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be positive finite seconds") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive finite seconds")
    return number


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="passed verbatim to colab; never read by this guard")
    parser.add_argument("--session", required=True, help="one existing, uniquely named Colab session")
    parser.add_argument("--max-runtime", type=positive_seconds, default=7200.0)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument("--remote-endpoint", default="/content/omniembed-endpoint.json")
    parser.add_argument("--sync-interval", type=positive_seconds, default=60.0)
    settings = Settings(**vars(parser.parse_args(argv)))
    if settings.max_runtime > 86400:
        parser.error("max-runtime must not exceed 86400 seconds")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", settings.session):
        parser.error("session must be a safe, explicit name of at most 128 characters")
    if not settings.remote_endpoint.startswith("/") or any(ord(c) < 32 for c in settings.remote_endpoint):
        parser.error("remote endpoint must be an absolute path")
    if settings.state_file.resolve() == settings.endpoint_file.resolve():
        parser.error("state-file and endpoint-file must be different files")
    # Accidentally overwriting CLI credential state would destroy the ability
    # to release the lease. Comparing names does not read that private file.
    if settings.config.resolve() in {settings.state_file.resolve(), settings.endpoint_file.resolve()}:
        parser.error("guard output files must differ from the CLI config")
    return settings


def main(argv: list[str] | None = None) -> int:
    guard = LeaseGuard(parse_args(argv))
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda signum, frame: guard.request_stop(signum))
    try:
        return guard.run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
