"""Finite-lived research inference supervisor; never renews a Colab runtime.

Run inside the Colab VM. Keep monitoring outside the VM with monitor_colab.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PUBLIC_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com\b")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_endpoint(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as out:
        os.chmod(temp, 0o600)
        json.dump(value, out, ensure_ascii=False, indent=2)
    temp.replace(path)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def request_json(url, key, timeout=3):
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + key})
    # Local health checks must not accidentally use an environment HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as ex:
        try:
            body = json.load(ex)
        except (ValueError, UnicodeError):
            body = {}
        return ex.code, body


def cloudflared_binary(directory: Path) -> str:
    existing = os.environ.get("CLOUDFLARED_BIN") or shutil.which("cloudflared")
    if existing:
        return existing
    if platform.system() != "Linux":
        raise RuntimeError("Install cloudflared from Cloudflare before using --public outside Linux")
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
    if not arch:
        raise RuntimeError("Unsupported cloudflared architecture")
    name = "cloudflared-linux-" + arch
    request = urllib.request.Request(
        "https://api.github.com/repos/cloudflare/cloudflared/releases/latest",
        headers={"User-Agent": "OmniEmbedding-Colab"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    asset = next(item for item in release["assets"] if item["name"] == name)
    digest = asset.get("digest", "")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise RuntimeError("Vendor release lacks a SHA256 digest; install cloudflared manually")
    url = asset["browser_download_url"]
    if not url.startswith("https://github.com/cloudflare/cloudflared/releases/download/"):
        raise RuntimeError("Unexpected cloudflared release origin")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest[7:]:
        return str(destination)
    temp = destination.with_suffix(".download")
    checksum, size = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as out:
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > 64 * 2**20:
                    raise RuntimeError("cloudflared download exceeds its expected size")
                checksum.update(chunk)
                out.write(chunk)
        if checksum.hexdigest() != digest[7:]:
            raise RuntimeError("cloudflared SHA256 verification failed")
        temp.chmod(0o700)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"event": "cloudflared_verified", "version": release["tag_name"]}), flush=True)
    return str(destination)


class Tunnel:
    def __init__(self, binary, local_url, log_dir):
        self.url = None
        self.event = threading.Event()
        self.log_path = log_dir / "tunnel.log"
        self.process = subprocess.Popen(
            [binary, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", local_url],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        with self.log_path.open("a", encoding="utf-8") as log:
            self.log_path.chmod(0o600)
            for line in self.process.stdout:
                log.write(line)
                log.flush()
                match = PUBLIC_URL.search(line)
                if match:
                    self.url = match.group(0)
                    self.event.set()

    def wait(self, stop, timeout=90):
        deadline = time.monotonic() + timeout
        while not self.event.wait(0.5):
            if stop.is_set() or self.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Tunnel did not become available; inspect tunnel.log")
        return self.url

    def close(self):
        stop_process(self.process)
        self.reader.join(timeout=3)


def start_server(port: int, key: str, log_dir: Path):
    environment = dict(os.environ)
    environment["OMNI_INFERENCE_API_KEY"] = key
    environment["PYTHONUNBUFFERED"] = "1"
    log_path = log_dir / "inference.log"
    with log_path.open("a", encoding="utf-8") as log:
        log_path.chmod(0o600)
        return subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "omniembed.inference:create_app", "--factory",
             "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
        )


def wait_ready(process, local_url, key, stop, deadline):
    while time.monotonic() < deadline and not stop.wait(1):
        if process.poll() is not None:
            raise RuntimeError("Inference process exited; inspect inference.log")
        try:
            code, state = request_json(local_url + "/health/ready", key)
        except (OSError, ValueError):
            continue
        if code == 200 and state.get("ready"):
            return state
        if code in (401, 403):
            raise RuntimeError("Local inference authentication failed")
        if state.get("status") in ("failed", "error", "degraded"):
            raise RuntimeError("Model warmup failed (" + str(state.get("last_error", "unknown"))
                               + "); inspect inference.log")
    raise RuntimeError("Model startup exceeded the configured runtime or startup deadline")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint-file", type=Path, default=ROOT / "data/colab/endpoint.json")
    parser.add_argument("--max-runtime", type=int, default=7200, help="Seconds, including model startup")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--restart-limit", type=int, default=3)
    parser.add_argument("--public", action="store_true", help="Temporary authenticated research endpoint")
    parser.add_argument("--release-runtime-on-expiry", action="store_true",
                        help="Request Colab VM release at expiry; external lease guard is still required")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 30 <= args.max_runtime <= 86400:
        parser.error("Invalid port or runtime (30..86400 seconds)")
    if args.restart_limit < 0 or args.startup_timeout < 1:
        parser.error("Invalid restart/startup limit")
    key = args.key_file.read_text(encoding="utf-8").strip()
    if len(key) < 32 or any(character.isspace() for character in key):
        parser.error("Use an API key containing at least 32 non-whitespace characters")
    args.key_file.chmod(0o600)
    log_dir = args.endpoint_file.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    expired = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    started = time.monotonic()
    deadline = started + args.max_runtime
    local_url = "http://127.0.0.1:" + str(args.port)
    process = tunnel = None
    def expire():
        expired.set()
        stop.set()
        # This independent timer also applies during downloads and cold starts.
        # The external lease guard remains responsible for confirming release.
        if args.release_runtime_on_expiry:
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as ex:
                print(json.dumps({"event": "runtime_release_requested",
                                  "result": type(ex).__name__}), flush=True)
        if process is not None:
            stop_process(process)
        if tunnel is not None:
            stop_process(tunnel.process)

    lease_timer = threading.Timer(args.max_runtime, expire)
    lease_timer.daemon = True
    lease_timer.start()
    state = {"status": "starting", "local_url": local_url, "url": local_url,
             "started_at": utc_now(), "expires_at": datetime.fromtimestamp(
                 time.time() + args.max_runtime, timezone.utc).isoformat(), "auth": "bearer"}
    write_endpoint(args.endpoint_file, state)
    try:
        binary = cloudflared_binary(args.endpoint_file.parent / "bin") if args.public else None
        if stop.is_set():
            raise RuntimeError("Runtime expired before inference startup")
        process = start_server(args.port, key, log_dir)
        health = wait_ready(process, local_url, key, stop,
                            min(deadline, time.monotonic() + args.startup_timeout))
        if args.public:
            tunnel = Tunnel(binary, local_url, log_dir)
            state["url"] = tunnel.wait(stop, timeout=min(90, max(0, deadline - time.monotonic())))
        state.update({k: health.get(k) for k in ("boot_id", "model", "model_revision", "device", "dtype")})
        state.update(status="ready", updated_at=utc_now())
        write_endpoint(args.endpoint_file, state)
        print(json.dumps({"event": "ready", **state}), flush=True)
        server_restarts = tunnel_restarts = health_failures = 0
        while not stop.wait(min(5, max(0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                break
            stalled = False
            try:
                _, health = request_json(local_url + "/health/ready", key)
                stalled = health.get("status") == "stalled"
                health_failures = 0
            except (OSError, ValueError):
                health_failures += 1
            if process.poll() is not None or stalled or health_failures >= 3:
                if server_restarts >= args.restart_limit:
                    raise RuntimeError("Inference restart limit reached")
                server_restarts += 1
                stop_process(process)
                if stop.wait(min(2 ** server_restarts, max(0, deadline - time.monotonic()))):
                    break
                process = start_server(args.port, key, log_dir)
                health = wait_ready(process, local_url, key, stop,
                                    min(deadline, time.monotonic() + args.startup_timeout))
                health_failures = 0
                state.update(boot_id=health.get("boot_id"), updated_at=utc_now())
                write_endpoint(args.endpoint_file, state)
                print(json.dumps({"event": "inference_restarted", "count": server_restarts,
                                  "boot_id": state["boot_id"]}), flush=True)
            if tunnel is not None and tunnel.process.poll() is not None:
                if tunnel_restarts >= args.restart_limit:
                    raise RuntimeError("Tunnel restart limit reached")
                tunnel_restarts += 1
                tunnel.close()
                tunnel = Tunnel(binary, local_url, log_dir)
                state.update(url=tunnel.wait(stop, timeout=min(90, max(0, deadline - time.monotonic()))),
                             updated_at=utc_now())
                write_endpoint(args.endpoint_file, state)
                print(json.dumps({"event": "endpoint_changed", "url": state["url"]}), flush=True)
        state.update(status="stopped", reason="runtime_expired"
                     if expired.is_set() or time.monotonic() >= deadline else "requested")
    except Exception as ex:
        state.update(status="failed", reason=type(ex).__name__)
        print(json.dumps({"event": "failed", "error": type(ex).__name__, "detail": str(ex)}), flush=True)
        return 1
    finally:
        lease_timer.cancel()
        try:
            if tunnel is not None:
                tunnel.close()
        finally:
            stop_process(process)
            state["updated_at"] = utc_now()
            write_endpoint(args.endpoint_file, state)
            print(json.dumps({"event": "stopped", "reason": state.get("reason")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
