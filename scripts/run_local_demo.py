#!/usr/bin/env python3
"""Manage the local demo without installing or downloading a model.

Use the Python in data/local-demo/runtime. start/status/stop are local actions;
they do not start, stop, or change billing of the AutoDL inference instance.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def process_identity(pid):
    command = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    started = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True)
    if command.returncode or started.returncode:
        return None
    return {"command_sha256": hashlib.sha256(command.stdout.strip().encode()).hexdigest(),
            "started": started.stdout.strip(), "command": command.stdout.strip()}


def process_matches(record, directory):
    if not record or type(record.get("pid")) is not int or record["pid"] <= 1:
        return False
    if record.get("data_dir") != str(directory):
        return False
    identity = process_identity(record["pid"])
    return bool(identity and str(ROOT / "scripts/run_local_demo.py") in identity["command"]
                and re.search(r"(?:^|\s)serve(?:\s|$)", identity["command"])
                and identity["started"] == record.get("started")
                and identity["command_sha256"] == record.get("command_sha256"))


def status(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/demo/overview", timeout=5) as response:
            payload = json.load(response)
        return {"running": True, "url": f"http://127.0.0.1:{port}/demo/", "overview": payload}
    except (OSError, ValueError):
        return {"running": False, "url": f"http://127.0.0.1:{port}/demo/"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("start", "serve", "status", "stop"), default="start")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/local-demo")
    parser.add_argument("--endpoint-file", type=Path, default=ROOT / "data/autodl/endpoint.json")
    parser.add_argument("--key-file", type=Path, default=ROOT / "data/autodl/inference.key")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    directory = args.data_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    run = directory / "run"
    run.mkdir(exist_ok=True)
    log = directory / "logs"
    log.mkdir(exist_ok=True)
    record_file = run / "web-process.json"
    if args.action == "status":
        print(json.dumps(status(args.port), ensure_ascii=False, default=str))
        return 0
    if args.action == "serve":
        lock = (run / "web.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This demo data directory already has a Web process") from None
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        from omniembed.demo_app import create_app
        import uvicorn
        app = create_app(data_dir=directory, endpoint_file=args.endpoint_file.resolve(), key_file=args.key_file.resolve())
        identity = process_identity(os.getpid())
        record_file.write_text(json.dumps({"pid": os.getpid(), "port": args.port, "root": str(ROOT),
                                          "data_dir": str(directory), "started": identity["started"],
                                          "command_sha256": identity["command_sha256"]}))
        record_file.chmod(0o600)
        try:
            uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info", access_log=False)
        finally:
            if (read_json(record_file) or {}).get("pid") == os.getpid():
                record_file.unlink(missing_ok=True)
        return 0
    record = read_json(record_file)
    if args.action == "stop":
        if not process_matches(record, directory):
            print(json.dumps({"stopped": False, "reason": "No verified demo process; no signal sent"}))
            return 1
        os.kill(record["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process_matches(record, directory):
            time.sleep(.2)
        if process_matches(record, directory):
            print(json.dumps({"stopped": False, "reason": "Graceful shutdown pending; no forced kill"}))
            return 1
        print(json.dumps({"stopped": True, "autodl_instance_changed": False}))
        return 0
    if process_matches(record, directory):
        print(json.dumps(status(record.get("port", args.port)), ensure_ascii=False, default=str))
        return 0
    with socket.socket() as listener:
        if listener.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit("Port is occupied by another process; choose --port")
    command = [sys.executable, str(Path(__file__).resolve()), "serve", "--port", str(args.port),
               "--data-dir", str(directory), "--endpoint-file", str(args.endpoint_file.resolve()), "--key-file", str(args.key_file.resolve())]
    with (log / "web.log").open("ab", buffering=0) as output:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise SystemExit("Web process exited; inspect data/local-demo/logs/web.log")
        current = status(args.port)
        if current["running"]:
            print(json.dumps({"running": True, "pid": child.pid, "url": current["url"], "logs": str(log / "web.log")}, ensure_ascii=False))
            return 0
        time.sleep(.3)
    print(json.dumps({"launch_requested": True, "pid": child.pid, "logs": str(log / "web.log")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
