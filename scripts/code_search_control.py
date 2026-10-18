#!/usr/bin/env python3
"""管理本机 OmniIndex 服务；不会启动、停止或调整 AutoDL 实例。"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/run_code_search.py"
MODEL = "google/embeddinggemma-2"


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def process_identity(pid):
    """Record both launch time and command to reject PID reuse."""
    if type(pid) is not int or pid <= 1:
        return None
    try:
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True, timeout=3)
        started = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                                 capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if command.returncode or started.returncode or not command.stdout.strip() or not started.stdout.strip():
        return None
    value = command.stdout.strip()
    return {"command_sha256": hashlib.sha256(value.encode()).hexdigest(),
            "started": started.stdout.strip(), "command": value}


def runner_parameters(identity, recorded_argv=None):
    """Accept only the exact runner script and its known command arguments."""
    if not identity:
        return None
    try:
        if recorded_argv is not None:
            if not isinstance(recorded_argv, list) or not all(isinstance(value, str) for value in recorded_argv):
                return None
            # ps flattens argument boundaries; retain Popen's original argv so
            # spaces and quotes in source paths remain unambiguous. The full
            # live ps command is still pinned by its recorded SHA and start.
            if not re.search(r"(?:^|[\s'\"])" + re.escape(str(RUNNER)) + r"(?=$|[\s'\"])", identity["command"]):
                return None
            argv = recorded_argv
        else:
            argv = shlex.split(identity["command"])
        position = argv.index(str(RUNNER))
        if position < 1:
            return None
        parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
        parser.add_argument("--root", type=Path, default=Path.home() / "AiCode")
        parser.add_argument("--data-dir", type=Path, default=ROOT / "data/code-search")
        parser.add_argument("--port", type=int, default=8790)
        parser.add_argument("--endpoint-file")
        parser.add_argument("--key-file")
        parser.add_argument("--sync", action="store_true")
        parser.add_argument("--scan-only", action="store_true")
        args, unknown = parser.parse_known_args(argv[position + 1:])
        if unknown:
            return None
        return {"root": str(args.root.expanduser().resolve()), "data_dir": str(args.data_dir.expanduser().resolve()),
                "port": args.port}
    except (ValueError, KeyError, argparse.ArgumentError):
        return None


def process_matches(record, directory):
    if not record or record.get("data_dir") != str(directory):
        return False
    identity = process_identity(record.get("pid"))
    parameters = runner_parameters(identity, record.get("argv"))
    return bool(parameters and all(parameters[key] == record.get(key) for key in ("root", "data_dir", "port"))
                and identity["started"] == record.get("started")
                and identity["command_sha256"] == record.get("command_sha256"))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def service_overview(port, root):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/code/overview", timeout=3) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            return None
        value = json.loads(raw)
        if (not isinstance(value, dict) or value.get("model") != MODEL or value.get("dimensions") != 768
                or value.get("root") != str(root)):
            return None
        return value
    except (OSError, ValueError, UnicodeError, urllib.error.URLError):
        return None


def write_record(path, pid, root, directory, port, argv=None):
    identity = process_identity(pid)
    parameters = runner_parameters(identity, argv)
    expected = {"root": str(root), "data_dir": str(directory), "port": port}
    if parameters != expected:
        return None
    record = {"pid": pid, **expected, "started": identity["started"],
              "command_sha256": identity["command_sha256"]}
    if argv is not None:
        record["argv"] = argv
    descriptor, temporary = tempfile.mkstemp(prefix=".service-process-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(record, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return record


def adopt_legacy_process(record_file, root, directory, port):
    """Register an existing runner only after process AND HTTP identity checks."""
    try:
        pid = int((directory / "service.pid").read_text().strip())
    except (OSError, ValueError, UnicodeError):
        return None
    identity = process_identity(pid)
    if runner_parameters(identity) != {"root": str(root), "data_dir": str(directory), "port": port}:
        return None
    if service_overview(port, root) is None:
        return None
    return write_record(record_file, pid, root, directory, port)


def status(record, directory, port, root):
    verified = process_matches(record, directory)
    actual_port = record["port"] if verified else port
    actual_root = Path(record["root"]) if verified else root
    overview = service_overview(actual_port, actual_root) if verified else None
    result = {"running": verified, "process_verified": verified, "service_ready": overview is not None,
              "url": f"http://127.0.0.1:{actual_port}/", "root": str(actual_root),
              "data_dir": str(directory), "autodl_instance_changed": False}
    if verified:
        result["pid"] = record["pid"]
    if overview is not None:
        result.update(model=overview["model"], revision=overview.get("revision"),
                      files=overview.get("files", 0), chunks=overview.get("chunks", 0), vectors=overview.get("vectors", 0))
        job = overview.get("job")
        result["job"] = {key: job.get(key) for key in ("id", "action", "status", "phase", "files_done", "chunks_done", "vectors_done", "vectors_total", "errors")} if isinstance(job, dict) else None
    elif verified:
        result["message"] = "本机服务进程已核验，网页接口暂未就绪。"
    else:
        result["message"] = "没有已核验的本机服务进程。"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("action", choices=("start", "status", "stop"), nargs="?", default="status",
                        help="start：后台启动；status：读取状态；stop：停止已核验的本机服务")
    parser.add_argument("--root", type=Path, default=Path.home() / "AiCode", help="源目录，默认 ~/AiCode")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/code-search", help="独立数据库与运行目录")
    parser.add_argument("--port", type=int, default=8790, help="本机网页端口，默认 8790")
    parser.add_argument("--endpoint-file", type=Path, default=ROOT / "data/autodl/endpoint.json", help="AutoDL 地址配置文件")
    parser.add_argument("--key-file", type=Path, default=ROOT / "data/autodl/inference.key", help="私有推理密钥文件，不会打印内容")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--sync", action="store_true", help="仅 start：启动后扫描并续跑缺失向量")
    actions.add_argument("--scan-only", action="store_true", help="仅 start：只扫描并建立 BM25，无需 GPU")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("端口需为 1024—65535")
    if args.action != "start" and (args.sync or args.scan_only):
        parser.error("--sync 与 --scan-only 仅适用于 start")
    root, directory = args.root.expanduser().resolve(), args.data_dir.expanduser().resolve()
    if args.action == "start" and not root.is_dir():
        parser.error("源目录不存在")
    run = directory / "run"
    run.mkdir(parents=True, exist_ok=True)
    record_file = run / "service-process.json"
    with (run / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = read_json(record_file)
        if not process_matches(record, directory):
            record = adopt_legacy_process(record_file, root, directory, args.port)
        if args.action == "status":
            result = status(record, directory, args.port, root)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["process_verified"] else 1
        if args.action == "stop":
            if not process_matches(record, directory):
                print(json.dumps({"stopped": False, "message": "进程身份未通过核验，没有发送停止信号。"}, ensure_ascii=False))
                return 1
            # Recheck directly before signaling; never signal a PID alone.
            if not process_matches(record, directory):
                return 1
            try:
                os.kill(record["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and process_matches(record, directory):
                time.sleep(.2)
            stopped = not process_matches(record, directory)
            if stopped and read_json(record_file) == record:
                record_file.unlink(missing_ok=True)
                # Keep the legacy marker from becoming an adoption candidate.
                legacy = directory / "service.pid"
                try:
                    if legacy.read_text().strip() == str(record["pid"]):
                        legacy.unlink()
                except OSError:
                    pass
            print(json.dumps({"stopped": stopped, "autodl_instance_changed": False,
                              "message": "本机服务已停止，索引保留。" if stopped else "仍在等待正常退出，没有强制终止。"}, ensure_ascii=False))
            return 0 if stopped else 1
        if process_matches(record, directory):
            if record["root"] != str(root):
                parser.error("该数据目录已有其他源目录的服务运行，请指定新的 --data-dir。")
            print(json.dumps({**status(record, directory, args.port, root), "reused": True}, ensure_ascii=False))
            return 0
        with socket.socket() as listener:
            if listener.connect_ex(("127.0.0.1", args.port)) == 0:
                parser.error("端口已被其他进程使用，没有启动或停止任何服务。")
        command = [sys.executable, str(RUNNER), "--root", str(root), "--data-dir", str(directory),
                   "--port", str(args.port), "--endpoint-file", str(args.endpoint_file.expanduser().resolve()),
                   "--key-file", str(args.key_file.expanduser().resolve())]
        if args.sync:
            command.append("--sync")
        elif args.scan_only:
            command.append("--scan-only")
        log = directory / "service.log"
        with log.open("ab", buffering=0) as output:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if child.poll() is not None:
                print(json.dumps({"running": False, "message": "本机服务启动失败，请查看日志。", "log": str(log)}, ensure_ascii=False))
                return 1
            record = write_record(record_file, child.pid, root, directory, args.port, argv=command)
            if record and service_overview(args.port, root) is not None:
                print(json.dumps({**status(record, directory, args.port, root), "log": str(log)}, ensure_ascii=False))
                return 0
            time.sleep(.2)
        print(json.dumps({"launch_requested": True, "pid": child.pid, "url": f"http://127.0.0.1:{args.port}/",
                          "log": str(log), "message": "启动请求已提交，可用 status 继续查看。"}, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
