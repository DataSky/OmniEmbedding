"""Process-control tests never signal or launch a real process."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from scripts import code_search_control as control


def identity(command):
    return {"command": command, "started": "Sun Oct 18 18:03:04 2026",
            "command_sha256": hashlib.sha256(command.encode()).hexdigest()}


def record_for(tmp_path, pid=12345):
    root, directory = tmp_path / "source", tmp_path / "service"
    root.mkdir(exist_ok=True)
    run = directory / "run"
    run.mkdir(parents=True, exist_ok=True)
    command = f"/usr/bin/python3 {control.RUNNER} --root {root} --data-dir {directory} --port 8790 --sync"
    value = {"pid": pid, "root": str(root), "data_dir": str(directory), "port": 8790,
             **{k: v for k, v in identity(command).items() if k != "command"}}
    path = run / "service-process.json"
    path.write_text(json.dumps(value))
    return root, directory, path, value, identity(command)


def test_exact_runner_path_and_known_arguments_are_required():
    value = identity(f"/usr/bin/python3 {control.RUNNER} --sync")
    assert control.runner_parameters(value) == {"root": str(Path.home() / "AiCode"),
                                               "data_dir": str(control.ROOT / "data/code-search"), "port": 8790}
    assert control.runner_parameters(identity(f"/usr/bin/python3 {control.RUNNER}.copy --sync")) is None
    assert control.runner_parameters(identity(f"/usr/bin/python3 {control.RUNNER} --unknown")) is None
    assert control.runner_parameters(identity("/usr/bin/python3 /tmp/unrelated.py")) is None


@pytest.mark.parametrize("changed", [{"started": "another launch"}, {"command_sha256": "different"},
                                     {"command": "/usr/bin/python3 /tmp/unrelated.py"}, None])
def test_pid_reuse_or_unrelated_command_cannot_pass_verification(tmp_path, monkeypatch, changed):
    _, directory, _, record, current = record_for(tmp_path)
    altered = {**current, **changed} if changed is not None else None
    monkeypatch.setattr(control, "process_identity", lambda pid: altered)
    assert not control.process_matches(record, directory)


def test_atomic_process_record_requires_exact_parameters_and_has_private_mode(tmp_path, monkeypatch):
    root, directory, path, record, current = record_for(tmp_path)
    monkeypatch.setattr(control, "process_identity", lambda pid: current)
    saved = control.write_record(path, record["pid"], root, directory, 8790)
    assert saved == record
    assert path.stat().st_mode & 0o777 == 0o600
    assert "command" not in json.loads(path.read_text())
    assert control.write_record(path, record["pid"], tmp_path / "different", directory, 8790) is None
    assert json.loads(path.read_text()) == record


def test_record_preserves_argument_boundaries_for_spaces_quotes_and_shell_characters(tmp_path, monkeypatch):
    root = tmp_path / "source with 'quotes' ; $(untouched)"
    directory = tmp_path / "service with spaces"
    run = directory / "run"
    run.mkdir(parents=True)
    argv = ["/usr/bin/python3", str(control.RUNNER), "--root", str(root),
            "--data-dir", str(directory), "--port", "8790", "--scan-only"]
    current = identity(" ".join(argv))  # ps output need not quote original arguments.
    monkeypatch.setattr(control, "process_identity", lambda pid: current)
    record = control.write_record(run / "service-process.json", 12345, root, directory, 8790, argv=argv)
    assert record and record["root"] == str(root) and record["argv"] == argv
    assert control.process_matches(record, directory)
    assert not (tmp_path / "untouched").exists()


def test_legacy_adoption_requires_both_process_and_http_identity(tmp_path, monkeypatch):
    root, directory, path, record, current = record_for(tmp_path)
    path.unlink()
    (directory / "service.pid").write_text(str(record["pid"]))
    monkeypatch.setattr(control, "process_identity", lambda pid: current)
    monkeypatch.setattr(control, "service_overview", lambda *args: None)
    assert control.adopt_legacy_process(path, root, directory, 8790) is None
    assert not path.exists()
    monkeypatch.setattr(control, "service_overview", lambda *args: {"model": control.MODEL})
    assert control.adopt_legacy_process(path, root, directory, 8790) == record
    assert control.process_matches(record, directory)


def forbid(*args, **kwargs):
    raise AssertionError("must not launch or signal a process")


def test_status_has_compact_progress_and_never_launches_or_signals(tmp_path, monkeypatch, capsys):
    root, directory, _, record, current = record_for(tmp_path)
    monkeypatch.setattr(control, "process_identity", lambda pid: current)
    monkeypatch.setattr(control.subprocess, "Popen", forbid)
    monkeypatch.setattr(control.os, "kill", forbid)
    monkeypatch.setattr(control, "service_overview", lambda *args: {
        "model": control.MODEL, "revision": "fixed", "files": 7, "chunks": 10, "vectors": 5,
        "job": {"status": "running", "phase": "embed", "vectors_done": 5, "vectors_total": 10,
                "details": {"do_not_duplicate": "big payload"}}})
    assert control.main(["status", "--root", str(root), "--data-dir", str(directory)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["process_verified"] and result["service_ready"]
    assert result["pid"] == record["pid"] and result["vectors"] == 5
    assert result["job"]["vectors_total"] == 10 and "details" not in result["job"]


def test_stop_rejects_unverified_pid_without_sending_signal(tmp_path, monkeypatch, capsys):
    root, directory, _, _, _ = record_for(tmp_path)
    monkeypatch.setattr(control, "process_identity", lambda pid: None)
    monkeypatch.setattr(control.os, "kill", forbid)
    monkeypatch.setattr(control.subprocess, "Popen", forbid)
    assert control.main(["stop", "--root", str(root), "--data-dir", str(directory)]) == 1
    assert json.loads(capsys.readouterr().out)["stopped"] is False


def test_stop_signals_only_verified_pid_and_removes_own_record(tmp_path, monkeypatch, capsys):
    root, directory, path, record, _ = record_for(tmp_path)
    checks = []

    def matches(*args):
        checks.append(True)
        return len(checks) <= 3

    signals = []
    monkeypatch.setattr(control, "process_matches", matches)
    monkeypatch.setattr(control.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(control.subprocess, "Popen", forbid)
    assert control.main(["stop", "--root", str(root), "--data-dir", str(directory)]) == 0
    assert signals == [(record["pid"], signal.SIGTERM)]
    assert not path.exists()
    assert json.loads(capsys.readouterr().out)["autodl_instance_changed"] is False


def test_start_reuses_existing_service_without_spawn(tmp_path, monkeypatch, capsys):
    root, directory, _, _, current = record_for(tmp_path)
    monkeypatch.setattr(control, "process_identity", lambda pid: current)
    monkeypatch.setattr(control, "service_overview", lambda *args: None)
    monkeypatch.setattr(control.subprocess, "Popen", forbid)
    monkeypatch.setattr(control.os, "kill", forbid)
    assert control.main(["start", "--root", str(root), "--data-dir", str(directory), "--sync"]) == 0
    assert json.loads(capsys.readouterr().out)["reused"] is True


@pytest.mark.parametrize("flag", ["--sync", "--scan-only", None])
def test_start_constructs_only_local_runner_command(tmp_path, monkeypatch, capsys, flag):
    root, directory, path, record, current = record_for(tmp_path)
    path.unlink()
    commands = []
    monkeypatch.setattr(control, "process_matches", lambda record, directory: bool(record))
    monkeypatch.setattr(control, "adopt_legacy_process", lambda *args: None)
    monkeypatch.setattr(control, "write_record", lambda *args, **kwargs: record)
    monkeypatch.setattr(control, "service_overview", lambda *args: {"model": control.MODEL})
    monkeypatch.setattr(control.os, "kill", forbid)

    class Listener:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect_ex(self, address):
            assert address == ("127.0.0.1", 8790)
            return 1

    monkeypatch.setattr(control.socket, "socket", Listener)

    def popen(command, **kwargs):
        commands.append(command)
        assert kwargs["start_new_session"] and kwargs["cwd"] == control.ROOT
        return SimpleNamespace(pid=record["pid"], poll=lambda: None)

    monkeypatch.setattr(control.subprocess, "Popen", popen)
    args = ["start", "--root", str(root), "--data-dir", str(directory)]
    assert control.main(args + ([flag] if flag else [])) == 0
    assert len(commands) == 1 and commands[0][1] == str(control.RUNNER)
    assert (flag in commands[0]) if flag else "--sync" not in commands[0] and "--scan-only" not in commands[0]
    assert (directory / "service.log").is_file()
    assert json.loads(capsys.readouterr().out)["autodl_instance_changed"] is False
