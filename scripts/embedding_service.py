#!/usr/bin/env python3
"""Portable AutoDL embedding-service workflow; no background scheduler.

Run beside autodl_service_manager.py, autodl_service_probe.py and control.json.
Explicit start powers on a stopped existing instance, then verifies the service.
It never automatically shuts down, creates or releases an instance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import time


class WorkflowError(Exception):
    def __init__(self, code, *, detail=None, unknown=False):
        super().__init__(code)
        self.code, self.detail, self.unknown = code, detail, unknown


def atomic_json(path, value):
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise WorkflowError("output_requires_regular_file")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".workflow-", delete=False) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_workflow(path):
    path = Path(os.path.abspath(path))
    try:
        with path.open("rb") as handle:
            raw = handle.read(65537)
        if len(raw) > 65536:
            raise WorkflowError("configuration_too_large")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise WorkflowError("configuration_invalid")
    except (OSError, ValueError, RecursionError):
        raise WorkflowError("configuration_unreadable_or_invalid") from None
    def resolve(name, fallback):
        value = data.get(name, fallback)
        if not isinstance(value, str) or not value or "\x00" in value:
            raise WorkflowError("configuration_path_invalid")
        return Path(os.path.abspath(path.parent / value))
    data.update(config_file=path,
                client_env_file=resolve("client_env_file", "RTX5090.client.env"),
                endpoint_file=resolve("endpoint_file", "endpoint.json"))
    return data


class Workflow:
    def __init__(self, config, *, timeout=900, executor=subprocess.run, clock=time.monotonic,
                 script_dir=None):
        if isinstance(timeout, bool) or not isinstance(timeout, (int,float)) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise WorkflowError("timeout_outside_1_to_3600_seconds")
        self.config, self.execute, self.clock = config, executor, clock
        self.started = clock()
        self.deadline = self.started + timeout
        self.scripts = Path(script_dir) if script_dir else Path(__file__).parent
        self.steps = []
        self.pinned_endpoint = None
        self.expected_configuration = None
        self._temporary = None

    def remaining(self):
        value = self.deadline - self.clock()
        if value <= 0:
            raise WorkflowError("workflow_deadline_exceeded")
        return value

    def child(self, name, script, arguments, *, mutating=False):
        command = [sys.executable, str(self.scripts / script), *map(str, arguments)]
        try:
            result = self.execute(command, capture_output=True, text=True,
                                  timeout=self.remaining(), check=False)
        except subprocess.TimeoutExpired:
            raise WorkflowError(name + "_timeout", unknown=mutating) from None
        except OSError:
            raise WorkflowError(name + "_could_not_start") from None
        try:
            payload = json.loads(result.stdout)
        except (ValueError, TypeError, RecursionError):
            raise WorkflowError(name + "_invalid_child_report", unknown=mutating) from None
        if not isinstance(payload, dict):
            raise WorkflowError(name + "_invalid_child_report", unknown=mutating)
        self.steps.append({"step":name,"report":payload})
        passed = payload.get("ok") is True if "ok" in payload else payload.get("status")=="pass"
        if result.returncode != 0 or not passed:
            raise WorkflowError(name + "_failed", detail=payload,
                                unknown=payload.get("outcome_unknown") is True)
        return payload

    def manager(self, command, *arguments, mutating=False):
        return self.child(command, "autodl_service_manager.py",
                          ["--config", self.config["config_file"], command, *arguments], mutating=mutating)

    def probe(self, command):
        return self.child(command, "autodl_service_probe.py", [command,
                          "--env", self.config["client_env_file"],
                          "--endpoint-file", self.pinned_endpoint or self.config["endpoint_file"],
                          "--timeout-seconds", self.remaining()])

    def probe_module(self):
        spec=importlib.util.spec_from_file_location("_handoff_probe",self.scripts/"autodl_service_probe.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def pin_configuration(self):
        module=self.probe_module()
        url,key=module.load_config(self.config["client_env_file"],self.config["endpoint_file"])
        self.expected_configuration=(url,hashlib.sha256(key.encode()).hexdigest())
        self._temporary=tempfile.TemporaryDirectory(prefix="embedding-validation-")
        self.pinned_endpoint=Path(self._temporary.name)/"endpoint.json"
        atomic_json(self.pinned_endpoint,{"url":url})

    def publish_verified_configuration(self, ready):
        """Only update caller-facing URL after authenticated identity validation."""
        module = self.probe_module()
        url, key = module.load_config(self.config["client_env_file"], self.config["endpoint_file"])
        if self.expected_configuration != (url,hashlib.sha256(key.encode()).hexdigest()):
            raise WorkflowError("configuration_changed_after_validation")
        endpoint_raw=self.config["endpoint_file"].read_bytes()
        endpoint=json.loads(endpoint_raw)
        if endpoint.get("url","").rstrip("/") != url:
            raise WorkflowError("endpoint_changed_after_validation")
        env_path = self.config["client_env_file"]
        if env_path.is_symlink() or not env_path.is_file():
            raise WorkflowError("client_environment_requires_regular_file")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=env_path.parent,
                                             prefix=".client-env-", delete=False) as handle:
                temporary = Path(handle.name)
                os.chmod(temporary, 0o600)
                handle.write("export OMNI_INFERENCE_URL="+shlex.quote(url)+"\n")
                handle.write("export OMNI_INFERENCE_API_KEY="+shlex.quote(key)+"\n")
            os.replace(temporary, env_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        if self.config["endpoint_file"].read_bytes()!=endpoint_raw:
            raise WorkflowError("endpoint_changed_during_configuration_publish")
        endpoint.update(status="ready", model=module.MODEL, model_revision=module.REVISION,
                        verified_at=datetime.now(timezone.utc).isoformat(), boot_id=ready.get("boot_id"))
        atomic_json(self.config["endpoint_file"], endpoint)
        self.steps.append({"step":"configuration_synchronized","url":url,"credentials_printed":False})

    def run(self, command, *, test_after_start=False, name=None, image_id=None):
        report = {"schema_version":1,"command":command,"status":"fail","steps":self.steps}
        try:
            if command == "start":
                self.manager("power-on", "--wait", "--wait-timeout", self.remaining(), mutating=True)
                self.manager("refresh-endpoint")
                self.pin_configuration()
                ready = self.probe("wait-ready")
                if test_after_start:
                    ready = self.probe("test")
                self.publish_verified_configuration(ready)
                report["service_verified"] = True
                report["modalities_tested"] = bool(test_after_start)
            elif command == "status":
                self.manager("status")
                self.probe("ready")
            elif command == "test":
                self.probe("test")
            elif command == "images":
                self.manager("images")
            elif command == "image-save":
                if not name:
                    raise WorkflowError("image_name_required")
                saved=self.manager("image-save", "--name", name, mutating=True)
                accepted=saved.get("data",{}).get("image_uuid")
                if not isinstance(accepted,str) or not accepted.startswith("image-"):
                    raise WorkflowError("image_save_accepted_without_valid_id",unknown=True)
                report["accepted_image_uuid"]=accepted
                report["resume_command"]="image-wait --image-id "+accepted
                self.manager("image-wait","--image-id",accepted,"--wait-timeout",self.remaining())
            elif command == "image-wait":
                if not image_id:
                    raise WorkflowError("image_id_required")
                self.manager("image-wait", "--image-id", image_id, "--wait-timeout", self.remaining())
            else:
                raise WorkflowError("unknown_command")
            report["status"] = "pass"
        except WorkflowError as error:
            report.update(error=error.code, outcome_unknown=error.unknown)
            if error.detail is not None:
                report["failure_detail"] = error.detail
        except Exception:
            # No raw exception/headers/secret material in the handoff report.
            report["error"] = "local_configuration_or_output_failure"
        finally:
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary=None
                self.pinned_endpoint=None
        report["elapsed_seconds"] = round(self.clock()-self.started,6)
        report["automatic_shutdown"] = False
        report["automatic_new_instance"] = False
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("RTX5090.control.json"))
    parser.add_argument("--timeout-seconds",type=float,default=900)
    sub = parser.add_subparsers(dest="command",required=True)
    for command in ("status","test","images"):
        sub.add_parser(command)
    sub.add_parser("start").add_argument("--test",action="store_true",help="run all modality smoke tests after readiness")
    sub.add_parser("image-save").add_argument("--name",required=True)
    sub.add_parser("image-wait").add_argument("--image-id",required=True)
    args=parser.parse_args(argv)
    try:
        config=load_workflow(args.config)
        directory=config["config_file"].parent/"state"
        directory.mkdir(mode=0o700,exist_ok=True)
        if directory.is_symlink():
            raise WorkflowError("state_directory_symlink_rejected")
        lock_path=directory/"workflow.lock"
        fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
        with os.fdopen(fd,"r+") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise WorkflowError("workflow_lock_requires_regular_file")
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise WorkflowError("another_workflow_is_running_in_this_directory") from None
            result=Workflow(config,timeout=args.timeout_seconds).run(args.command,
                        test_after_start=getattr(args,"test",False),name=getattr(args,"name",None),
                        image_id=getattr(args,"image_id",None))
            atomic_json(directory/("last-"+args.command+".json"),result)
    except WorkflowError as error:
        result={"status":"fail","error":error.code,"outcome_unknown":error.unknown}
    except Exception:
        result={"status":"fail","error":"local_configuration_or_output_failure"}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0 if result["status"]=="pass" else 1


if __name__=="__main__":
    raise SystemExit(main())
