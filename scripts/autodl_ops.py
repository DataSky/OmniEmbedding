#!/usr/bin/env python3
"""固定 AutoDL 推理目录的本机运维；不安装、不下载、不操作 AutoDL 账户。

用法：.conda/bin/python scripts/autodl_ops.py [status|start|stop|restart|preflight]
默认 status。start/restart --offline-start 只设置离线环境，尚不代表冷启动已验收。
status 的退出码0表示探测完成且模型ready；未ready或拒绝危险操作时退出1。
start 的成功只表示监督进程已启动/启动已请求，仍需随后检查status和真实推理。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path('/root/omniembedding-service')
PROC = Path('/proc')
MODEL = 'google/embeddinggemma-2'
REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
MAX_HEALTH_BYTES = 64 * 1024


class OpsError(RuntimeError):
    """Only fixed local error codes, never sensitive exception messages."""


def require(test, code):
    if not test:
        raise OpsError(code)


def _regular(path, maximum=8192, *, private=False):
    """Read one bounded regular file without following a symlink or blocking on FIFO."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode), 'not_regular_file')
        if private:
            require(info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o600,
                    'private_file_owner_or_mode')
        raw = handle.read(maximum + 1)
    require(len(raw) <= maximum, 'file_too_large')
    return raw


def _key():
    value = _regular(ROOT / 'inference.key', private=True).decode('ascii').strip()
    require(32 <= len(value) <= 8192 and all(33 <= ord(x) <= 126 for x in value), 'invalid_service_key')
    return value


def _pid():
    try:
        raw = _regular(ROOT / 'run/supervisor.pid', 32).decode('ascii').strip()
    except FileNotFoundError:
        return None
    require(re.fullmatch(r'[1-9][0-9]{0,9}', raw) is not None and int(raw) > 1, 'invalid_pid_file')
    return int(raw)


def _start_ticks(pid):
    # Field22 is starttime; /proc/<pid>/stat comm can contain spaces or parentheses.
    rest = (PROC / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
    return int(rest[19])


def _lock_line_matches(line, pid, info):
    fields = line.split()
    try:
        i = fields.index('FLOCK')
        if fields[i + 1:i + 3] != ['ADVISORY', 'WRITE'] or int(fields[i + 3]) != pid:
            return False
        major, minor, inode = fields[i + 4].split(':')
        return (int(major, 16), int(minor, 16), int(inode)) == (
            os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
    except (ValueError, IndexError):
        return False


def _identity(pid):
    """A PID file alone is insufficient: require cwd, local Python and live FLOCK."""
    require(type(pid) is int and pid > 1, 'invalid_pid')
    try:
        beginning = _start_ticks(pid)
        directory = PROC / str(pid)
        cwd = (directory / 'cwd').resolve(strict=True)
        executable = (directory / 'exe').resolve(strict=True)
        require(cwd == ROOT, 'supervisor_cwd_mismatch')
        require(executable.parent == ROOT / '.conda/bin' and
                re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', executable.name),
                'supervisor_python_mismatch')
        lock = ROOT / 'run/supervisor.lock'
        info = lock.lstat()
        require(stat.S_ISREG(info.st_mode), 'supervisor_lock_not_regular')
        held = False
        for fd in (directory / 'fd').iterdir():
            try:
                opened = fd.stat()
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    continue
                fdinfo = (directory / 'fdinfo' / fd.name).read_text()
                if any(_lock_line_matches(line, pid, info) for line in fdinfo.splitlines()):
                    held = True
                    break
            except (OSError, ValueError):
                continue
        require(held, 'supervisor_does_not_hold_flock')
        require(_start_ticks(pid) == beginning, 'pid_reused_during_inspection')
        return {'pid': pid, 'start_ticks': beginning, 'cwd': str(cwd),
                'python': str(executable), 'holds_supervisor_flock': True}
    except (FileNotFoundError, ProcessLookupError):
        raise OpsError('supervisor_process_missing') from None
    except (ValueError, IndexError, PermissionError):
        raise OpsError('supervisor_identity_unreadable') from None


def _supervisor():
    pid = _pid()
    return _identity(pid) if pid is not None else None


def _listener_present():
    for family, address in ((socket.AF_INET, ('127.0.0.1', 6006)),
                            (socket.AF_INET6, ('::1', 6006))):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as stream:
                stream.settimeout(.25)
                if stream.connect_ex(address) == 0:
                    return True
        except OSError:
            continue
    return False


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        return None


def _health(path, key=None):
    headers = {'Accept': 'application/json'}
    if key is not None:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request('http://127.0.0.1:6006' + path, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        try:
            response = opener.open(request, timeout=2)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            code = response.code
            raw = response.read(MAX_HEALTH_BYTES + 1)
        require(len(raw) <= MAX_HEALTH_BYTES, 'health_response_too_large')
        value = json.loads(raw)
        require(isinstance(value, dict), 'invalid_health_json')
        result = {'http_status': code}
        for name in ('status', 'boot_id', 'model', 'model_revision', 'device', 'dtype'):
            label = value.get(name)
            if isinstance(label, str) and len(label) <= 128 and re.fullmatch(r'[A-Za-z0-9_./:-]+', label):
                result[name] = label
        if type(value.get('ready')) is bool:
            result['ready'] = value['ready']
        for name in ('uptime_s', 'busy_for_s'):
            number = value.get(name)
            if type(number) in (int, float) and 0 <= number < 1e12:
                result[name] = number
        result['identity_matches'] = value.get('model') == MODEL and value.get('model_revision') == REVISION
        return result
    except OpsError as error:
        return {'error': str(error), 'ready': False}
    except (OSError, ValueError, RecursionError):
        return {'error': 'health_unavailable_or_invalid', 'ready': False}


def status():
    result = {'ok': True, 'operation': 'status', 'root': str(ROOT), 'service_ready': False}
    try:
        result['supervisor'] = _supervisor()
    except (OpsError, OSError, UnicodeError):
        result['supervisor'] = None
        result['supervisor_warning'] = 'pid_missing_stale_or_identity_unverified'
    result['port_6006_listening'] = _listener_present()
    result['live'] = _health('/health/live')
    try:
        key = _key()
    except (OpsError, OSError, UnicodeError):
        result['ready'] = {'error': 'service_key_unavailable_or_invalid', 'ready': False}
    else:
        result['ready'] = _health('/health/ready', key)
    ready = result['ready']
    result['service_ready'] = bool(ready.get('http_status') == 200 and ready.get('ready') is True
                                  and ready.get('status') == 'ok' and ready.get('identity_matches'))
    return result


def _directory(name):
    path = ROOT / name
    path.mkdir(mode=0o700, exist_ok=True)
    require(path.is_dir() and not path.is_symlink() and path.resolve() == path,
            'unsafe_runtime_directory')
    require(path.stat().st_uid == os.geteuid(), 'runtime_directory_not_owned')
    return path


def _launch_env(offline):
    environment = os.environ.copy()
    if offline:
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMNI_AUTODL_NETWORK_TURBO='0')
    return environment


def start(*, offline=False, wait_s=5):
    _directory('run')
    _directory('logs')
    warning = None
    try:
        previous = _supervisor()
    except (OpsError, OSError, UnicodeError):
        previous = None
        warning = 'stale_or_unverified_pid_file_ignored_only_for_start'
    if previous:
        return {'ok': True, 'operation': 'start', 'already_running': True,
                'supervisor_pid': previous['pid'], 'readiness_pending': True,
                'offline_start_requested': offline, 'offline_applied_to_existing_process': False}
    require(not _listener_present(), 'port_6006_occupied_without_verified_supervisor')
    # Probe the actual lock before launching; never unlink a lock held by another process.
    lockfd = os.open(ROOT / 'run/supervisor.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(lockfd).st_mode), 'supervisor_lock_not_regular')
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OpsError('supervisor_lock_busy_identity_unknown') from None
    finally:
        os.close(lockfd)
    _key()  # Validate without printing or passing the secret on a command line.
    runner = ROOT / 'run-autodl.sh'
    require(runner.is_file() and not runner.is_symlink(), 'existing_runner_required_no_bootstrap')
    python = ROOT / '.conda/bin/python'
    require(python.is_file() and os.access(python, os.X_OK), 'existing_python_required_no_install')
    logfd = os.open(ROOT / 'logs/supervisor.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(logfd, 'ab', buffering=0) as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'supervisor_log_not_regular')
        child = subprocess.Popen(['/bin/bash', str(runner)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                 close_fds=True, env=_launch_env(offline))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            current = _supervisor()
        except (OpsError, OSError, UnicodeError):
            current = None
        if current:
            return {'ok': True, 'operation': 'start', 'supervisor_pid': current['pid'],
                    'readiness_pending': True, 'offline_start_requested': offline,
                    'offline_cold_start_verified': False, 'warning': warning}
        require(child.poll() is None, 'runner_exited_before_supervisor_verified')
        time.sleep(.1)
    return {'ok': True, 'operation': 'start', 'launch_requested_pid': child.pid,
            'supervisor_identity_pending': True, 'readiness_pending': True,
            'offline_start_requested': offline, 'offline_cold_start_verified': False, 'warning': warning}


def stop(*, timeout_s=25):
    pid = _pid()
    if pid is None:
        require(not _listener_present(), 'port_6006_active_no_verified_supervisor_refuse_stop')
        return {'ok': True, 'operation': 'stop', 'already_stopped': True,
                'autodl_instance_billing_stopped': False}
    identity = _identity(pid)  # Any stale/unrelated PID is rejected, never signalled.
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'),
            'safe_pidfd_signalling_unavailable')
    try:
        descriptor = os.pidfd_open(pid, 0)
    except OSError:
        raise OpsError('safe_pidfd_open_failed_no_signal_sent') from None
    try:
        current = _identity(pid)
        require(current == identity, 'pid_identity_changed_no_signal_sent')
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        exited = bool(poller.poll(max(1, int(timeout_s * 1000))))
    except ProcessLookupError:
        exited = True
    finally:
        os.close(descriptor)
    require(exited, 'supervisor_exit_timeout_no_force_kill')
    # Do not kill an orphan/foreign listener; report it and refuse restart instead.
    require(not _listener_present(), 'port_6006_still_active_after_supervisor_exit')
    return {'ok': True, 'operation': 'stop', 'stopped_supervisor_pid': pid,
            'signal': 'SIGTERM', 'autodl_instance_billing_stopped': False}


def _json_file(path, maximum=512 * 1024):
    base = (ROOT / 'cache/huggingface/hub/models--google--embeddinggemma-2').resolve(strict=True)
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(base), 'cache_json_outside_model_cache')
    descriptor = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'cache_json_not_regular')
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, 'cache_json_too_large')
    value = json.loads(raw)
    return value


def cache_preflight():
    base = ROOT / 'cache/huggingface/hub/models--google--embeddinggemma-2'
    snapshot = base / 'snapshots' / REVISION
    result = {'revision': REVISION, 'snapshot': str(snapshot), 'snapshot_present': snapshot.is_dir(),
              'offline_cold_start_verified': False, 'broken_or_unsafe_paths': [], 'required_files_missing': []}
    if not result['snapshot_present']:
        result['ok'] = False
        return result
    base_real = base.resolve(strict=True)
    files = []
    for path in snapshot.rglob('*'):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = str(path.relative_to(snapshot))
        try:
            resolved = path.resolve(strict=True)
            require(resolved.is_relative_to(base_real), 'cache_target_outside_model_cache')
            require(resolved.is_file() and resolved.stat().st_size > 0, 'cache_file_empty_or_not_regular')
            files.append(path)
        except (OSError, RuntimeError, OpsError):
            result['broken_or_unsafe_paths'].append(relative)
    for name in ('config.json', 'modules.json', 'tokenizer.json', 'tokenizer_config.json'):
        if not (snapshot / name).is_file():
            result['required_files_missing'].append(name)
    if result['broken_or_unsafe_paths'] or result['required_files_missing']:
        result['ok'] = False
        result['broken_or_unsafe_paths'] = result['broken_or_unsafe_paths'][:20]
        return result
    modules_ok = False
    try:
        modules = _json_file(snapshot / 'modules.json')
        require(isinstance(modules, list) and modules, 'invalid_modules_json')
        for module in modules:
            require(isinstance(module, dict) and isinstance(module.get('path'), str), 'invalid_module_entry')
            name = module['path']
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe_module_path')
            kind = module.get('type')
            require(isinstance(kind, str), 'invalid_module_type')
            # Normalize has no persisted files; an empty directory is not stored by HF/Git.
            if not kind.endswith('.Normalize'):
                require((snapshot / name).is_dir(), 'missing_module_directory')
            if kind.endswith(('.Pooling', '.Dense')):
                require((snapshot / name / 'config.json').is_file(), 'missing_module_config')
        modules_ok = True
    except (OSError, ValueError, OpsError, TypeError, RecursionError):
        result['module_configuration_error'] = 'modules_or_required_module_config_invalid'
    weights = []
    try:
        index = snapshot / 'model.safetensors.index.json'
        if index.is_file():
            mapping = _json_file(index, 2 * 1024 * 1024).get('weight_map')
            require(isinstance(mapping, dict) and mapping, 'invalid_weight_index')
            for name in set(mapping.values()):
                require(isinstance(name, str) and name.endswith('.safetensors')
                        and not Path(name).is_absolute() and '..' not in Path(name).parts,
                        'unsafe_weight_shard')
                path = snapshot / name
                require(path.is_file() and path.stat().st_size > 0, 'missing_weight_shard')
                weights.append(path)
        else:
            path = snapshot / 'model.safetensors'
            require(path.is_file() and path.stat().st_size > 0, 'missing_native_weights')
            weights.append(path)
    except (OSError, ValueError, OpsError, TypeError, AttributeError, RecursionError):
        result['weight_configuration_error'] = 'native_weights_missing_or_invalid'
    result.update({'file_count': len(files), 'native_weight_file_count': len(weights),
                   'native_weight_bytes': sum(p.stat().st_size for p in weights),
                   'module_configuration_valid': modules_ok})
    result['ok'] = not (result['broken_or_unsafe_paths'] or result['required_files_missing']) and modules_ok and bool(weights) and 'weight_configuration_error' not in result
    result['broken_or_unsafe_paths'] = result['broken_or_unsafe_paths'][:20]
    result['note'] = 'Static local cache/link check only; does not prove all runtime loader files or offline cold startup.'
    return result


LIBRARY_CHECK = r'''
import importlib, importlib.metadata, json
result={'ok':True,'python':None,'libraries':{},'cuda_available':False,'cuda_wheel':None}
import sys
result['python']='.'.join(map(str,sys.version_info[:3]))
for name,package in [('torch','torch'),('torchvision','torchvision'),('numpy','numpy'),('scipy','scipy'),('transformers','transformers'),('sentence_transformers','sentence-transformers'),('fastapi','fastapi'),('uvicorn','uvicorn')]:
    try:
        module=importlib.import_module(name)
        result['libraries'][package]={'import_ok':True,'version':importlib.metadata.version(package)}
    except Exception:
        result['libraries'][package]={'import_ok':False}
        result['ok']=False
try:
    import torch
    result['cuda_available']=bool(torch.cuda.is_available())
    result['cuda_wheel']=str(torch.version.cuda)
except Exception:
    result['ok']=False
pins={'torch':'2.11.0+cu128','torchvision':'0.26.0+cu128','numpy':'2.5.2','scipy':'1.18.1'}
result['fixed_pins_match']=all(result['libraries'].get(k,{}).get('version')==v for k,v in pins.items())
result['ok']=result['ok'] and result['cuda_available'] and result['cuda_wheel']=='12.8' and sys.version_info[:2]==(3,12) and result['fixed_pins_match']
print('OMNI_OPS_RESULT:'+json.dumps(result,allow_nan=False))
'''


def preflight(*, pip_check=False):
    python = ROOT / '.conda/bin/python'
    result = {'ok': False, 'operation': 'preflight', 'no_install_or_download': True,
              'runner_present': (ROOT / 'run-autodl.sh').is_file(), 'offline_cold_start_verified': False}
    try:
        _key()
        result['service_key_valid'] = True
    except (OpsError, OSError, UnicodeError):
        result['service_key_valid'] = False
    result['cache'] = cache_preflight()
    require(python.is_file() and os.access(python, os.X_OK), 'existing_python_required_no_install')
    env = _launch_env(True)
    try:
        check = subprocess.run([str(python), '-c', LIBRARY_CHECK], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=60, check=False)
        lines = [x.removeprefix('OMNI_OPS_RESULT:') for x in check.stdout.splitlines() if x.startswith('OMNI_OPS_RESULT:')]
        require(check.returncode == 0 and len(lines) == 1, 'library_check_failed')
        result['environment'] = json.loads(lines[0])
    except (OSError, ValueError, OpsError, subprocess.TimeoutExpired):
        result['environment'] = {'ok': False, 'error': 'library_or_cuda_check_failed'}
    if pip_check:
        try:
            check = subprocess.run([str(python), '-m', 'pip', 'check'], cwd=ROOT, env=env,
                                   capture_output=True, timeout=60, check=False)
            result['pip_check'] = {'ok': check.returncode == 0, 'output_omitted': True}
        except (OSError, subprocess.TimeoutExpired):
            result['pip_check'] = {'ok': False, 'error': 'pip_check_failed_or_timeout'}
    result['ok'] = bool(result['runner_present'] and result['service_key_valid'] and result['cache']['ok']
                        and result['environment'].get('ok') and result.get('pip_check', {'ok': True})['ok'])
    return result


def runtime_guard():
    require(sys.platform.startswith('linux'), 'linux_only')
    require(os.geteuid() == 0, 'run_as_root_on_autodl')
    require(ROOT.is_dir() and not ROOT.is_symlink() and ROOT.resolve(strict=True) == ROOT,
            'fixed_service_root_required')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', default='status', choices=('status','start','stop','restart','preflight'))
    parser.add_argument('--offline-start', action='store_true', help='仅start/restart：设置离线变量；不宣称已通过离线冷启动')
    parser.add_argument('--pip-check', action='store_true', help='仅preflight：额外运行只读pip check')
    parser.add_argument('--stop-timeout', type=float, default=25, help='SIGTERM后等待秒数，1–60；超时不强杀')
    args = parser.parse_args(argv)
    if args.offline_start and args.command not in ('start','restart'):
        parser.error('--offline-start只适用于start/restart')
    if args.pip_check and args.command != 'preflight':
        parser.error('--pip-check只适用于preflight')
    if not 1 <= args.stop_timeout <= 60:
        parser.error('--stop-timeout须为1–60秒')
    try:
        runtime_guard()
        if args.command == 'status':
            result = status()
        elif args.command == 'start':
            result = start(offline=args.offline_start)
        elif args.command == 'stop':
            result = stop(timeout_s=args.stop_timeout)
        elif args.command == 'restart':
            stopped = stop(timeout_s=args.stop_timeout)
            result = start(offline=args.offline_start)
            result.update(operation='restart', stop_result=stopped)
        else:
            result = preflight(pip_check=args.pip_check)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result.get('ok') and (args.command != 'status' or result.get('service_ready')) else 1
    except OpsError as error:
        output = {'ok': False, 'operation': args.command, 'error': str(error)}
    except Exception as error:
        output = {'ok': False, 'operation': args.command, 'error': 'operation_failed', 'error_class': type(error).__name__}
    print(json.dumps(output, ensure_ascii=False))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
