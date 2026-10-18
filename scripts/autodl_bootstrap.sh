#!/usr/bin/env bash
# Run after extracting the source whitelist into /root/omniembedding-service.
# No AutoDL account credential is needed or accessed by this script.
set -euo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$(uname -s)" == Linux && "$ROOT" == /root/omniembedding-service ]] || {
  echo 'Run this script on AutoDL in /root/omniembedding-service.' >&2; exit 1;
}
cd "$ROOT"
for source in requirements-autodl.txt requirements-colab.txt omniembed/{__init__,config,health,encoder,inference}.py; do
  [[ -f "$source" ]] || { echo "Missing source file: $source" >&2; exit 1; }
done
[[ ! -e .env ]] || { echo 'Remove the uploaded .env; service settings are defined here.' >&2; exit 1; }
[[ ! -L inference.key ]] || { echo 'inference.key must not be a symbolic link.' >&2; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi -L
free_kib="$(df -Pk "$ROOT" | awk 'END {print $4}')"
(( free_kib >= 16 * 1024 * 1024 )) || {
  echo 'At least 16 GiB free system-disk space is required for installation and model weights.' >&2; exit 1;
}
CONDA_BIN=""
for candidate in "${CONDA_EXE:-}" "$(command -v conda || true)" /root/miniconda3/bin/conda /opt/conda/bin/conda; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
done
[[ -n "$CONDA_BIN" ]] || { echo 'Conda was not found; select an AutoDL Conda image.' >&2; exit 1; }
mkdir -p logs run cache/conda cache/huggingface tmp
export CONDA_PKGS_DIRS="$ROOT/cache/conda" TMPDIR="$ROOT/tmp"
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
if [[ ! -x .conda/bin/python ]]; then
  "$CONDA_BIN" create --yes --prefix "$ROOT/.conda" --override-channels --channel conda-forge python=3.12 pip
fi
PYTHON="$ROOT/.conda/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
# Official matched wheels: https://pytorch.org/get-started/previous-versions/#v2110
"$PYTHON" -m pip install --no-cache-dir --no-deps torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -m pip install --no-cache-dir -r requirements-autodl.txt \
  --index-url "${OMNI_AUTODL_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
"$PYTHON" -m pip check
"$PYTHON" - <<'PYTHON'
import os
from pathlib import Path
import secrets
import torch
assert torch.cuda.is_available(), "CUDA is unavailable; check the host NVIDIA driver."
assert torch.version.cuda == "12.8", "Expected the CUDA 12.8 PyTorch wheel."
key = Path("inference.key")
if not key.exists():
    with os.fdopen(os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
        output.write(secrets.token_urlsafe(48) + "\n")
key.chmod(0o600)
secret = key.read_text().strip()
assert len(secret) >= 32 and not any(c.isspace() for c in secret), "Invalid inference.key"
print("CUDA verified:", torch.cuda.get_device_name(0), "; service key saved without displaying it.")
PYTHON
cat > run-autodl.sh <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Keep AutoDL's external-network proxy out of the Conda/PyPI installation steps.
if [[ "${OMNI_AUTODL_NETWORK_TURBO:-1}" != 0 && -r /etc/network_turbo ]]; then
  set +u
  source /etc/network_turbo >/dev/null 2>&1 || {
    set -u
    echo 'AutoDL network acceleration failed to initialize.' >&2
    exit 1
  }
  set -u
fi
exec .conda/bin/python - <<'PYTHON'
import collections
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

root = Path.cwd()
lock = (root / "run/supervisor.lock").open("w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit("The service supervisor is already running; readiness still needs checking.")
pid_file = root / "run/supervisor.pid"
pid_file.write_text(str(os.getpid()) + "\n")
logger = logging.getLogger("autodl-service")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(root / "logs/inference.log", maxBytes=10 * 1024**2, backupCount=2)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)
secret_path = root / "inference.key"
if secret_path.is_symlink():
    sys.exit("inference.key must not be a symbolic link")
secret_path.chmod(0o600)
secret = secret_path.read_text().strip()
if len(secret) < 32 or any(c.isspace() for c in secret):
    sys.exit("Invalid inference.key")
environment = dict(os.environ, OMNI_INFERENCE_API_KEY=secret,
    OMNI_MODEL="google/embeddinggemma-2",
    OMNI_MODEL_REVISION="914f7f89142e33e77833254d9c9b90c3cef7303b",
    OMNI_DEVICE="cuda", OMNI_DTYPE="auto", OMNI_ENCODER_BACKEND="local",
    OMNI_MODEL_STARTUP_TIMEOUT_S="900", OMNI_INFERENCE_TIMEOUT_S="120",
    HF_HOME=str(root / "cache/huggingface"), HF_HUB_DISABLE_XET="1", PYTHONUNBUFFERED="1")
stop = threading.Event()
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: stop.set())
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
restarts = collections.deque()

def copy_logs(process):
    for line in process.stdout:
        logger.info("%s", line.rstrip())

try:
    while not stop.is_set():
        process = subprocess.Popen([sys.executable, "-m", "uvicorn",
            "omniembed.inference:create_app", "--factory", "--host", "0.0.0.0",
            "--port", "6006", "--workers", "1", "--no-access-log"],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        reader = threading.Thread(target=copy_logs, args=(process,), daemon=True)
        reader.start()
        started, failures, reason = time.monotonic(), 0, "process exited"
        logger.info("Started inference PID %s; model readiness is pending", process.pid)
        try:
            while not stop.wait(10):
                if process.poll() is not None:
                    break
                request = urllib.request.Request("http://127.0.0.1:6006/health/ready",
                    headers={"Authorization": "Bearer " + secret})
                try:
                    try:
                        response = opener.open(request, timeout=5)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        status = json.load(response).get("status")
                    failures = 0
                    if status in {"stalled", "degraded"}:
                        reason = "model " + status
                        break
                    if status == "loading" and time.monotonic() - started > 930:
                        reason = "model startup timeout"
                        break
                except (OSError, ValueError):
                    failures += 1
                    if failures >= 3:
                        reason = "health endpoint unresponsive"
                        break
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reader.join(timeout=2)
        if stop.is_set():
            break
        now = time.monotonic()
        while restarts and restarts[0] < now - 900:
            restarts.popleft()
        restarts.append(now)
        logger.error("Restart required: %s", reason)
        if len(restarts) >= 5:
            logger.error("Five failures within 15 minutes; supervisor stopped. Inspect the log before restarting.")
            sys.exit(1)
        stop.wait(min(60, 2 ** len(restarts)))
finally:
    pid_file.unlink(missing_ok=True)
    logger.info("Supervisor stopped")
PYTHON
RUNNER
chmod 700 run-autodl.sh
touch logs/supervisor.log
chmod 600 logs/supervisor.log
nohup bash "$ROOT/run-autodl.sh" >> "$ROOT/logs/supervisor.log" 2>&1 < /dev/null &
echo "Supervisor launch requested (PID $!). Model readiness has NOT been verified."
echo "Check authenticated /health/ready and /health/inference before using the service."
echo "Logs: $ROOT/logs/inference.log ; restart: bash $ROOT/run-autodl.sh"
echo "Stop: kill \$(cat $ROOT/run/supervisor.pid) ; this stops the service, not AutoDL billing."
