#!/usr/bin/env bash
# Install only on request; normal start never downloads a model or reinstalls.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
ACTION="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON="$ROOT/data/local-demo/runtime/bin/python"
if [[ "$ACTION" == install ]]; then
  SETUP_PYTHON="${OMNI_DEMO_SETUP_PYTHON:-python3}"
  "$SETUP_PYTHON" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "Use Python 3.11 or 3.12; set OMNI_DEMO_SETUP_PYTHON to its path"'
  if [[ ! -x "$PYTHON" ]]; then "$SETUP_PYTHON" -m venv "$ROOT/data/local-demo/runtime"; fi
  REQUIREMENTS="$ROOT/requirements-demo.lock.txt"
  [[ -f "$REQUIREMENTS" ]] || REQUIREMENTS="$ROOT/requirements-demo.txt"
  "$PYTHON" -m pip install -r "$REQUIREMENTS" \
    --index-url "${OMNI_DEMO_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" --timeout 30
  exit 0
fi
[[ -x "$PYTHON" ]] || { echo 'Run: bash scripts/local_demo.sh install' >&2; exit 1; }
exec "$PYTHON" "$ROOT/scripts/run_local_demo.py" "$ACTION" "$@"
