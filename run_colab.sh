#!/usr/bin/env bash
# Minimal runner for Colab/remote notebooks: no venv, no pip installs.
# If invoked with `sh`, re-exec under bash to support arrays and pipefail.
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# Usage:
#   bash run_colab.sh --panels 8 --num-steps 25 --export

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
if [[ -f .env ]]; then
  echo "Loading .env"
  set +u
  # shellcheck disable=SC1090
  source .env
  set -u
fi

# Ensure src/ is on PYTHONPATH so `python -m manga_ai` works without install
export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"

# Prefer module entry; fallback to test10.py
if python -c "import importlib; importlib.import_module('manga_ai')" >/dev/null 2>&1; then
  RUN_CMD=(python -m manga_ai)
else
  RUN_CMD=(python test10.py)
fi

echo "Running: ${RUN_CMD[*]} $*"
"${RUN_CMD[@]}" "$@"
