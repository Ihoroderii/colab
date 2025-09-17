#!/usr/bin/env bash
# Simple runner to export env vars, ensure deps, and run the manhwa generator.
# Usage examples:
#   ./run.sh --panels 8 --num-steps 25 --export
#   HF_TOKEN=xxx DEVICE=cpu ./run.sh --no-save-panels

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1) Load .env if present (non-fatal if missing)
if [[ -f .env ]]; then
  echo "Loading .env"
  set +u
  # shellcheck disable=SC1090
  source .env
  set -u
fi

# 2) Pick a virtual environment (.venv or existing menv), create if absent
VENV_DIR=${VENV_DIR:-.venv}
if [[ ! -d "$VENV_DIR" ]]; then
  if [[ -d menv ]]; then
    VENV_DIR=menv
  else
    echo "Creating virtual environment in $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
fi

# 3) Activate the environment
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
else
  echo "Warning: Could not find $VENV_DIR/bin/activate; continuing without activation"
fi

# 4) Ensure pip and install project (editable if packaging present)
python -m pip install --upgrade pip >/dev/null
if [[ -f setup.cfg || -f pyproject.toml ]]; then
  echo "Installing package (editable)"
  pip install -e .
else
  echo "Installing requirements"
  pip install -r requirements.txt
fi

# 5) If package import not available, add src/ to PYTHONPATH
if ! python -c "import importlib; importlib.import_module('manga_ai')" >/dev/null 2>&1; then
  export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
fi

# 6) Select run command: prefer console script, then module, fallback to test10.py
if command -v manga-ai >/dev/null 2>&1; then
  RUN_CMD=(manga-ai)
elif python -c "import importlib; importlib.import_module('manga_ai')" >/dev/null 2>&1; then
  RUN_CMD=(python -m manga_ai)
else
  echo "Package not importable; falling back to test10.py"
  RUN_CMD=(python test10.py)
fi

echo "Running: ${RUN_CMD[*]} $*"
"${RUN_CMD[@]}" "$@"
