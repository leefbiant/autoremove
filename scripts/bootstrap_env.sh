#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-venv}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"
INCLUDE_OPTIONAL_MODELS="${INCLUDE_OPTIONAL_MODELS:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN"
  exit 1
fi

echo "[1/4] Creating virtualenv at $VENV_DIR ..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "[2/4] Activating virtualenv ..."
source "$VENV_DIR/bin/activate"

echo "[3/4] Installing python packages from requirements.txt ..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  echo "[4/4] Downloading models from models/manifest.json ..."
  if [[ "$INCLUDE_OPTIONAL_MODELS" == "1" ]]; then
    python scripts/download_models.py --include-optional
  else
    python scripts/download_models.py
  fi
else
  echo "[4/4] Skipped model download. Set DOWNLOAD_MODELS=1 to enable."
fi

echo "Environment ready."
echo "Run web: ./run.sh web 8788"
echo "Run cli: ./run.sh cli --input 2.png --output 2_out.png"

