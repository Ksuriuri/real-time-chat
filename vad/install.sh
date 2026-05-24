#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
PRELOAD_MODEL="${VAD_PRELOAD_MODEL:-true}"

echo "Creating VAD virtual environment with Python ${PYTHON_VERSION}..."
uv venv --python "${PYTHON_VERSION}" --seed

echo "Installing VAD dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Checking VAD service imports..."
uv run python -c "import app; print('VAD_IMPORT_OK')"

if [[ "${PRELOAD_MODEL}" != "0" && "${PRELOAD_MODEL}" != "false" && "${PRELOAD_MODEL}" != "no" ]]; then
  echo "Loading Silero VAD ONNX model..."
  uv run python -c "from app import get_model; get_model(); print('VAD_MODEL_READY')"
else
  echo "Skipping model preload because VAD_PRELOAD_MODEL=${PRELOAD_MODEL}."
fi

echo "VAD install complete."
