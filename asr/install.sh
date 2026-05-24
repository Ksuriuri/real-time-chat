#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
DOWNLOAD_MODEL="${ASR_DOWNLOAD_MODEL:-true}"

echo "Creating ASR virtual environment with Python ${PYTHON_VERSION}..."
uv venv --python "${PYTHON_VERSION}" --seed

echo "Installing ASR dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Checking ASR service imports..."
uv run python -c "import app; print('ASR_IMPORT_OK')"

if [[ "${DOWNLOAD_MODEL}" != "0" && "${DOWNLOAD_MODEL}" != "false" && "${DOWNLOAD_MODEL}" != "no" ]]; then
  echo "Downloading and preloading ASR model if needed..."
  uv run python -c "from app import get_model; get_model(); print('ASR_MODEL_READY')"
else
  echo "Skipping model download because ASR_DOWNLOAD_MODEL=${DOWNLOAD_MODEL}."
fi

echo "ASR install complete."
