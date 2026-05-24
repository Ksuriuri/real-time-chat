#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.13}"

echo "Creating LLM virtual environment with Python ${PYTHON_VERSION}..."
uv venv --python "${PYTHON_VERSION}" --seed

echo "Installing LLM dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Checking LLM imports..."
uv run python -c "from ark_language_model import ArkLanguageModel; print('LLM_IMPORT_OK')"

echo "LLM install complete."
