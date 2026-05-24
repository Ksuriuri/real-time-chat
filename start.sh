#!/usr/bin/env bash
# One-click launcher: ensure venvs exist, start the local VAD service, then
# run the chat orchestrator in the foreground. Ctrl+C cleans up VAD.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the VAD health check." >&2
  exit 1
fi

# Single source of truth: <repo-root>/.env.
if [[ ! -f "$ROOT/.env" ]]; then
  cat >&2 <<EOF
No .env file found at $ROOT/.env.

  cp .env.example .env
  \$EDITOR .env

Required keys: ARK_API_KEY, VOLCENGINE_ASR_API_KEY, VOLCENGINE_TTS_API_KEY.
EOF
  exit 1
fi

# Export every key in .env so child processes (VAD uvicorn, chat cli) inherit
# them without each script needing its own dotenv loader.
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

ensure_venv() {
  local module="$1"
  if [[ ! -d "$ROOT/$module/.venv" ]]; then
    echo ">> Installing $module venv..."
    (cd "$ROOT/$module" && ./install.sh)
  fi
}

ensure_venv vad
ensure_venv chat

LOG_DIR="$ROOT/.cache"
mkdir -p "$LOG_DIR"
VAD_LOG="$LOG_DIR/vad.log"
VAD_PID_FILE="$LOG_DIR/vad.pid"

VAD_HOST="${VAD_HOST:-127.0.0.1}"
VAD_PORT="${VAD_PORT:-8010}"

cleanup() {
  local exit_code=$?
  if [[ -f "$VAD_PID_FILE" ]]; then
    local pid
    pid="$(cat "$VAD_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo
      echo ">> Stopping VAD service (pid $pid)..."
      kill "$pid" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$VAD_PID_FILE"
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo ">> Starting VAD service on http://${VAD_HOST}:${VAD_PORT} (logs: $VAD_LOG)"
(
  cd "$ROOT/vad"
  exec uv run uvicorn app:app --host "$VAD_HOST" --port "$VAD_PORT" >>"$VAD_LOG" 2>&1
) &
VAD_PID=$!
echo "$VAD_PID" > "$VAD_PID_FILE"

echo -n ">> Waiting for VAD health"
ready=0
for _ in $(seq 1 60); do
  if curl -sf "http://${VAD_HOST}:${VAD_PORT}/health" >/dev/null 2>&1; then
    ready=1
    echo " ... ready."
    break
  fi
  if ! kill -0 "$VAD_PID" 2>/dev/null; then
    echo
    echo ">> VAD service exited before becoming ready. Tail of $VAD_LOG:" >&2
    tail -n 40 "$VAD_LOG" >&2 || true
    exit 1
  fi
  echo -n "."
  sleep 1
done

if [[ $ready -ne 1 ]]; then
  echo
  echo ">> VAD service did not become ready within 60s. Tail of $VAD_LOG:" >&2
  tail -n 40 "$VAD_LOG" >&2 || true
  exit 1
fi

echo ">> Starting chat orchestrator. Speak into the mic; Ctrl+C to quit."
cd "$ROOT/chat"
exec uv run python cli.py "$@"
