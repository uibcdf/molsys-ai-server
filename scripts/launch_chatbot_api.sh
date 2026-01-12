#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MOLSYS_AI_ENGINE_URL="${MOLSYS_AI_ENGINE_URL:-http://127.0.0.1:8001}"
MOLSYS_AI_PROJECT_INDEX_DIR="${MOLSYS_AI_PROJECT_INDEX_DIR:-server/chat_api/data/indexes}"
MOLSYS_AI_EMBEDDINGS="${MOLSYS_AI_EMBEDDINGS:-sentence-transformers}"
MOLSYS_AI_EMBEDDINGS_DEVICE="${MOLSYS_AI_EMBEDDINGS_DEVICE:-cpu}"
MOLSYS_AI_CORS_ORIGINS="${MOLSYS_AI_CORS_ORIGINS:-https://www.uibcdf.org}"
BACKGROUND="${BACKGROUND:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/scripts/_logs}"
PID_DIR="${PID_DIR:-${ROOT_DIR}/scripts/_pids}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/molsys-ai-chat-api.log}"
PID_FILE="${PID_FILE:-${PID_DIR}/molsys-ai-chat-api.pid}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --daemon|--background)
      BACKGROUND="1"
      shift 1
      ;;
    --foreground)
      BACKGROUND="0"
      shift 1
      ;;
    --log)
      LOG_FILE="${2:-}"
      shift 2
      ;;
    --pid)
      PID_FILE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export MOLSYS_AI_ENGINE_URL
export MOLSYS_AI_PROJECT_INDEX_DIR
export MOLSYS_AI_EMBEDDINGS
export MOLSYS_AI_EMBEDDINGS_DEVICE
export MOLSYS_AI_CORS_ORIGINS

cmd=(./dev/run_chat_api.sh --host 127.0.0.1 --port 8000)

if [[ "${BACKGROUND}" == "1" ]]; then
  mkdir -p "${LOG_DIR}" "${PID_DIR}"
  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" >/dev/null 2>&1; then
      echo "Chat API already running (PID ${old_pid})." >&2
      echo "Log: ${LOG_FILE}" >&2
      exit 1
    fi
  fi
  nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Chat API started (PID ${pid})."
  echo "Log: ${LOG_FILE}"
  echo "PID file: ${PID_FILE}"
  exit 0
fi

exec "${cmd[@]}"
