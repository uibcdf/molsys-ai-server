#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_CONFIG="${MODEL_CONFIG:-server/model_server/config.yaml}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
BACKGROUND="${BACKGROUND:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/scripts/_logs}"
PID_DIR="${PID_DIR:-${ROOT_DIR}/scripts/_pids}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/molsys-ai-model.log}"
PID_FILE="${PID_FILE:-${PID_DIR}/molsys-ai-model.pid}"

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

if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Missing model config: ${MODEL_CONFIG}" >&2
  echo "Tip: start from server/model_server/config.example.yaml" >&2
  exit 1
fi

cmd=(./dev/run_model_server.sh --config "${MODEL_CONFIG}" --cuda-devices "${CUDA_DEVICES}" --warmup)

if [[ "${BACKGROUND}" == "1" ]]; then
  mkdir -p "${LOG_DIR}" "${PID_DIR}"
  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" >/dev/null 2>&1; then
      echo "Model server already running (PID ${old_pid})." >&2
      echo "Log: ${LOG_FILE}" >&2
      exit 1
    fi
  fi
  nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "Model server started (PID ${pid})."
  echo "Log: ${LOG_FILE}"
  echo "PID file: ${PID_FILE}"
  exit 0
fi

exec "${cmd[@]}"
