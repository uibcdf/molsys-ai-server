#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the MolSys-AI model engine server (vLLM) locally.

This starts `server/model_server/server.py` via uvicorn and serves:
- POST /v1/engine/chat

Usage:
  ./dev/run_model_server.sh --config /path/to/model_server.yaml [options]

Optional env vars:
  HOST                 (default: 127.0.0.1)
  PORT                 (default: 8001)
  CUDA_VISIBLE_DEVICES (default: unset)
  CUDA_HOME            (default: /usr/local/cuda)
  RELOAD               (default: 0)

Options:
  --config PATH        Required (or set MOLSYS_AI_MODEL_CONFIG).
  --host HOST          Override host (default: 127.0.0.1).
  --port PORT          Override port (default: 8001).
  --cuda-devices LIST  Sets CUDA_VISIBLE_DEVICES (e.g. "0" or "0,1,2").
  --cuda-home PATH     Override CUDA_HOME (default: /usr/local/cuda).
  --reload             Enable uvicorn --reload (dev only).
  --warmup             After startup, send a small warmup request to reduce first-user latency.

Example:
  ./dev/run_model_server.sh --config /tmp/molsys_ai_vllm.yaml --cuda-devices 0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/server:${ROOT_DIR}/client:${PYTHONPATH:-}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
RELOAD="${RELOAD:-0}"
WARMUP="${WARMUP:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      MOLSYS_AI_MODEL_CONFIG="${2:-}"
      shift 2
      ;;
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --cuda-devices)
      CUDA_VISIBLE_DEVICES="${2:-}"
      shift 2
      ;;
    --cuda-home)
      CUDA_HOME="${2:-}"
      shift 2
      ;;
    --reload)
      RELOAD="1"
      shift 1
      ;;
    --warmup)
      WARMUP="1"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MOLSYS_AI_MODEL_CONFIG:-}" ]]; then
  echo "Missing model config. Provide --config PATH or set MOLSYS_AI_MODEL_CONFIG." >&2
  echo "Tip: see server/model_server/config.example.yaml" >&2
  exit 1
fi

# Make nvcc visible for vLLM FlashInfer JIT (if installed system-wide).
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

echo "[model_server] starting on http://${HOST}:${PORT}"
echo "[model_server] OpenAPI: http://${HOST}:${PORT}/docs"
echo "[model_server] Health:  curl -fsS http://${HOST}:${PORT}/healthz"

args=(model_server.server:app --host "${HOST}" --port "${PORT}")
if [[ "${RELOAD}" == "1" ]]; then
  args+=(--reload)
fi

if [[ "${WARMUP}" == "1" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "Missing command: curl (required for --warmup)" >&2
    exit 1
  fi

  cleanup() {
    if [[ -n "${UVICORN_PID:-}" ]]; then
      kill -TERM "${UVICORN_PID}" >/dev/null 2>&1 || true
      sleep 1
      kill -KILL "${UVICORN_PID}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT

  python -m uvicorn "${args[@]}" &
  UVICORN_PID=$!

  echo "[model_server] waiting for readiness..."
  for _ in $(seq 1 240); do
    if curl -fsS "http://${HOST}:${PORT}/docs" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  curl -fsS "http://${HOST}:${PORT}/docs" >/dev/null 2>&1 || { echo "[model_server] did not become ready" >&2; exit 1; }

  echo "[model_server] warmup request..."
  # If the engine endpoint is protected, set MOLSYS_AI_ENGINE_API_KEY in your environment.
  API_HEADER=()
  if [[ -n "${MOLSYS_AI_ENGINE_API_KEY:-}" ]]; then
    API_HEADER=(-H "Authorization: Bearer ${MOLSYS_AI_ENGINE_API_KEY}")
  fi
  curl -fsS -X POST "http://${HOST}:${PORT}/v1/engine/chat" \
    -H 'Content-Type: application/json' \
    "${API_HEADER[@]}" \
    -d '{"messages":[{"role":"user","content":"warmup: reply only OK"}]}' >/dev/null
  echo "[model_server] warmup done"

  trap - EXIT
  wait "${UVICORN_PID}"
else
  python -m uvicorn "${args[@]}"
fi
