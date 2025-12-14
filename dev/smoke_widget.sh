#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
End-to-end smoke test for the Sphinx widget → chat_api → model_server path.

This script:
- builds the Sphinx pilot docs,
- starts a model_server (vLLM) on a local port,
- starts chat_api with CORS enabled for the docs origin,
- serves the built docs via python http.server,
- validates:
  - the HTML includes widget scripts,
  - CORS preflight (OPTIONS) succeeds,
  - cross-origin POST to /v1/chat works and returns the expected answer,
- cleans up on exit.

Prerequisites:
- Run in an environment with: python, sphinx-build, uvicorn, curl
- For vLLM backend: CUDA Toolkit + nvcc available and a local model dir.

Env vars (optional):
  HOST                 (default: 127.0.0.1)
  MODEL_PORT           (default: 8001)
  CHAT_API_PORT        (default: 8000)
  DOCS_HTTP_PORT       (default: 8080)
  CUDA_VISIBLE_DEVICES (default: 0)
  CUDA_HOME            (default: /usr/local/cuda)
  MODEL_DIR            (default: ./models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4)
  GPU_MEM_UTIL         (default: 0.75)
  MAX_MODEL_LEN        (default: 4096)
  ENFORCE_EAGER        (default: true)
  EMBEDDINGS           (default: sentence-transformers)

Examples:
  ./dev/smoke_widget.sh
  DOCS_HTTP_PORT=8081 MODEL_PORT=8091 CHAT_API_PORT=8090 ./dev/smoke_widget.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/server:${ROOT_DIR}/client:${PYTHONPATH:-}"

HOST="${HOST:-127.0.0.1}"
MODEL_PORT="${MODEL_PORT:-8001}"
CHAT_API_PORT="${CHAT_API_PORT:-8000}"
DOCS_HTTP_PORT="${DOCS_HTTP_PORT:-8080}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
EMBEDDINGS="${EMBEDDINGS:-sentence-transformers}"

MODEL_CFG="/tmp/molsys_ai_widget_model_${MODEL_PORT}.yaml"
MODEL_LOG="/tmp/molsys_ai_widget_model_${MODEL_PORT}.log"
CHAT_API_LOG="/tmp/molsys_ai_widget_chat_api_${CHAT_API_PORT}.log"
DOCS_HTTP_LOG="/tmp/molsys_ai_widget_docs_http_${DOCS_HTTP_PORT}.log"
DOCS_DIR="/tmp/molsys_ai_widget_docs_src_${CHAT_API_PORT}"
DOCS_INDEX="/tmp/molsys_ai_widget_docs_index_${CHAT_API_PORT}.pkl"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

cleanup() {
  if [[ -n "${HTTP_PID:-}" ]]; then
    kill -TERM "${HTTP_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CHAT_API_PID:-}" ]]; then
    kill -TERM "${CHAT_API_PID}" >/dev/null 2>&1 || true
    pkill -TERM -P "${CHAT_API_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${MODEL_PID:-}" ]]; then
    kill -TERM "${MODEL_PID}" >/dev/null 2>&1 || true
    pkill -TERM -P "${MODEL_PID}" >/dev/null 2>&1 || true
  fi
  sleep 2
  if [[ -n "${HTTP_PID:-}" ]]; then
    kill -KILL "${HTTP_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CHAT_API_PID:-}" ]]; then
    kill -KILL "${CHAT_API_PID}" >/dev/null 2>&1 || true
    pkill -KILL -P "${CHAT_API_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${MODEL_PID:-}" ]]; then
    kill -KILL "${MODEL_PID}" >/dev/null 2>&1 || true
    pkill -KILL -P "${MODEL_PID}" >/dev/null 2>&1 || true
  fi

  # Best-effort cleanup in case vLLM left an EngineCore behind.
  pkill -TERM -f 'VLLM::EngineCore' >/dev/null 2>&1 || true
}
trap cleanup EXIT

require_cmd python
require_cmd sphinx-build
require_cmd uvicorn
require_cmd curl

python - <<'PY'
import os
import socket
import sys

host = os.environ.get("HOST", "127.0.0.1")
ports = [
    int(os.environ.get("MODEL_PORT", "8001")),
    int(os.environ.get("CHAT_API_PORT", "8000")),
    int(os.environ.get("DOCS_HTTP_PORT", "8080")),
]

for port in ports:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
    except OSError:
        print(f"Port already in use: {port}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        s.close()
PY

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc not found at ${CUDA_HOME}/bin/nvcc (set CUDA_HOME or install CUDA Toolkit)." >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

echo "[widget] building Sphinx docs"
sphinx-build -b html "$ROOT_DIR/docs" "$ROOT_DIR/docs/_build/html" >/tmp/molsys_ai_widget_sphinx_build_${DOCS_HTTP_PORT}.log 2>&1

mkdir -p "${DOCS_DIR}"
cat >"${DOCS_DIR}/example.md" <<'MD'
# Widget smoke doc

The example PDB id is **1VII**.
MD

cat >"${MODEL_CFG}" <<YAML
model:
  backend: "vllm"
  local_path: "${MODEL_DIR}"
  tensor_parallel_size: 1
  quantization: "awq"
  max_model_len: ${MAX_MODEL_LEN}
  gpu_memory_utilization: ${GPU_MEM_UTIL}
  enforce_eager: ${ENFORCE_EAGER}
YAML

rm -f "${MODEL_LOG}" "${CHAT_API_LOG}" "${DOCS_HTTP_LOG}" "${DOCS_INDEX}"

echo "[widget] starting model_server on ${HOST}:${MODEL_PORT} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MOLSYS_AI_MODEL_CONFIG="${MODEL_CFG}" \
MOLSYS_AI_ENGINE_API_KEYS="${MOLSYS_AI_ENGINE_API_KEYS:-}" \
uvicorn model_server.server:app --host "${HOST}" --port "${MODEL_PORT}" >"${MODEL_LOG}" 2>&1 &
MODEL_PID=$!

for _ in $(seq 1 240); do
  if curl -fsS "http://${HOST}:${MODEL_PORT}/docs" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://${HOST}:${MODEL_PORT}/docs" >/dev/null 2>&1 || { tail -n 160 "${MODEL_LOG}" >&2 || true; exit 1; }

CHAT_AUTH_HEADER=()
if [[ -n "${MOLSYS_AI_CHAT_API_KEY:-}" ]]; then
  CHAT_AUTH_HEADER=(-H "Authorization: Bearer ${MOLSYS_AI_CHAT_API_KEY}")
fi

echo "[widget] starting chat_api on ${HOST}:${CHAT_API_PORT}"
MOLSYS_AI_ENGINE_URL="http://${HOST}:${MODEL_PORT}" \
MOLSYS_AI_ENGINE_API_KEY="${MOLSYS_AI_ENGINE_API_KEY:-}" \
MOLSYS_AI_DOCS_DIR="${DOCS_DIR}" \
MOLSYS_AI_DOCS_INDEX="${DOCS_INDEX}" \
MOLSYS_AI_EMBEDDINGS="${EMBEDDINGS}" \
MOLSYS_AI_CORS_ORIGINS="http://${HOST}:${DOCS_HTTP_PORT},http://localhost:${DOCS_HTTP_PORT}" \
MOLSYS_AI_CHAT_API_KEYS="${MOLSYS_AI_CHAT_API_KEYS:-}" \
uvicorn chat_api.backend:app --host "${HOST}" --port "${CHAT_API_PORT}" >"${CHAT_API_LOG}" 2>&1 &
CHAT_API_PID=$!

for _ in $(seq 1 120); do
  if curl -fsS "http://${HOST}:${CHAT_API_PORT}/docs" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://${HOST}:${CHAT_API_PORT}/docs" >/dev/null 2>&1 || { tail -n 200 "${CHAT_API_LOG}" >&2 || true; exit 1; }

for _ in $(seq 1 120); do
  if [[ -f "${DOCS_INDEX}" ]]; then
    break
  fi
  sleep 1
done
[[ -f "${DOCS_INDEX}" ]] || { echo "[widget] docs index not created" >&2; tail -n 200 "${CHAT_API_LOG}" >&2 || true; exit 1; }

echo "[widget] serving docs on ${HOST}:${DOCS_HTTP_PORT}"
python -m http.server "${DOCS_HTTP_PORT}" --directory "$ROOT_DIR/docs/_build/html" >"${DOCS_HTTP_LOG}" 2>&1 &
HTTP_PID=$!

for _ in $(seq 1 30); do
  if curl -fsS "http://${HOST}:${DOCS_HTTP_PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS "http://${HOST}:${DOCS_HTTP_PORT}/" | grep -q "_static/molsys_ai_config.js"
curl -fsS "http://${HOST}:${DOCS_HTTP_PORT}/" | grep -q "_static/molsys_ai_widget.js"
echo "[widget] html includes widget scripts"

echo "[widget] CORS preflight:"
curl -i -sS -X OPTIONS "http://${HOST}:${CHAT_API_PORT}/v1/chat" \
  -H "Origin: http://${HOST}:${DOCS_HTTP_PORT}" \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: content-type,authorization' \
  | grep -i '^access-control-allow-origin:' | head -n 1

echo "[widget] cross-origin POST:"
ans="$(curl -sS -X POST "http://${HOST}:${CHAT_API_PORT}/v1/chat" \
  -H "Origin: http://${HOST}:${DOCS_HTTP_PORT}" \
  -H 'Content-Type: application/json' \
  "${CHAT_AUTH_HEADER[@]}" \
  -d '{"query":"What is the example PDB id? Reply with just the id.","k":2}' \
  | python -c 'import json,sys; print(json.load(sys.stdin).get("answer",""))')"
echo "[widget] answer: ${ans}"

python - <<PY
ans = """${ans}""".strip().lower()
assert "1vii" in ans
print("[widget] OK")
PY

echo "[widget] Open in a browser:"
echo "http://${HOST}:${DOCS_HTTP_PORT}/?molsys_ai_mode=backend&molsys_ai_backend_url=http://${HOST}:${CHAT_API_PORT}/v1/chat"
