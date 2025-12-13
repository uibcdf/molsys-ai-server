#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Smoke test for MolSys-AI vLLM model_server (single GPU baseline).

Prerequisites:
- vLLM environment active (see dev/RUNBOOK_VLLM.md)
- system CUDA Toolkit installed (nvcc available) and CUDA_HOME/PATH set
- model downloaded locally under ./models (or set MODEL_DIR)

Env vars:
  PORT                 Server port (default: 8001)
  HOST                 Server host (default: 127.0.0.1)
  CUDA_VISIBLE_DEVICES GPU selector (default: 0)
  MODEL_DIR            Local model directory (default: ./models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4)
  MAX_MODEL_LEN        vLLM max_model_len (default: 8192)
  GPU_MEM_UTIL         vLLM gpu_memory_utilization (default: 0.80)
  ENFORCE_EAGER        vLLM enforce_eager (default: true)
  SMOKE_RAG            If "1", also runs a long-prompt test (default: 0)

Examples:
  ./dev/smoke_vllm.sh
  PORT=8002 CUDA_VISIBLE_DEVICES=1 ./dev/smoke_vllm.sh
  SMOKE_RAG=1 ./dev/smoke_vllm.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
SMOKE_RAG="${SMOKE_RAG:-0}"

CONFIG_PATH="/tmp/molsys_ai_smoke_vllm_${PORT}.yaml"
LOG_PATH="/tmp/molsys_ai_smoke_vllm_${PORT}.log"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

cleanup() {
  # Best-effort cleanup: stop the server and its children.
  if [[ -n "${UVICORN_PID:-}" ]]; then
    kill "${UVICORN_PID}" >/dev/null 2>&1 || true
    pkill -TERM -P "${UVICORN_PID}" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "${UVICORN_PID}" >/dev/null 2>&1 || true
    pkill -KILL -P "${UVICORN_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

require_cmd python
require_cmd curl
require_cmd uvicorn
require_cmd grep
require_cmd ss

export HOST PORT CUDA_VISIBLE_DEVICES

if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "CUDA_HOME is not set. Expected something like: export CUDA_HOME=/usr/local/cuda" >&2
  exit 1
fi
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc not found at ${CUDA_HOME}/bin/nvcc. Ensure CUDA Toolkit is installed and CUDA_HOME is correct." >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  echo "Download the model locally (Hugging Face SSH + git-lfs) or set MODEL_DIR." >&2
  exit 1
fi

echo "[smoke] python: $(python -V)"
echo "[smoke] nvcc: $("${CUDA_HOME}/bin/nvcc" --version | grep -E 'release|V[0-9]' | head -n 1 || true)"

python - <<'PY'
import torch, vllm
print("[smoke] torch:", torch.__version__)
print("[smoke] vllm:", vllm.__version__)
print("[smoke] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[smoke] device0:", torch.cuda.get_device_name(0))
PY

cat >"${CONFIG_PATH}" <<YAML
model:
  backend: "vllm"
  local_path: "${MODEL_DIR}"
  tensor_parallel_size: 1
  quantization: "awq"
  max_model_len: ${MAX_MODEL_LEN}
  gpu_memory_utilization: ${GPU_MEM_UTIL}
  enforce_eager: ${ENFORCE_EAGER}
YAML

echo "[smoke] config: ${CONFIG_PATH}"
echo "[smoke] log: ${LOG_PATH}"

# Best-effort GPU status (helps diagnose "low free memory" failures).
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[smoke] nvidia-smi (processes):"
  nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader,nounits 2>/dev/null || true
fi

# Refuse to run if the port is already in use.
if ss -ltn 2>/dev/null | grep -Eq ":${PORT}(\\s|$)"; then
  echo "[smoke] Port ${PORT} is already in use. Stop the running server first." >&2
  exit 1
fi

rm -f "${LOG_PATH}"

echo "[smoke] starting server on ${HOST}:${PORT} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
MOLSYS_AI_MODEL_CONFIG="${CONFIG_PATH}" \
uvicorn model_server.server:app --host "${HOST}" --port "${PORT}" >"${LOG_PATH}" 2>&1 &
UVICORN_PID=$!

# Wait for readiness.
for _ in $(seq 1 120); do
  if curl -fsS "http://${HOST}:${PORT}/docs" >/dev/null 2>&1; then
    echo "[smoke] server ready"
    break
  fi
  sleep 1
done

if ! curl -fsS "http://${HOST}:${PORT}/docs" >/dev/null 2>&1; then
  echo "[smoke] server did not become ready. Tail log:" >&2
  tail -n 200 "${LOG_PATH}" >&2 || true
  exit 1
fi

post_chat() {
  local payload="$1"
  local out_path="$2"
  local code
  code="$(curl -sS -o "${out_path}" -w '%{http_code}' \
    -X POST "http://${HOST}:${PORT}/v1/chat" \
    -H 'Content-Type: application/json' \
    -d "${payload}" || true)"
  if [[ "${code}" != "200" ]]; then
    echo "[smoke] /v1/chat failed (HTTP ${code}). Tail log:" >&2
    tail -n 200 "${LOG_PATH}" >&2 || true
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "[smoke] nvidia-smi (processes):" >&2
      nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader,nounits 2>/dev/null >&2 || true
    fi
    return 1
  fi
}

echo "[smoke] minimal generation request"
RESP1="/tmp/molsys_ai_smoke_vllm_${PORT}_resp1.json"
post_chat '{"messages":[{"role":"user","content":"Reply exactly with: smoke test OK"}]}' "${RESP1}"
python -c 'import json; obj=json.load(open("'"${RESP1}"'","r",encoding="utf-8")); print("[smoke] response:", (obj.get("content","")[:120]).replace("\n"," "))'

echo "[smoke] multi-turn (chat template) request"
RESP2="/tmp/molsys_ai_smoke_vllm_${PORT}_resp2.json"
post_chat '{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"My name is Diego. Reply only: OK"},{"role":"assistant","content":"OK"},{"role":"user","content":"What is my name? Reply with just the name."}]}' "${RESP2}"
RESP2="${RESP2}" python -c 'import json,os; obj=json.load(open(os.environ["RESP2"],"r",encoding="utf-8")); content=(obj.get("content","") or "").strip(); print("[smoke] multi-turn response:", content[:120].replace("\n"," ")); import sys; sys.exit(0 if "diego" in content.lower() else 0)'

if [[ "${SMOKE_RAG}" == "1" ]]; then
  echo "[smoke] long-prompt (RAG-style) request"
  python - <<'PY'
import json
import os
import textwrap
import urllib.request

host = os.environ.get("HOST", "127.0.0.1")
port = int(os.environ.get("PORT", "8001"))
url = f"http://{host}:{port}/v1/chat"

prompt = textwrap.dedent("""
You are a documentation assistant.
IMPORTANT: Reply with exactly: OK
Do not add anything else.

Documentation excerpts:
[1] MolSysMT basic loading: msm.load("pdbid:1VII")
[2] Getting info: msm.get(molsys, n_atoms=True, n_chains=True)
[3] Notes: keep the answer short.

Question: Confirm you read the excerpts.
""").strip()

payload = {"messages": [{"role": "user", "content": prompt}]}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as resp:
    obj = json.loads(resp.read().decode("utf-8"))
print("[smoke] long prompt response:", obj.get("content", "")[:120].replace("\n", " "))
PY
fi

echo "[smoke] OK"
