#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the MolSys-AI chat API (RAG orchestrator) locally.

This starts `server/chat_api/backend.py` via uvicorn and serves:
- POST /v1/chat

Required:
  model_server must be reachable (default: http://127.0.0.1:8001)

Usage:
  ./dev/run_chat_api.sh [options]

Options (non-secret; env vars also work):
  --host HOST                 (default: 127.0.0.1)
  --port PORT                 (default: 8000)
  --reload                    Enable uvicorn --reload (dev only).

  --engine-url URL            (default: http://127.0.0.1:8001)
  --cors ORIGINS              (example: http://127.0.0.1:8080,http://localhost:8080)

  --embeddings NAME           (default: sentence-transformers)
  --embeddings-device DEVICE  (default: cpu)
  --embeddings-batch-size N   (default: 64)

  --docs-dir PATH             (default: server/chat_api/data/docs)
  --docs-index PATH           (default: server/chat_api/data/rag_index.pkl)
  --docs-anchors PATH         (default: server/chat_api/data/anchors.json)

Example (with local docs on 8080):
  ./dev/run_chat_api.sh --engine-url http://127.0.0.1:8001 --cors http://127.0.0.1:8080,http://localhost:8080
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/server:${ROOT_DIR}/client:${PYTHONPATH:-}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-0}"

ENGINE_URL_DEFAULT="http://127.0.0.1:8001"

MOLSYS_AI_ENGINE_URL="${MOLSYS_AI_ENGINE_URL:-${ENGINE_URL_DEFAULT}}"
MOLSYS_AI_EMBEDDINGS="${MOLSYS_AI_EMBEDDINGS:-sentence-transformers}"
MOLSYS_AI_EMBEDDINGS_DEVICE="${MOLSYS_AI_EMBEDDINGS_DEVICE:-cpu}"
MOLSYS_AI_EMBEDDINGS_BATCH_SIZE="${MOLSYS_AI_EMBEDDINGS_BATCH_SIZE:-64}"

MOLSYS_AI_DOCS_DIR="${MOLSYS_AI_DOCS_DIR:-}"
MOLSYS_AI_DOCS_INDEX="${MOLSYS_AI_DOCS_INDEX:-}"
MOLSYS_AI_DOCS_ANCHORS="${MOLSYS_AI_DOCS_ANCHORS:-}"
MOLSYS_AI_CORS_ORIGINS="${MOLSYS_AI_CORS_ORIGINS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --reload)
      RELOAD="1"
      shift 1
      ;;
    --engine-url)
      MOLSYS_AI_ENGINE_URL="${2:-}"
      shift 2
      ;;
    --cors)
      MOLSYS_AI_CORS_ORIGINS="${2:-}"
      shift 2
      ;;
    --embeddings)
      MOLSYS_AI_EMBEDDINGS="${2:-}"
      shift 2
      ;;
    --embeddings-device)
      MOLSYS_AI_EMBEDDINGS_DEVICE="${2:-}"
      shift 2
      ;;
    --embeddings-batch-size)
      MOLSYS_AI_EMBEDDINGS_BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --docs-dir)
      MOLSYS_AI_DOCS_DIR="${2:-}"
      shift 2
      ;;
    --docs-index)
      MOLSYS_AI_DOCS_INDEX="${2:-}"
      shift 2
      ;;
    --docs-anchors)
      MOLSYS_AI_DOCS_ANCHORS="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export MOLSYS_AI_ENGINE_URL
export MOLSYS_AI_EMBEDDINGS
export MOLSYS_AI_EMBEDDINGS_DEVICE
export MOLSYS_AI_EMBEDDINGS_BATCH_SIZE
if [[ -n "${MOLSYS_AI_DOCS_DIR}" ]]; then export MOLSYS_AI_DOCS_DIR; fi
if [[ -n "${MOLSYS_AI_DOCS_INDEX}" ]]; then export MOLSYS_AI_DOCS_INDEX; fi
if [[ -n "${MOLSYS_AI_DOCS_ANCHORS}" ]]; then export MOLSYS_AI_DOCS_ANCHORS; fi
if [[ -n "${MOLSYS_AI_CORS_ORIGINS}" ]]; then export MOLSYS_AI_CORS_ORIGINS; fi

echo "[chat_api] starting on http://${HOST}:${PORT}"
echo "[chat_api] OpenAPI: http://${HOST}:${PORT}/docs"
echo "[chat_api] Health:  curl -fsS http://${HOST}:${PORT}/healthz"

args=(chat_api.backend:app --host "${HOST}" --port "${PORT}")
if [[ "${RELOAD}" == "1" ]]; then
  args+=(--reload)
fi

python -m uvicorn "${args[@]}"
