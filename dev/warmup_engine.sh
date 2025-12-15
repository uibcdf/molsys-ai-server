#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Warm up the MolSys-AI model engine server to reduce first-request latency.

This sends a small request to:
  POST /v1/engine/chat

Options:
  --host HOST   (default: 127.0.0.1)
  --port PORT   (default: 8001)
  --tries N     (default: 240)

Auth:
  If the engine endpoint is protected, set:
    MOLSYS_AI_ENGINE_API_KEY

Example:
  ./dev/warmup_engine.sh
  ./dev/warmup_engine.sh --host 127.0.0.1 --port 8001
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

HOST="127.0.0.1"
PORT="8001"
TRIES="240"

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
    --tries)
      TRIES="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  echo "Missing command: curl" >&2
  exit 1
fi

echo "[warmup] waiting for engine readiness at http://${HOST}:${PORT}/docs ..."
for _ in $(seq 1 "${TRIES}"); do
  if curl -fsS "http://${HOST}:${PORT}/docs" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://${HOST}:${PORT}/docs" >/dev/null 2>&1 || { echo "[warmup] engine not ready" >&2; exit 1; }

AUTH_HEADER=()
if [[ -n "${MOLSYS_AI_ENGINE_API_KEY:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${MOLSYS_AI_ENGINE_API_KEY}")
fi

echo "[warmup] sending warmup request..."
curl -fsS -X POST "http://${HOST}:${PORT}/v1/engine/chat" \
  -H 'Content-Type: application/json' \
  "${AUTH_HEADER[@]}" \
  -d '{"messages":[{"role":"user","content":"warmup: reply only OK"}]}' >/dev/null

echo "[warmup] done"

