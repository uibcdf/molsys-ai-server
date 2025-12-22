#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the benchmark "quality gate" for the public chat API.

This enforces semantic symbol correctness (no invented tool symbols), in addition
to the usual formatting/citations checks.

Usage:
  ./dev/benchmarks/run_gate.sh [options]

Options:
  --base-url URL      Chat API base URL (default: http://127.0.0.1:8000)
  --in PATH           Questions JSONL (default: dev/benchmarks/questions_v0.jsonl)
  --api-key KEY       Optional API key for /v1/chat
EOF
}

BASE_URL="http://127.0.0.1:8000"
IN_PATH="dev/benchmarks/questions_v0.jsonl"
API_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --in)
      IN_PATH="${2:-}"
      shift 2
      ;;
    --api-key)
      API_KEY="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

args=(python3 dev/benchmarks/run_chat_bench.py
  --base-url "${BASE_URL}"
  --in "${IN_PATH}"
  --check-symbols
  --strict-symbols
)
if [[ -n "${API_KEY}" ]]; then
  args+=(--api-key "${API_KEY}")
fi

exec "${args[@]}"

