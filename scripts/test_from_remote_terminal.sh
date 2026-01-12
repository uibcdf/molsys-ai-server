#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-https://api.uibcdf.org}"
API_KEY="${API_KEY:-}"

headers=(-H 'Content-Type: application/json')
if [[ -n "${API_KEY}" ]]; then
  headers+=(-H "Authorization: Bearer ${API_KEY}")
fi

curl -fsS -X POST "${BASE_URL}/v1/chat" \
  "${headers[@]}" \
  -d '{
    "messages":[{"role":"user","content":"What is MolSysMT? Give a short answer and cite sources."}],
    "rag":"on",
    "sources":"on",
    "k":5
  }'
