#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build a full, "serious" RAG corpus snapshot + derived layers + indices.

This is the canonical offline/batch refresh pipeline for MolSys-AI Server.

What it does:
  - snapshots sibling repos into a literal docs tree (default: server/chat_api/data/docs)
  - builds derived layers (api_surface, symbol_cards, recipes, anchors)
  - builds the embedding index + per-project indices
  - optionally builds a BM25 sidecar (recommended)

Usage:
  ./dev/refresh_rag_full.sh [options]

Options:
  --dest PATH              Destination docs snapshot dir (default: server/chat_api/data/docs)
  --index PATH             Global index output path (default: server/chat_api/data/rag_index.pkl)
  --index-dir PATH         Per-project index dir (default: server/chat_api/data/indexes)
  --anchors-out PATH       Anchors JSON output path (default: server/chat_api/data/anchors.json)
  --corpus-config PATH     Optional corpus selection config (.toml/.json)

  --index-parallel         Enable parallel index build (multiprocess; optionally multi-GPU)
  --index-devices LIST     Comma-separated GPU ids for index workers (e.g. "0,1,2")
  --index-workers N        Override worker count (default: auto)

  --no-bm25                Do not build BM25 sidecars (default: build)

Notes:
  - This script always uses --clean to avoid stale artifacts (critical for correctness).
  - "Offline mode" means "do not attempt downloads". If needed, run with:
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./dev/refresh_rag_full.sh ...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEST="${DEST:-${ROOT_DIR}/server/chat_api/data/docs}"
INDEX="${INDEX:-${ROOT_DIR}/server/chat_api/data/rag_index.pkl}"
INDEX_DIR="${INDEX_DIR:-${ROOT_DIR}/server/chat_api/data/indexes}"
ANCHORS_OUT="${ANCHORS_OUT:-${ROOT_DIR}/server/chat_api/data/anchors.json}"
CORPUS_CONFIG="${CORPUS_CONFIG:-}"

INDEX_PARALLEL="0"
INDEX_DEVICES="${INDEX_DEVICES:-}"
INDEX_WORKERS="${INDEX_WORKERS:-}"
BUILD_BM25="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="${2:-}"
      shift 2
      ;;
    --index)
      INDEX="${2:-}"
      shift 2
      ;;
    --index-dir)
      INDEX_DIR="${2:-}"
      shift 2
      ;;
    --anchors-out)
      ANCHORS_OUT="${2:-}"
      shift 2
      ;;
    --corpus-config)
      CORPUS_CONFIG="${2:-}"
      shift 2
      ;;
    --index-parallel)
      INDEX_PARALLEL="1"
      shift 1
      ;;
    --index-devices)
      INDEX_DEVICES="${2:-}"
      shift 2
      ;;
    --index-workers)
      INDEX_WORKERS="${2:-}"
      shift 2
      ;;
    --no-bm25)
      BUILD_BM25="0"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cmd=(python3 dev/sync_rag_corpus.py
  --clean
  --dest "${DEST}"
  --index "${INDEX}"
  --index-dir "${INDEX_DIR}"
  --anchors-out "${ANCHORS_OUT}"
  --build-api-surface
  --build-symbol-cards
  --build-recipes
  --build-anchors
  --build-index
  --build-project-indices
)

if [[ -n "${CORPUS_CONFIG}" ]]; then
  cmd+=(--corpus-config "${CORPUS_CONFIG}")
fi

if [[ "${BUILD_BM25}" == "1" ]]; then
  cmd+=(--build-bm25)
fi

if [[ "${INDEX_PARALLEL}" == "1" ]]; then
  cmd+=(--build-index-parallel)
  if [[ -n "${INDEX_DEVICES}" ]]; then
    cmd+=(--index-devices "${INDEX_DEVICES}")
  fi
  if [[ -n "${INDEX_WORKERS}" ]]; then
    cmd+=(--index-workers "${INDEX_WORKERS}")
  fi
fi

echo "[refresh_rag_full] dest:        ${DEST}"
echo "[refresh_rag_full] index:       ${INDEX}"
echo "[refresh_rag_full] index_dir:   ${INDEX_DIR}"
echo "[refresh_rag_full] anchors_out: ${ANCHORS_OUT}"
if [[ -n "${CORPUS_CONFIG}" ]]; then
  echo "[refresh_rag_full] corpus_config: ${CORPUS_CONFIG}"
fi

exec "${cmd[@]}"

