# Benchmarking and Characterization Guide

This document centralizes what we measure, why we measure it, and how we measure it,
for the MolSys-AI server stack.

Scope:
- `server/model_server` (vLLM engine behind `POST /v1/engine/chat`)
- `server/chat_api` (RAG orchestrator behind `POST /v1/chat`)
- RAG corpus + index artifacts (`server/chat_api/data/`)
- Widget path (Sphinx HTML + JS widget → `POST /v1/chat`) when relevant

Related design records:
- Evaluation strategy (baseline): `dev/decisions/ADR-011.md`
- Corpus strategy + scaling plan: `dev/decisions/ADR-019.md`
- Symbol verification + symbol re-read guardrails: `dev/decisions/ADR-020.md`

## 1) What to measure (watchlist)

### Correctness and trust
- **API symbol correctness**: no invented functions/classes/modules in answers.
- **API usage correctness**: when an API symbol is used, its usage matches `api_surface/` (symbol re-read).
- **Cross-tool mixing**: avoid mixing APIs across MolSysSuite tools unless explicitly asked.
- **Grounding when needed**: citations `[1]`, `[2]` and a non-empty `sources` list when sources are requested or inferred.
- **Contradictions**: avoid “no, but yes” answers; prefer `NOT_DOCUMENTED` when evidence is missing.

### Latency
- **End-to-end (E2E)** time for `POST /v1/chat` (client → chat_api → model_server → response).
- **Engine latency** for `POST /v1/engine/chat` (pure generation, no RAG).
- **Retrieval overhead**: additional time for `rag=on` vs `rag=off` for the same short answer.
- **Warmup time**: time from process start until the first token (or until `/healthz` is ready + warmup request completes).

### Throughput and concurrency
- Requests/second at steady state.
- Tail latency under concurrency (p95/p99).
- Effective “max concurrency” for chosen `max_model_len` and GPU memory settings (vLLM logs provide this).

### Resource usage
- **GPU VRAM** (model weights + KV cache) and GPU utilization.
- **CPU RAM** (chat_api process, embedding model, index resident size).
- **CPU usage** (embedding + preprocessing).
- **Disk usage** of corpus + indices.

### Operational stability
- OOM/segfaults, restart frequency, error rates (HTTP 5xx).
- Determinism of corpus refresh (repeatable snapshot + manifest).

## 2) Baseline metadata to record for every run

Record these alongside every benchmark run:

- model identifier/path (AWQ repo path under `models/`)
- vLLM version, torch version, CUDA version
- engine config: `max_model_len`, `gpu_memory_utilization`, `enforce_eager`, TP/PP sizes
- embeddings: `MOLSYS_AI_EMBEDDINGS` and `MOLSYS_AI_EMBEDDINGS_DEVICE`
- corpus snapshot metadata: `server/chat_api/data/docs/_manifest.json` git SHAs
- index sizes: `du -sh server/chat_api/data/docs server/chat_api/data/rag_index.pkl server/chat_api/data/indexes`

## 3) How to measure each aspect

### 3.1 Corpus refresh time + coverage

The canonical refresh workflow is `dev/sync_rag_corpus.py` (see ADR-019). It writes:

- `server/chat_api/data/docs/_manifest.json`
- `server/chat_api/data/docs/_coverage.json`
- `server/chat_api/data/docs/_api_surface.json` (when `--build-api-surface` is enabled)
- `server/chat_api/data/anchors.json` (when `--build-anchors` is enabled)
- `server/chat_api/data/docs/_symbol_cards.json` (when `--build-symbol-cards` is enabled)
- `server/chat_api/data/docs/_recipes.json` (when `--build-recipes` is enabled)

Measure wall time (example):

```bash
/usr/bin/time -p python dev/sync_rag_corpus.py --clean --build-api-surface --build-index --build-project-indices --build-anchors
```

Recommended “code-aware” refresh (adds symbol cards + recipes):

```bash
/usr/bin/time -p python dev/sync_rag_corpus.py --clean --build-api-surface --build-symbol-cards --build-recipes \
  --build-index --build-project-indices --build-anchors
```

On multi-GPU hosts, shard the index build:

```bash
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
/usr/bin/time -p python dev/sync_rag_corpus.py --clean --build-api-surface --build-index \
  --build-project-indices --build-index-parallel --index-devices 0,1,2 --build-anchors
```

Audit “what got included”:

```bash
python dev/audit_rag_corpus.py --rescan-sources
```

### 3.2 Engine latency and warmup

Run the engine with the standard helper:

- `./dev/run_model_server.sh --config /path/to/config.yaml --cuda-devices 0 --warmup`

Warmup is important because the very first request can be much slower than steady-state.

Basic checks:

```bash
curl -fsS http://127.0.0.1:8001/healthz
curl -fsS http://127.0.0.1:8001/docs >/dev/null
```

### 3.3 Chat API E2E latency (with and without RAG)

Start chat API:

```bash
MOLSYS_AI_ENGINE_URL=http://127.0.0.1:8001 \
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
MOLSYS_AI_PROJECT_INDEX_DIR=server/chat_api/data/indexes \
uvicorn chat_api.backend:app --host 127.0.0.1 --port 8000
```

Measure an E2E request time:

```bash
/usr/bin/time -p curl -sS http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Reply with ONLY: OK","client":"cli","rag":"off","sources":"off"}' >/dev/null
```

Retrieval overhead estimate (same prompt, `rag=on`):

```bash
/usr/bin/time -p curl -sS http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Reply with ONLY: OK","client":"cli","rag":"on","sources":"off","k":5}' >/dev/null
```

### 3.4 GPU/CPU resource monitoring

GPU VRAM and utilization:

```bash
nvidia-smi -l 1
```

CPU RAM / CPU usage:

```bash
ps -o pid,ppid,cmd,%cpu,%mem,rss,vsz -C python | head
```

Disk usage of artifacts:

```bash
du -sh server/chat_api/data/docs server/chat_api/data/rag_index.pkl server/chat_api/data/indexes
```

### 3.5 Quality benchmarks (format + semantic)

Baseline regression harness:

- `dev/benchmarks/run_chat_bench.py`
- `dev/benchmarks/questions_v0.jsonl`
- `dev/benchmarks/view_run.py`

Run:

```bash
python dev/benchmarks/run_chat_bench.py --in dev/benchmarks/questions_v0.jsonl --base-url http://127.0.0.1:8000
```

Semantic checks (recommended):

```bash
python dev/benchmarks/run_chat_bench.py --in dev/benchmarks/questions_v0.jsonl --check-symbols
```

Important note:
- Passing the baseline benchmark means “format + grounding constraints are satisfied”.
- `--check-symbols` adds a best-effort semantic guardrail for invented symbols (ADR-020).
- The symbol check accepts dotted prefixes (e.g. `molsysmt.structure`) when deeper symbols exist in the registry.

### 3.6 Load testing (optional)

When needed, use a simple HTTP load generator to characterize concurrency and tail latency.
Keep `POST /v1/engine/chat` private; load-test `POST /v1/chat` via the reverse proxy configuration.

(Exact tooling is environment-dependent; document the chosen load generator when adopted.)

## 4) Interpreting results and next actions

- If E2E latency is dominated by the engine: tune vLLM config first (context length, KV cache, eager vs graphs).
- If retrieval overhead is high: keep embeddings on CPU for serving, cache aggressively, or move to an ANN backend.
- If semantic correctness is the main issue: prioritize ADR-020 (symbol verification + semantic benchmark checks).
- If the corpus grows significantly: follow ADR-019 scaling plan (derived corpus compaction + vector store upgrade).
