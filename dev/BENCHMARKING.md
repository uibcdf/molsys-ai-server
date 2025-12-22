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

The canonical refresh workflow is `dev/sync_rag_corpus.py` (see ADR-019). A convenience wrapper is also available:

- `./dev/refresh_rag_full.sh` (recommended for most runs; always uses `--clean`)

The refresh writes:

- `server/chat_api/data/docs/_manifest.json`
- `server/chat_api/data/docs/_coverage.json`
- `server/chat_api/data/docs/_api_surface.json` (when `--build-api-surface` is enabled)
- `server/chat_api/data/anchors.json` (when `--build-anchors` is enabled)
- `server/chat_api/data/docs/_symbol_cards.json` (when `--build-symbol-cards` is enabled)
- `server/chat_api/data/docs/_recipes.json` (when `--build-recipes` is enabled)
  - includes notebook-derived recipes at three granularities:
    - `recipes/notebooks_tutorials/` (tutorial overview per notebook),
    - `recipes/notebooks_sections/` (multi-cell section blocks; with stitched preambles),
    - `recipes/notebooks/` (per-cell snippets).

Measure wall time (example):

```bash
/usr/bin/time -p python dev/sync_rag_corpus.py --clean --build-api-surface --build-index --build-project-indices --build-anchors
```

If you need per-project control over which upstream directories are scanned, pass a corpus config:

```bash
python dev/sync_rag_corpus.py --corpus-config dev/corpus_config.toml --clean --build-index
```

Recommended “code-aware” refresh (adds symbol cards + recipes):

```bash
/usr/bin/time -p python dev/sync_rag_corpus.py --clean --build-api-surface --build-symbol-cards --build-recipes \
  --build-index --build-project-indices --build-anchors
```

If you want stronger identifier matching, also build a BM25 sidecar during indexing:

```bash
python dev/sync_rag_corpus.py --clean --build-index --build-bm25
```

Then tune BM25 mixing at runtime by setting (in the `chat_api` environment):

- `MOLSYS_AI_RAG_BM25_WEIGHT` (try `0.15`–`0.35`)
- `MOLSYS_AI_RAG_HYBRID_WEIGHT` (light lexical boost; default `0.15`)

If you also generate offline LLM-digested `recipe_cards/` (tutorial/section/test digests), rebuild indices after digestion:

```bash
python dev/digest_recipes_llm.py --docs-dir server/chat_api/data/docs
python dev/sync_rag_corpus.py --build-index --build-project-indices --build-index-parallel --index-devices 0,1
```

Notes on recipes:

- `--build-recipes` includes:
  - notebook-derived recipes (tutorial/section/cell),
  - test snippets (AST),
  - docstring examples (doctest/fenced code; AST),
  - fenced code blocks from Markdown pages.

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

Notebook recipe audit (tutorial → section → cell):

```bash
python dev/audit_notebook_recipes.py \
  --notebook molsysmt/docs/content/user/tools/basic/select.ipynb \
  --notebook molsysmt/docs/content/user/tools/structure/get_distances.ipynb \
  --notebook molsysmt/docs/content/showcase/barnase_barstar.ipynb
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
./dev/run_chat_api.sh --engine-url http://127.0.0.1:8001
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

Stricter semantic check (recommended when you expect the answer to be correct and grounded, even if the model says
`NOT_DOCUMENTED`):

```bash
python dev/benchmarks/run_chat_bench.py --in dev/benchmarks/questions_v0.jsonl --check-symbols --strict-symbols
```

Quality gate (recommended as a default regression run):

```bash
./dev/benchmarks/run_gate.sh
```

To review results without truncation:

```bash
python dev/benchmarks/view_run.py dev/benchmarks/runs/<run>.jsonl --only-fail
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
