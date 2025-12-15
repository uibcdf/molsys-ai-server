# Chat Quality Benchmarks (MVP)

This directory contains a minimal, repeatable workflow to track quality changes
as we iterate on:

- corpus contents,
- chunking parameters,
- prompts/routing,
- base model and future fine-tuning.

The benchmark runner is intentionally lightweight and uses only the Python
standard library.

## 1) Prerequisites

- `model_server` running on `http://127.0.0.1:8001`
- `chat_api` running on `http://127.0.0.1:8000`
- A corpus + index already built (recommended):
  - `python dev/sync_rag_corpus.py --clean --build-index --build-anchors`

## 2) Create a question set

Copy the sample and edit it:

```bash
cp dev/benchmarks/questions_v0.sample.jsonl dev/benchmarks/questions_v0.jsonl
```

Each line is a JSON object. Minimal fields:

- `id`: stable identifier
- `query`: user question

Optional lightweight checks:

- `want_sources: true|false` (expects citations like `[1]` and a non-empty `sources` list)
- `contains: ["substring", ...]`
- `contains_any: ["one", "of", "these"]` (at least one must appear)
- `not_contains: ["substring", ...]`

Optional per-question overrides (otherwise runner defaults apply):

- `k`, `client`, `rag`, `sources`
- `messages`: full ChatML list (instead of single-turn `query`)

## 3) Run the benchmark

```bash
python dev/benchmarks/run_chat_bench.py --in dev/benchmarks/questions_v0.jsonl
```

To enable a best-effort semantic check that fails when the answer mentions unknown
MolSysSuite API symbols (requires a local corpus refresh with `--build-api-surface`):

```bash
python dev/benchmarks/run_chat_bench.py --in dev/benchmarks/questions_v0.jsonl --check-symbols
```

This writes a JSONL result file under:

- `dev/benchmarks/runs/`

Each run starts with a `_meta` row describing the configuration.

## 4) Compare runs

The JSONL output is designed to be greppable/diffable:

- search for failures: `rg '"ok": false' dev/benchmarks/runs/*.jsonl`
- inspect sources for a specific question id: `rg '"id": "q3"' -n dev/benchmarks/runs/*.jsonl`

For a human-friendly view, use the viewer:

```bash
python dev/benchmarks/view_run.py dev/benchmarks/runs/<run>.jsonl --only-fail --full
```

For deeper evaluation (semantic correctness), manual review is still required
at this stage; this harness is a baseline that we can later extend.

Planned: semantic checks that fail when answers mention nonexistent API symbols
(see `dev/decisions/ADR-020.md`).
