# MolSys-AI Server — Agent Working Guide

This file is the single source of truth for how to work in this repository.
Start here.

## Project summary

This repository hosts the server-side components of MolSys-AI for the UIBCDF MolSysSuite tools (MolSysMT,
MolSysViewer, TopoMT, and related tools). The repository contains:

- client-side code intended to be split out later (`client/agent/`, `client/cli/`),
- server-side services (`server/model_server/`, `server/chat_api/`, `server/rag/`),
- a docs widget asset (`server/web_widget/`),
- internal design docs and ADRs (`dev/`),
- training placeholders (`train/`).

Note: the Python distribution installed by users remains `molsys-ai` (CLI-first).
This repository is the server-side codebase and is expected to live as `molsys-ai-server`
when the CLI/agent is split out.

## Language policy

All text in this repository (source code, comments, documentation, dev notes,
and configuration) must be written in **English**.

## Environments

MolSys-AI uses two complementary environment tracks:

1. **General development** (code + docs + tests): use `environment.yml`.
2. **Inference (vLLM)**: use the runbook in `dev/RUNBOOK_VLLM.md`.

Note: on some HPC systems, `conda` plugins can fail with `PermissionError`
(`multiprocessing.SemLock`). If this happens, run:

- `export CONDA_NO_PLUGINS=true`

### 1) General development (`environment.yml`)

- Intended for: development, docs builds, unit/smoke tests.
- After activating the environment, install the project in editable mode:
  - `pip install -e .` (CLI-only dependencies)
  - `pip install -e ".[dev]"` for server/RAG/docs/test tooling.
  - the docs build prefers `myst-nb` (with a fallback to `myst-parser`).

Local MolSysSuite dependencies (`molsysmt`, `molsysviewer`, `topomt`, etc.) may be
developed in sibling repos. Keep them commented out in `environment.yml` if you
install them manually from source, to avoid overwriting your dev versions with
Conda channel installs.

If you need local tool execution (MolSysSuite tools), prefer a dedicated agent
environment (for example: `conda create -n molsys-agent ...`) and install
`molsys-ai` there. Avoid installing MolSysSuite toolchains into the vLLM inference
environment used for the server.

### 2) Inference (vLLM)

The validated baseline is:

- vLLM installed via pip with CUDA-enabled wheels (`cu129`),
- system-level CUDA Toolkit installed so that `nvcc` is available (FlashInfer JIT),
- AWQ model `uibcdf/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` downloaded locally via
  Hugging Face SSH + `git-lfs`,
- stable single-GPU server settings for 11 GB GPUs:
  - `max_model_len=8192`,
  - `gpu_memory_utilization=0.80`,
  - `enforce_eager=true`.

Validated versions on the current machine:

- `torch==2.9.0+cu129`
- `vllm==0.12.0`
- `sentence-transformers==5.2.0` (for RAG quality; optional)

See `dev/RUNBOOK_VLLM.md` for the full procedure and smoke tests.

## Running servers (non-blocking)

Prefer the helper scripts under `dev/` (they set `PYTHONPATH`, apply a few
sanity checks, and keep the common flags consistent).

To run servers without blocking your terminal, use `nohup`:

```bash
nohup ./dev/run_model_server.sh --config /path/to/model_server.yaml --cuda-devices 0 --warmup > model_server.log 2>&1 &
nohup ./dev/run_chat_api.sh --engine-url http://127.0.0.1:8001 > chat_api.log 2>&1 &
```

For a reproducible, single-GPU vLLM smoke test (including a small multi-turn
check), use:

- `./dev/smoke_vllm.sh`

For the Sphinx widget end-to-end smoke (docs → widget → chat_api → model_server),
use:

- `./dev/smoke_widget.sh`

Note: vLLM may spawn a `VLLM::EngineCore` process; if a crash leaves GPU memory
allocated, kill the stray process before retrying.

Docs chatbot (backend) can be run separately (typically in the dev environment)
and pointed at the model server:

```bash
./dev/run_chat_api.sh --engine-url http://127.0.0.1:8001
```

## Testing conventions

- Run tests with `pytest`.
- Any test that directly or indirectly imports `molsysmt` must mock it to avoid
  `ModuleNotFoundError` in lightweight environments. The convention is to patch
  `sys.modules` before importing modules that depend on `molsysmt`:

```python
mocker.patch.dict("sys.modules", {"molsysmt": mocker.Mock()})
# import after the patch
```

See `tests/test_smoke.py` for the reference pattern.

## RAG indexing (MVP)

- The current MVP uses sentence-transformers embeddings and stores a pickled
  index on disk (default: `data/rag_index.pkl`).
- Index building is implemented in `rag.build_index.build_index(source_dir, index_path)`.
- The corpus sync script `dev/sync_rag_corpus.py` snapshots upstream docs/tutorials into
  `server/chat_api/data/docs/`. Large Markdown pages are truncated (instead of skipped)
  and notebooks are compacted (outputs stripped) to fit size limits.
- The sync always writes a coverage summary to `server/chat_api/data/docs/_coverage.json`.
- By default, upstream `examples/` directories are excluded because they can be stale and may reinforce legacy APIs/aliases.
  Opt-in with `python dev/sync_rag_corpus.py --include-examples ...` when you explicitly want them in the corpus.
- Offline fallback: if `sentence-transformers` is not installed, embeddings fall back
  to a deterministic hashing baseline. This keeps `chat_api` runnable for smoke tests,
  but retrieval quality will be much lower. You can force this mode with:
  - `MOLSYS_AI_EMBEDDINGS=hashing`

Coverage audit:

- After syncing, run `python dev/audit_rag_corpus.py --rescan-sources` to quantify what
  was included/truncated and whether API-surface extraction hit limits.
- If you need per-project control over which upstream directories are scanned, use
  `python dev/sync_rag_corpus.py --corpus-config dev/corpus_config.toml ...` (example:
  `dev/corpus_config.toml.example`).

## Documentation pointers

- Architecture: `dev/ARCHITECTURE.md`
- Roadmap: `dev/ROADMAP.md`
- Constraints: `dev/CONSTRAINTS.md`
- ADRs: `dev/decisions/`
- Status handoff: `checkpoint.md`
- vLLM runbook: `dev/RUNBOOK_VLLM.md`
- Chat API backend: `server/chat_api/README.md`
- Sphinx widget pilot: `docs/index.md`
- API deployment: `dev/DEPLOY_API.md`
- Benchmarking guide: `dev/BENCHMARKING.md`
- Caddy + systemd examples: `dev/Caddyfile.example`, `dev/systemd/`, `dev/molsys-ai.env.example`
- 443 deployment runbook: `dev/RUNBOOK_DEPLOY_443.md`
- API stability contract: `dev/API_CONTRACT_V1.md`

## Public API authentication (current policy)

- `POST /v1/chat` is the public chat API used by both the docs widget and the CLI.
  - It can be optionally protected with `MOLSYS_AI_CHAT_API_KEYS` (note: widget keys are public).
- `POST /v1/engine/chat` is the internal model engine endpoint and should not be exposed publicly:
  - protect it with `MOLSYS_AI_ENGINE_API_KEYS`,
  - set `MOLSYS_AI_ENGINE_API_KEY` on `chat_api` so it can call `http://127.0.0.1:8001/v1/engine/chat`.

## Large files (do not read)

Some directories may contain very large or binary artifacts (for example,
locally downloaded model weights under `models/` via Hugging Face + git-lfs).

- Treat `models/` as binary artifacts. Never read the contents of model weight
  files.
- Do not open, grep, or inline-read model weight files (e.g. `*.safetensors`,
  `*.bin`) or any large binaries.
- It is fine to list the directory or reference the expected model path in
  documentation, but avoid reading the contents to prevent unnecessary context
  bloat and slowdowns.

The same applies to other large artifacts such as:

- `data/rag_index.pkl` (generated RAG index)
- `server/chat_api/data/docs/` (generated corpus snapshot; thousands of files)
- `server/chat_api/data/rag_index.pkl` (generated docs RAG index)
