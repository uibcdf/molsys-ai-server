# MolSys-AI Server — Agent Working Guide

This file is the single source of truth for how to work in this repository.
Start here.

## Project summary

This repository hosts the server-side components of MolSys-AI for the UIBCDF MolSys* ecosystem (MolSysMT,
MolSysViewer, TopoMT, and related tools). The repository contains:

- client-side code intended to be split out later (`client/agent/`, `client/cli/`),
- server-side services (`server/model_server/`, `server/docs_chat/`, `server/rag/`),
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

Local MolSys* dependencies (`molsysmt`, `molsysviewer`, `topomt`, etc.) may be
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

To run servers without blocking your terminal, use `nohup`:

```bash
nohup uvicorn model_server.server:app --reload > server.log 2>&1 &
```

For a reproducible, single-GPU vLLM smoke test (including a small multi-turn
check), use:

- `./dev/smoke_vllm.sh`

For the Sphinx widget end-to-end smoke (docs → widget → docs_chat → model_server),
use:

- `./dev/smoke_widget.sh`

Note: vLLM may spawn a `VLLM::EngineCore` process; if a crash leaves GPU memory
allocated, kill the stray process before retrying.

Docs chatbot (backend) can be run separately (typically in the dev environment)
and pointed at the model server:

```bash
MOLSYS_AI_MODEL_SERVER_URL=http://127.0.0.1:8001 uvicorn docs_chat.backend:app --host 127.0.0.1 --port 8000
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
- Offline fallback: if `sentence-transformers` is not installed, embeddings fall back
  to a deterministic hashing baseline. This keeps `docs_chat` runnable for smoke tests,
  but retrieval quality will be much lower. You can force this mode with:
  - `MOLSYS_AI_EMBEDDINGS=hashing`

## Documentation pointers

- Architecture: `dev/ARCHITECTURE.md`
- Roadmap: `dev/ROADMAP.md`
- Constraints: `dev/CONSTRAINTS.md`
- ADRs: `dev/decisions/`
- Status handoff: `checkpoint.md`
- vLLM runbook: `dev/RUNBOOK_VLLM.md`
- Docs chatbot backend: `server/docs_chat/README.md`
- Sphinx widget pilot: `docs/index.md`
- API deployment: `dev/DEPLOY_API.md`
- Caddy + systemd examples: `dev/Caddyfile.example`, `dev/systemd/`, `dev/molsys-ai.env.example`
- 443 deployment runbook: `dev/RUNBOOK_DEPLOY_443.md`
- API stability contract: `dev/API_CONTRACT_V1.md`

## Public API authentication (current policy)

- `POST /v1/docs-chat` is intended for the public docs widget (CORS-enabled).
  - It can be optionally protected with `MOLSYS_AI_DOCS_CHAT_API_KEYS` if needed.
- `POST /v1/chat` is intended for CLI / non-browser clients and should be protected in production:
  - set `MOLSYS_AI_CHAT_API_KEYS` on the model server,
  - set `MOLSYS_AI_MODEL_SERVER_API_KEY` on docs_chat so it can call `/v1/chat`.

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
- `server/docs_chat/data/docs/` (generated corpus snapshot; thousands of files)
- `server/docs_chat/data/rag_index.pkl` (generated docs RAG index)
