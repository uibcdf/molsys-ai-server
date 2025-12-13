# MolSys-AI — Agent Working Guide

This file is the single source of truth for how to work in this repository.
Start here.

## Project summary

MolSys-AI is an AI assistant project for the UIBCDF MolSys* ecosystem (MolSysMT,
MolSysViewer, TopoMT, and related tools). The repository contains:

- an MVP agent loop (`agent/`),
- a model server (`model_server/`),
- a RAG layer (`rag/`),
- a docs chatbot backend + widget (`docs_chat/`, `web_widget/`),
- internal design docs and ADRs (`dev/`),
- training placeholders (`train/`).

## Language policy

All text in this repository (source code, comments, documentation, dev notes,
and configuration) must be written in **English**.

## Environments

MolSys-AI uses two complementary environment tracks:

1. **General development** (code + docs + tests): use `environment.yml`.
2. **Inference (vLLM)**: use the runbook in `dev/RUNBOOK_VLLM.md`.

### 1) General development (`environment.yml`)

- Intended for: development, docs builds, unit/smoke tests.
- After activating the environment, install the project in editable mode:
  - `pip install -e .`
  - or `pip install -e ".[dev]"` for lint/test/docs tooling.

Local MolSys* dependencies (`molsysmt`, `molsysviewer`, `topomt`, etc.) may be
developed in sibling repos. Keep them commented out in `environment.yml` if you
install them manually from source, to avoid overwriting your dev versions with
Conda channel installs.

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

See `dev/RUNBOOK_VLLM.md` for the full procedure and smoke tests.

## Running servers (non-blocking)

To run servers without blocking your terminal, use `nohup`:

```bash
nohup uvicorn model_server.server:app --reload > server.log 2>&1 &
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

## Documentation pointers

- Architecture: `dev/ARCHITECTURE.md`
- Roadmap: `dev/ROADMAP.md`
- Constraints: `dev/CONSTRAINTS.md`
- ADRs: `dev/decisions/`
- Status handoff: `checkpoint.md`
- vLLM runbook: `dev/RUNBOOK_VLLM.md`

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
