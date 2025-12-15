
# MolSys-AI Server

This repository hosts the **server-side** components of the MolSys-AI project for the UIBCDF ecosystem:

- MolSysMT
- MolSysViewer
- TopoMT
- and related tools in the MolSysSuite ecosystem.

MolSys-AI aims to provide:

- An **autonomous agent** that can design and execute workflows using MolSysSuite tools.
- A **CLI interface** to interact with the agent from the terminal.
- A **documentation chatbot** embedded in Sphinx/GitHub Pages.
- A flexible **model serving** layer for self-hosted LLMs.

This repository currently contains the initial architecture, decisions and development roadmap.

Chat API note:

- `POST /v1/chat` returns an `answer` that can cite bracketed sources (`[1]`, `[2]`, ...) and a `sources` list that can
  deep-link to published docs under `https://www.uibcdf.org/<tool>/...#Label` (when an anchors map is available).

## Repository naming note

This repository is the server-side codebase and is expected to live as **`molsys-ai-server`**.
The Python package and user-facing command for end users remain `molsys-ai`.

See:

- `dev/ARCHITECTURE.md` for high-level architecture.
- `dev/ROADMAP.md` for the initial roadmap.
- `dev/CONSTRAINTS.md` for current constraints and assumptions.
- `dev/decisions/` for Architectural Decision Records (ADRs).

## Development environment

Two environments are commonly used:

1. **Development (general code + docs + tests)** via `environment.yml`.
2. **Inference (vLLM on RTX GPUs)** via the runbook in `dev/RUNBOOK_VLLM.md`.

### Option A — Development environment (`environment.yml`)

The recommended way to create a development environment is with `mamba` or `conda`:

```bash
mamba env create -f environment.yml
mamba activate molsys-ai
```

or:

```bash
conda env create -f environment.yml
conda activate molsys-ai
```

This environment uses Python 3.12 and includes the core Python dependencies used by this repository.  
Alternative setups using `venv` + `pip` are described in `dev/DEV_GUIDE.md`.

Note: `environment.yml` intentionally keeps MolSysSuite tools commented out by default to avoid
pulling extra CUDA-related stacks into the same environment used for vLLM inference.
For local tool execution, use a dedicated agent environment (see `client/cli/README.md`).

Note on Python packaging:

- `pip install molsys-ai` is intended to install the **CLI client** (lightweight).
- Server/RAG/docs dependencies are installed via extras (for development use `pip install -e ".[dev]"`).

### Option B — Inference environment (vLLM)

For the current, working vLLM setup (CUDA 12.9 + AWQ model + `uvicorn` smoke tests),
see:

- `dev/RUNBOOK_VLLM.md`
- `healthy_vllm_env.md` (minimal Conda+pip environment note)

## Language policy

All text in this repository (source code, comments, documentation, development notes, and configuration files) must be written in **English**.
