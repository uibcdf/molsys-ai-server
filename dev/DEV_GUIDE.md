
# MolSys-AI – Developer Guide (Setup & First Steps)

This guide explains how to set up a development environment for MolSys-AI
and what to run first once the repository is cloned.

## 1. Python version

- Recommended: **Python 3.12**
- Minimum: **Python 3.10**

Check your version:

```bash
python --version
```

or, depending on your system:

```bash
python3 --version
```

## 2. Create and activate a development environment

You can use either a pre-defined conda/mamba environment (recommended) or a manual environment.

### Option A – conda/mamba with `environment.yml` (recommended)

From the root of the repository:

```bash
mamba env create -f environment.yml
mamba activate molsys-ai
```

or, if you use `conda`:

```bash
conda env create -f environment.yml
conda activate molsys-ai
```

This environment includes:
- Python 3.12,
- the MolSys* ecosystem tools from the `uibcdf` channel,
- and the core Python tooling used by MolSys-AI.

### Option B – manual environment (`conda` or `venv`)

If you prefer to build the environment manually, you can:

- Use `conda`/`mamba`:

  ```bash
  conda create -n molsys-ai python=3.12
  conda activate molsys-ai
  ```

- Or use a standard `venv`:

From the root of the repository:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows (PowerShell/CMD)
```

You should see the environment name (e.g. `(.venv)` or `(molsys-ai)`) in your shell prompt after activation.

## 3. Install development dependencies

With your environment active, install the project in editable mode.

Basic (runtime) install:

```bash
pip install -e .
```

Developer extras (lint/test/docs tooling):

```bash
pip install -e ".[dev]"
```

This installs:

- fastapi, uvicorn, pydantic
- rich
- pytest
- ruff
- mypy (optional type checking)

## 4. (Optional) Install MolSys* ecosystem tools via conda (manual environments only)

If you created the environment manually (Option B) and plan to develop or test tools that call the MolSys* libraries,  
you can install them from the laboratory’s `uibcdf` conda channel:

```bash
conda install -c uibcdf molsysmt molsysviewer topomt elastnet pharmacophoremt
```

You can select only the subset you actually need. These libraries are already included if you created the environment from `environment.yml`.

## 5. Run the smoke tests

To verify that the basic skeleton imports correctly:

```bash
pytest
```

You should see all tests passing.

## 6. Run the model server

### 6.1 Stub backend (default)

If you have **no** `model_server/config.yaml`, or if it contains:

```yaml
model:
  backend: "stub"
```

the server will run in stub mode (echoing the last user message) and **no**
model weights will be loaded.

Launch the FastAPI model server:

```bash
uvicorn model_server.server:app --reload
```

By default this will start on `http://127.0.0.1:8000`.

You can inspect the OpenAPI docs at:

- http://127.0.0.1:8000/docs

In this mode, the `/v1/chat` endpoint returns a stubbed reply.

### 6.2 vLLM backend (current baseline)

To use a real model via vLLM:

1. Set up the vLLM environment and install a system CUDA Toolkit (`nvcc`).
2. Download the AWQ model locally (Hugging Face SSH + `git-lfs`).
3. Create `model_server/config.yaml` based on `model_server/config.example.yaml`.

For the full, validated procedure, see:

- `dev/RUNBOOK_VLLM.md`

## 7. Try the CLI

With the virtual environment active:

```bash
python -m cli.main --message "Hello from MolSys-AI"
```

For now this should print a simple message from the stub model client. You can
also point the CLI to a running model server:

```text
molsys-ai --server-url http://127.0.0.1:8000 --message "Hello from MolSys-AI"
```

As the agent and model server evolve, this CLI will be extended to support
more commands and interactive sessions.

## 8. Code style and tooling

- Linting and formatting: `ruff`
- Testing: `pytest`
- Optional static type checking: `mypy`

Examples:

```bash
ruff check .
pytest
mypy agent model_server rag
```

These commands are not yet enforced by CI, but are recommended during development.

## 9. Where to start hacking

Suggested starting points for development:

- Agent core:
  - `agent/core.py`
  - `agent/planner.py`
  - `agent/executor.py`

- Model server:
  - `model_server/server.py`
  - `model_server/config.example.yaml`

- RAG:
  - `rag/build_index.py`
  - `rag/retriever.py`
  - `rag/embeddings.py`

- CLI:
  - `cli/main.py`

For a conceptual overview, read:

- `dev/ARCHITECTURE.md`
- `dev/ROADMAP.md`
- `dev/CONSTRAINTS.md`
- ADRs in `dev/decisions/`

## 10. Language conventions

- All text in this repository must be written in **English**.
- This includes source code, comments, documentation, Markdown files (including `dev/`), and configuration files.
- Existing non-English fragments should be translated to English; new contributions in other languages should not be added.

## 11. Questions and open points

See `dev/OPEN_QUESTIONS.md` for items that are intentionally left open
for discussion (embedding model choice, backend refinements, etc.).
