
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
- and the core Python tooling used by MolSys-AI.

Note: MolSysSuite tools (`molsysmt`, `molsysviewer`, etc.) are intentionally
commented out in `environment.yml` by default to avoid pulling heavy scientific
stacks (some tools pull extra CUDA-related dependencies) into the same environment used for server-side vLLM
inference.

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

Basic (CLI-only) install:

```bash
pip install -e .
```

Developer extras (server/RAG/docs + lint/test tooling):

```bash
pip install -e ".[dev]"
```

This installs:

- fastapi, uvicorn, pydantic, PyYAML
- numpy + sentence-transformers (RAG)
- sphinx + myst-nb (docs)
- pytest
- ruff
- mypy (optional type checking)

## 4. (Optional) Install MolSysSuite tools (recommended: separate agent environment)

If you plan to develop or test tool execution with MolSysSuite, it is recommended to use a dedicated environment for the local agent
so you do not contaminate the server/vLLM environment.

Example:

```bash
conda create -n molsys-agent python=3.12 -c conda-forge
conda activate molsys-agent
conda install -c uibcdf -c conda-forge molsysmt
pip install -e ".[cli]"
```

If you still prefer a single manual environment (Option B), you can install a
subset of tools from the laboratory’s `uibcdf` conda channel:

```bash
conda install -c uibcdf molsysmt molsysviewer topomt elastnet pharmacophoremt
```

Select only the subset you actually need.

## 5. Run the smoke tests

To verify that the basic skeleton imports correctly:

```bash
pytest
```

## 6. Run the local servers (manual)

For interactive local work (e.g. running the Sphinx widget against a live backend),
use the helper scripts:

- Engine server (vLLM): `./dev/run_model_server.sh`
- Chat API (RAG): `./dev/run_chat_api.sh`

See `dev/RUNBOOK_VLLM.md` for the validated vLLM environment and a sample model YAML.

Example (two terminals):

```bash
# Terminal 1
cp server/model_server/config.example.yaml dev/model_server.local.yaml
# Edit dev/model_server.local.yaml and set:
# - model.local_path: "<ABS PATH>/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
./dev/run_model_server.sh --config dev/model_server.local.yaml --cuda-devices 0 --warmup
```

```bash
# Terminal 2 (if serving docs on 8080)
./dev/run_chat_api.sh --engine-url http://127.0.0.1:8001 --cors http://127.0.0.1:8080,http://localhost:8080
```

You should see all tests passing.

### 6.1 Public docs demo (uibcdf.org)

When the docs are published under `https://www.uibcdf.org/...`, the widget defaults to backend mode and
targets `https://api.uibcdf.org/v1/chat`. To reproduce the current public demo on this host:

```bash
MOLSYS_AI_ENGINE_URL=http://127.0.0.1:8001 \
MOLSYS_AI_PROJECT_INDEX_DIR=server/chat_api/data/indexes \
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
MOLSYS_AI_CORS_ORIGINS=https://www.uibcdf.org \
./dev/run_chat_api.sh --host 127.0.0.1 --port 8000
```

Then open the published page (example):

- `https://www.uibcdf.org/molsys-ai-server/`

The API endpoint should respond:

```bash
curl -fsS https://api.uibcdf.org/healthz
```

## 7. Run the model server

### 6.1 Stub backend (default)

If you have **no** `server/model_server/config.yaml`, or if it contains:

```yaml
model:
  backend: "stub"
```

the server will run in stub mode (echoing the last user message) and **no**
model weights will be loaded.

Launch the FastAPI model server:

```bash
PYTHONPATH=server:client python -m uvicorn model_server.server:app --host 127.0.0.1 --port 8001 --reload
```

Note: if you installed only `pip install -e .`, you must install server deps too:

```bash
pip install -e ".[server]"
```

By default this will start on `http://127.0.0.1:8001`.

You can inspect the OpenAPI docs at:

- http://127.0.0.1:8001/docs

In this mode, the `/v1/engine/chat` endpoint returns a stubbed reply.

### 6.2 vLLM backend (current baseline)

To use a real model via vLLM:

1. Set up the vLLM environment and install a system CUDA Toolkit (`nvcc`).
2. Download the AWQ model locally (Hugging Face SSH + `git-lfs`).
3. Create `server/model_server/config.yaml` based on `server/model_server/config.example.yaml`.

For the full, validated procedure, see:

- `dev/RUNBOOK_VLLM.md`

## 8. Try the CLI

With the virtual environment active:

```bash
molsys-ai --help
```

Log in (store API key locally):

```bash
molsys-ai login
```

Chat (single message):

```bash
molsys-ai chat -m "Hello from MolSys-AI"
```

## 9. Code style and tooling

- Linting and formatting: `ruff`
- Testing: `pytest`
- Optional static type checking: `mypy`

Examples:

```bash
ruff check .
pytest
mypy client/agent server/model_server server/rag server/chat_api client/cli
```

These commands are not yet enforced by CI, but are recommended during development.

## 10. Where to start hacking

Suggested starting points for development:

- Agent core:
  - `client/agent/core.py`
  - `client/agent/planner.py`
  - `client/agent/executor.py`

- Model server:
  - `server/model_server/server.py`
  - `server/model_server/config.example.yaml`

- RAG:
  - `server/rag/build_index.py`
  - `server/rag/retriever.py`
  - `server/rag/embeddings.py`

- CLI:
  - `client/cli/main.py`

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
