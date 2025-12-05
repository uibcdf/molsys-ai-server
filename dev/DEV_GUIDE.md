
# MolSys-AI – Developer Guide (Setup & First Steps)

This guide explains how to set up a development environment for MolSys-AI
and what to run first once the repository is cloned.

## 1. Python version

- Recommended: **Python 3.11**
- Minimum: **Python 3.10**

Check your version:

```bash
python --version
```

or, depending on your system:

```bash
python3 --version
```

## 2. Create and activate a virtual environment

From the root of the repository:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows (PowerShell/CMD)
```

You should see `(.venv)` in your shell prompt after activation.

## 3. Install development dependencies

Assuming you have `requirements-dev.txt` in the repo root:

```bash
pip install -r requirements-dev.txt
```

This will install:

- fastapi, uvicorn, pydantic
- rich
- pytest
- ruff
- mypy (optional type checking)

## 4. Run the smoke tests

To verify that the basic skeleton imports correctly:

```bash
pytest
```

You should see all tests passing.

## 5. Run the model server (MVP stub)

Launch the FastAPI model server (for the MVP it just echoes messages):

```bash
uvicorn model_server.server:app --reload
```

By default this will start on `http://127.0.0.1:8000`.

You can inspect the OpenAPI docs at:

- http://127.0.0.1:8000/docs

In the MVP, the `/v1/chat` endpoint returns a stubbed reply.

## 6. Try the CLI

With the virtual environment active:

```bash
python -m cli.main
```

For now this should print a simple message like:

```text
[MolSys-AI] CLI skeleton is in place. Agent wiring will come next.
```

Once the agent and model server are wired, this CLI will be extended
to open an interactive chat with the agent.

## 7. Code style and tooling

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

## 8. Where to start hacking

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

## 9. Language conventions

- Code, module names and comments: **English**.
- Internal docs in `dev/` may mix English/Spanish, but external/public
  user-facing docs should be consistent and preferably in English, unless
  otherwise decided.

## 10. Questions and open points

See `dev/OPEN_QUESTIONS.md` for items that are intentionally left open
for discussion (embedding model choice, backend refinements, etc.).
