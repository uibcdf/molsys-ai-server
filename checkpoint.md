# MolSys-AI – Checkpoint 2025-12-05

This checkpoint summarises the current state of the MolSys-AI repository and
outlines the next steps. It reflects the work done up to 2025-12-05.

## 1. Packaging, environment and tooling

- **Packaging**
  - `pyproject.toml` added using `setuptools.build_meta` as build backend.
  - Project metadata:
    - `name = "molsys-ai"`, `version = "0.1.0"`.
    - Runtime dependencies: `fastapi`, `uvicorn`, `pydantic`, `rich`, `requests`, `PyYAML`.
    - Dev extra: `pytest`, `ruff`, `mypy`, `sphinx`, `pydata-sphinx-theme`.
  - `project.scripts` defines a console entrypoint:
    - `molsys-ai = "cli.main:main"`.
  - Packages discovered via `setuptools.find_packages` include:
    - `agent`, `cli`, `model_server`, `rag`, `docs_chat`.

- **Environment**
  - `environment.yml` defines a conda/mamba environment `molsys-ai`:
    - Python 3.12.
    - MolSys* ecosystem from `uibcdf` channel:
      - `molsysmt`, `molsysviewer`, `topomt`, `elastnet`, `pharmacophoremt`.
    - Core Python dependencies:
      - `fastapi`, `uvicorn`, `pydantic`, `rich`, `requests`, `pytest`, `ruff`, `mypy`, `sphinx`, `pydata-sphinx-theme`, `pip`.
  - `requirements-dev.txt` mirrors the dev dependencies for non-conda setups.

## 2. Model server and CLI

- **Model server (`model_server/`)**
  - `model_server/server.py` now supports:
    - Config-driven backend selection via YAML:
      - Config path:
        - Default: `model_server/config.yaml`.
        - Or `MOLSYS_AI_MODEL_CONFIG` environment variable.
      - If no config exists:
        - Falls back to `{"model": {"backend": "stub"}}`.
    - Backends:
      - `StubBackend`:
        - Echoes the last user message.
        - Used when `model.backend == "stub"` or config is missing.
      - `LlamaCppBackend` (skeleton):
        - Activated when `model.backend == "llama_cpp"`.
        - Requires `llama_cpp-python` installed.
        - Expects `model.local_path` pointing to a local GGUF file.
        - Very simple prompt formatting (concatenated roles) and call:
          - `self._llama(prompt, max_tokens=256, stop=["USER:", "ASSISTANT:"])`.
    - Two cached helpers:
      - `load_config()` (reads YAML or returns stub config).
      - `get_model_backend()` (instantiates backend once, cached via `lru_cache`).
    - `/v1/chat` endpoint:
      - Uses the configured backend’s `chat(messages)` method.
      - Wraps backend errors as `HTTPException(500, detail=str(exc))`.
  - `model_server/config.example.yaml` updated:
    - Documents `backend: "stub"` and `backend: "llama_cpp"`.
    - Adds `local_path` as required for `llama_cpp`.
  - `model_server/README.md` documents:
    - Stub vs llama.cpp backends.
    - Config file location and environment override.
    - Outline for using a real GGUF model.

- **CLI (`cli/main.py`)**
  - Now supports:
    - `--version`:
      - Prints `MolSys-AI CLI (MVP)`.
    - `--message/-m "TEXT"`:
      - Sends a single user message using the agent.
    - `--server-url URL`:
      - When provided, uses `HTTPModelClient(base_url=URL)`.
      - Otherwise uses `EchoModelClient` (local stub).
  - Behaviour:
    - Builds `messages = [{"role": "user", "content": ...}]`.
    - Calls `MolSysAIAgent.chat(messages)`:
      - This path currently bypasses the planner; it is a direct call to the model client.
    - On `RuntimeError` from the HTTP client:
      - Prints an error to `stderr`.
      - Returns exit code `1`.
    - Default message when no arguments are given:
      - Explains usage of `--message` and `--server-url`.
  - `dev/DEV_GUIDE.md` briefly documents:
    - `python -m cli.main --message "Hello from MolSys-AI"`.
    - `molsys-ai --server-url http://127.0.0.1:8000 --message "Hello from MolSys-AI"`.

## 3. RAG and docs chatbot

- **RAG layer (`rag/`)**
  - `rag/retriever.py`:
    - `Document` dataclass:
      - `content: str`.
      - `metadata: Dict[str, Any]` (e.g. `{"path": ...}`).
    - In-memory index:
      - `_INDEX: List[Document]`.
      - `set_index(documents)` to replace the index.
    - `retrieve(query: str, k: int = 5) -> List[Document]`:
      - Simple ranking based on case-insensitive substring counts of `query` in `content`.
      - Returns up to `k` documents ordered by score.
  - `rag/embeddings.py`:
    - `EmbeddingModel` abstract base class with `embed(texts: List[str])`.
    - `DummyEmbeddingModel`: returns lists of `[0.0]`.
    - `get_default_embedding_model()` returns the dummy model.
  - `rag/build_index.py`:
    - `build_index(source_dir: Path, index_path: Path)`:
      - Reads `*.txt` files under `source_dir`.
      - Creates `Document` objects with `metadata={"path": str(path)}`.
      - Calls `set_index(docs)` to populate the in-memory index.
      - If `source_dir` does not exist, sets an empty index.
      - `index_path` is currently unused (reserved for future FAISS persistence).
  - `rag/README.md` updated to describe:
    - The MVP behaviour of `build_index`, `retrieve`.
    - How `docs_chat` triggers index building and uses RAG.
    - A quickstart to test RAG + `/v1/docs-chat`.

- **Docs chatbot backend (`docs_chat/backend.py`)**
  - FastAPI app `app = FastAPI(title="MolSys-AI Docs Chat Backend (MVP)")`.
  - Index building on startup:
    - Environment:
      - `MOLSYS_AI_DOCS_DIR`: docs source directory (default: `docs_chat/data/docs`).
      - `MOLSYS_AI_DOCS_INDEX`: reserved for future FAISS index path.
    - Startup handler:
      - Calls `build_index(DOCS_SOURCE_DIR, DOCS_INDEX_PATH)` to populate RAG index.
  - `/v1/docs-chat` endpoint:
    - Request model: `DocsChatRequest { query: str, k: int = 5 }`.
    - Response model: `DocsChatResponse { answer: str }`.
    - Behaviour:
      - Calls `retrieve(query, k)` to get `Document`s.
      - Builds a context block with excerpts and their `path`.
      - Constructs a prompt:
        - `system` message: instructs the model to answer based on the excerpts.
        - `user` message: includes the documentation excerpts and the question.
      - Creates `HTTPModelClient(base_url=MODEL_SERVER_URL)` where:
        - `MODEL_SERVER_URL` is `MOLSYS_AI_MODEL_SERVER_URL` or `http://127.0.0.1:8000`.
      - Calls `client.generate(messages)` and returns the reply as `answer`.

## 4. Sphinx documentation pilot and JS widget

- **Sphinx docs (`docs/`)**
  - `docs/conf.py`:
    - Uses `pydata_sphinx_theme` as `html_theme`.
    - Enables MyST via `extensions = ["myst_parser"]`.
    - `html_static_path` includes:
      - `docs/_static` and `../web_widget`.
    - `html_js_files`:
      - `molsys_ai_config.js` (default widget config).
      - `molsys_ai_widget.js` (chat widget).
  - `docs/index.md` (MyST):
    - Introduces the “MolSys-AI Documentation Pilot”.
    - Under “AI helper (pilot)” embeds the chat container:
      ```md
      ```{raw} html
      <div id="molsys-ai-chat"></div>
      ```
      ```
    - Explains how to build the docs:
      - `sphinx-build -b html docs docs/_build/html`.

- **Widget configuration (`docs/_static/molsys_ai_config.js`)**
  - Defines a default global config:
    ```js
    window.molsysAiChatConfig = window.molsysAiChatConfig || {
      mode: "placeholder",
      backendUrl: window.location.origin.replace(/\/+$/, "") + "/v1/docs-chat",
    };
    ```
  - Modes:
    - `"placeholder"` (default): only local, fixed responses.
    - `"backend"`: call the `/v1/docs-chat` endpoint and show its responses.

- **Widget implementation (`web_widget/molsys_ai_widget.js`)**
  - On load:
    - Checks if `div#molsys-ai-chat` exists:
      - If yes: renders the chat box inline in that container.
      - If no: creates a floating “AI” button in bottom-right that toggles a panel with the chat box.
  - Chat behaviour:
    - Shows an initial assistant message explaining that the bot is not ready yet.
    - When the user sends a message:
      - Always appends the user message.
      - If `mode === "backend"`:
        - Sends `fetch(backendUrl, POST, {"query": text, "k": 5})`.
        - Displays `data.answer` (or a friendly error message).
      - Else (`mode === "placeholder"`):
        - Displays a fixed message:
          - “This bot is still under development. In future versions it will answer questions about the MolSys* documentation.”

## 5. Agent core, planner, executor and notebook helpers

- **Model clients (`agent/model_client.py`)**
  - `ModelClient` abstract base class with `generate(messages) -> str`.
  - `EchoModelClient`:
    - Returns a stub reply echoing the last user message.
  - `HTTPModelClient`:
    - Sends `POST {base_url}/v1/chat` with `{"messages": [...]}`.
    - Expects JSON with a string `content`.
    - Raises `RuntimeError` on HTTP or parsing errors.

- **Planner (`agent/planner.py`)**
  - `Plan` dataclass:
    - `use_rag: bool`.
    - `use_tools: bool` (not used yet).
  - `SimplePlanner`:
    - `decide(messages, force_rag=False) -> Plan`:
      - If `force_rag`: sets `use_rag=True`.
      - Else: uses a naive heuristic:
        - If the last user message starts with “what is ”, “how to ”, or contains “docs”, then `use_rag=True`.
        - Otherwise `use_rag=False`.
    - `build_model_messages(messages, plan, rag_k=5) -> List[Dict[str, str]]`:
      - If `plan.use_rag`:
        - Extracts latest user message as query.
        - Calls `retrieve(query, k=rag_k)` to get docs.
        - Builds a new pair of messages (`system` + `user`) with excerpts + question.
      - If not, returns the original `messages`.

- **Executor (`agent/executor.py`)**
  - `ToolExecutor`:
    - In-memory registry of tools:
      - `register(name, func, description="")`.
      - `execute(name, *args, **kwargs)`.
    - Returns the result of the underlying Python function.
    - No argument validation or sandboxing yet (future work per ADR-010).
  - `create_default_executor()`:
    - Registers `molsysmt.dummy_info` mapping to `tool_dummy_molsysmt_info()` in `agent.tools.molsysmt_tools`.
    - Used as the default executor for the agent.

- **Agent core (`agent/core.py`)**
  - `MolSysAIAgent`:
    - Initialiser:
      - Accepts `model_client`, optional `planner` (`SimplePlanner` by default) and optional `executor` (`create_default_executor()` by default).
    - `chat(messages)`:
      - Directly delegates to `model_client.generate(messages)`.
    - `chat_with_planning(messages, force_rag=False)`:
      - Uses `planner.decide(...)` to obtain a `Plan`.
      - Uses `planner.build_model_messages(...)` to transform messages (RAG vs no RAG).
      - Calls `model_client.generate` with the planned messages.

- **Notebook helpers (`agent/notebook.py`)**
  - `create_notebook_agent(server_url=None, use_planner=True) -> MolSysAIAgent`:
    - Creates an agent using `HTTPModelClient` with:
      - `server_url` or `MOLSYS_AI_MODEL_SERVER_URL` or `http://127.0.0.1:8000`.
  - `NotebookChatSession`:
    - Maintains `messages: List[Dict[str, str]]`.
    - `ask(question, force_rag=False) -> str`:
      - Appends user message, calls `agent.chat_with_planning`, appends assistant reply, returns reply.
  - **Safe notebook workflow generation**:
    - `NotebookCellSpec`:
      - `cell_type: "markdown" | "code"`.
      - `source: str`.
    - `create_workflow_notebook(cells, output_path) -> Path`:
      - Writes a new `.ipynb` to `output_path` with the provided cells.
      - The current notebook is not modified.
  - **Placeholder for in-place editing**:
    - `inject_workflow_into_current_notebook(...)`:
      - Raises `NotImplementedError` with a clear message.
      - Intended for future JupyterLab/VS Code integration.

## 6. Training skeleton (`train/`)

- `train/README.md`:
  - Describes the training area for LoRA/QLoRA.
  - Clarifies separation between training (Node B) and serving (Node A).
- Subdirectories:
  - `train/configs/README.md`:
    - Suggests YAML/TOML experiment configs (base model, LoRA params, data, outputs).
  - `train/scripts/`:
    - `__init__.py` with docstring.
    - `train_lora.py`:
      - CLI with `--config`.
      - Prints help and a message indicating training is not implemented yet.
  - `train/jobs/README.md`:
    - Intended for SLURM/bash job scripts for training runs.
  - `train/notebooks/README.md`:
    - For exploratory notebooks; core logic should live in `train/scripts/`.
  - `train/data/README.md`:
    - States that real datasets should not live in the repo; only small examples/metadata.
- `dev/ARCHITECTURE.md` updated to include `train/` in the high-level component list.

## 7. Notebook integration ADR

- `dev/decisions/ADR-013.md`:
  - Defines the strategy for notebook integration:
    - Level 1 (implemented): safe generation of new notebooks via `create_workflow_notebook`.
    - Level 2 (future): in-place editing via `inject_workflow_into_current_notebook`, initially for JupyterLab (and later VS Code), with explicit user confirmation and visual feedback.
  - Justifies a two-tier approach:
    - Safety and reproducibility first.
    - Clear upgrade path for advanced users.

## 8. Internal tests

- `tests/test_smoke.py` updated to import the key modules:
  - `agent.core`, `agent.model_client`, `agent.planner`, `agent.executor`, `agent.notebook`.
  - `rag.build_index`, `rag.embeddings`, `rag.retriever`.
  - `model_server.server`, `docs_chat.backend`, `cli.main`.
- Purpose:
  - Ensure that changes to the skeleton do not break basic imports.

## 9. Known gaps and items not fully documented yet

These are aspects that exist in code but are only partially or indirectly
documented in Markdown files and may need more explicit coverage later:

- **CLI usage and subcommands**
  - There is no dedicated `cli/README.md` yet.
  - Current behaviour is briefly documented in `dev/DEV_GUIDE.md`, but a
    focused CLI doc would help (flags, exit codes, examples).

- **Docs chatbot backend details**
  - `docs_chat/backend.py` behaviour is described mainly via docstrings and the
    RAG README.
  - A short `docs_chat/README.md` could clarify how it is expected to be
    deployed alongside the model server and how the widget interacts with it.

- **Notebook helpers and workflows**
  - The high-level strategy is documented in ADR-013 and the code is in
    `agent/notebook.py`, but there is no user-facing "how-to" Markdown yet.
  - A future `dev/NOTEBOOKS.md` or similar could gather examples for
    `NotebookChatSession`, `create_workflow_notebook`, and planned in-place
    editing.

These gaps do not block development but are good candidates for future
documentation passes once the corresponding features stabilise.

## 10. Next steps (short-term roadmap)

1. **Agent integration in CLI and docs chatbot**
   - Decide when/how to use `chat_with_planning` in the CLI (e.g. a flag for
     RAG-enabled answers).
   - Optionally refactor docs chatbot `docs_chat` to reuse planner logic
     instead of duplicating prompt building.

2. **First concrete MolSys* tools**
   - Implement the first real tools in `agent.tools.molsysmt_tools` and
     register them in `ToolExecutor`.
   - Define the minimal interface for TopoMT/MolSysViewer tools expected by
     the agent.

3. **Training configs and dataset design**
   - Add 1–2 concrete configs under `train/configs/` that describe:
     - base model,
     - LoRA/QLoRA hyperparameters,
     - dataset manifests (paths, formats).
   - Draft a document (possibly `train/DESIGN.md`) describing the first
     training dataset(s) (e.g. MolSysMT Q&A, workflows).

4. **RAG improvements**
   - Replace the simple substring scoring in `rag.retriever` with a more robust
     embedding-based retrieval once an embedding model is chosen.
   - Decide on an initial FAISS-based index format and update
     `build_index`/`retrieve` accordingly.

5. **Sphinx/docs integration**
   - Optionally refine the chat widget styling for `pydata_sphinx_theme`
     (e.g. dedicated launcher icon, theme-aware colors).
   - Consider a small Sphinx extension (or theme override) to inject the
     widget consistently across pages without requiring manual HTML blocks.

6. **Notebook UX**
   - Add examples (in a separate docs file or notebooks) showing:
     - interactive chat from a notebook,
     - generation of a workflow notebook via `create_workflow_notebook`.
   - Start exploring the JupyterLab APIs needed to implement
     `inject_workflow_into_current_notebook` safely (no code yet).

This checkpoint is intended as a reference to align future work and to make it
easy for new contributors to understand the current state of the project.

