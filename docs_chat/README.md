# MolSys-AI Docs Chat Backend

This directory contains the FastAPI backend used by the MolSys-AI documentation
chatbot. It is intended to power the small chat widget embedded in Sphinx /
MyST documentation sites (see `docs/` and `web_widget/`).

## Overview

- The backend exposes a single HTTP endpoint:
  - `POST /v1/docs-chat`
    - Request body: JSON with
      - `query: str` – user question.
      - `k: int` (optional, default `5`) – number of RAG documents to retrieve.
    - Response body:
      - `{"answer": "..."}` – model-generated answer.
- On startup it:
  - builds an in-memory RAG index from `.txt` documents,
  - uses `rag.build_index` and `rag.retriever` to populate the index.

## Configuration

The backend is configured via environment variables:

- `MOLSYS_AI_DOCS_DIR`
  - Directory containing `.txt` documents used for RAG.
  - Default: `docs_chat/data/docs` relative to this directory.

- `MOLSYS_AI_DOCS_INDEX`
  - Reserved for a future FAISS-based index path.
  - Currently not used; the index lives in memory.

- `MOLSYS_AI_MODEL_SERVER_URL`
  - Base URL of the MolSys-AI model server.
  - Default: `http://127.0.0.1:8000`.
  - The docs-chat backend uses this URL to call `POST /v1/chat` via
    `agent.model_client.HTTPModelClient`.

## Startup behaviour

On FastAPI startup, the handler:

- Calls `rag.build_index.build_index(DOCS_SOURCE_DIR, DOCS_INDEX_PATH)` to:
  - read all `*.txt` files under `DOCS_SOURCE_DIR`,
  - create `Document` objects with basic metadata (`{"path": ...}`),
  - populate the in-memory index for `rag.retriever.retrieve`.

When `POST /v1/docs-chat` is called:

1. The backend retrieves up to `k` documents using `rag.retriever.retrieve`.
2. It builds a context block with excerpts and their source paths.
3. It constructs a prompt with:
   - a `system` message describing the role of the assistant,
   - a `user` message containing documentation excerpts and the question.
4. It calls the model server (`/v1/chat`) via `HTTPModelClient`.
5. It returns the model's reply as `{"answer": "..."}`.

If the model server is not reachable or fails, a 500 error is returned.

## Running the docs-chat backend

In a development environment:

```bash
uvicorn docs_chat.backend:app --reload
```

By default this will listen on `http://127.0.0.1:8000` and use:

- `docs_chat/data/docs` as the source for `.txt` files (if the directory exists),
- `http://127.0.0.1:8000` as the model server URL.

## Integration with the web widget

- The Sphinx docs pilot (`docs/`) includes:
  - `docs/_static/molsys_ai_config.js`
  - `web_widget/molsys_ai_widget.js`

- Configuration in `molsys_ai_config.js`:

  ```js
  window.molsysAiChatConfig = window.molsysAiChatConfig || {
    mode: "placeholder", // or "backend"
    backendUrl: window.location.origin.replace(/\/+$/, "") + "/v1/docs-chat",
  };
  ```

- When `mode: "backend"` is set, the widget:
  - sends user messages to `/v1/docs-chat`,
  - displays the `answer` field returned by this backend.

## Future work

- Replace the simple substring-based retrieval in `rag.retriever` with a
  proper embedding-based retriever and FAISS index.
- Add authentication / rate limiting if needed for public deployments.
- Improve prompt construction for better use of documentation context.

