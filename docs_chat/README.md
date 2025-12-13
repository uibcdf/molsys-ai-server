# MolSys-AI Docs Chat Backend

This directory contains the FastAPI backend used by the MolSys-AI documentation
chatbot. It is intended to power the small chat widget embedded in Sphinx /
MyST documentation sites (see `docs/` and `web_widget/`).

## Overview

- The backend exposes a single HTTP endpoint:
  - `POST /v1/docs-chat`
    - Request body: JSON with
      - Either:
        - `messages: [{"role": "...", "content": "..."}]` – full conversation history (recommended).
        - `query: str` – single-turn question (backwards-compatible).
      - `k: int` (optional, default `5`) – number of RAG documents to retrieve.
    - Response body:
      - `{"answer": "..."}` – model-generated answer.
- On startup it:
  - builds an embedding-based RAG index from Markdown documents,
  - uses `rag.build_index` and `rag.retriever` to populate the index.

## Configuration

The backend is configured via environment variables:

- `MOLSYS_AI_DOCS_DIR`
  - Directory containing `*.md` documents used for RAG.
  - Default: `docs_chat/data/docs` relative to this directory.

- `MOLSYS_AI_DOCS_INDEX`
  - Path where the built index is stored (pickle file).
  - Default: `docs_chat/data/rag_index.pkl`.

- `MOLSYS_AI_MODEL_SERVER_URL`
  - Base URL of the MolSys-AI model server.
  - Default: `http://127.0.0.1:8001`.
  - The docs-chat backend uses this URL to call `POST /v1/chat` via
    `agent.model_client.HTTPModelClient`.

## Startup behaviour

On FastAPI startup, the handler:

- Calls `rag.build_index.build_index(DOCS_SOURCE_DIR, DOCS_INDEX_PATH)` to:
  - read all `*.md` files under `DOCS_SOURCE_DIR`,
  - create `Document` objects with basic metadata (`{"path": ...}`),
  - embed chunks using `sentence-transformers`,
  - store the index at `DOCS_INDEX_PATH` and load it into memory.

Note on embeddings:

- For good retrieval quality, install `sentence-transformers` and use the default model.
- For offline smoke tests, you can run without `sentence-transformers`; the code falls back
  to a lightweight hashing embedding model (set `MOLSYS_AI_EMBEDDINGS=hashing` to force it).
- If `DOCS_SOURCE_DIR` does not exist or contains no `*.md`, the server still starts, but retrieval returns no snippets.

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

- `docs_chat/data/docs` as the source for `*.md` files (if the directory exists),
- `http://127.0.0.1:8001` as the model server URL (run `model_server` on `8001`).

## End-to-end smoke (recommended)

Start the model server (vLLM) on `8001` (see `dev/RUNBOOK_VLLM.md`), then run:

```bash
mkdir -p /tmp/molsys_ai_docs_smoke
cat >/tmp/molsys_ai_docs_smoke/example.md <<'MD'
# Example

The example PDB id is **1VII**.
MD

MOLSYS_AI_MODEL_SERVER_URL=http://127.0.0.1:8001 \
MOLSYS_AI_DOCS_DIR=/tmp/molsys_ai_docs_smoke \
MOLSYS_AI_DOCS_INDEX=/tmp/molsys_ai_docs_smoke.pkl \
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
uvicorn docs_chat.backend:app --host 127.0.0.1 --port 8000
```

Then:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/docs-chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the example PDB id? Reply with just the id.","k":3}'
```

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
  - keeps conversation history in the browser and sends it as `messages`,
  - displays the `answer` field returned by this backend.

## Future work

- Replace the current pickle-based index with a FAISS-based vector store.
- Add authentication / rate limiting if needed for public deployments.
- Improve prompt construction for better use of documentation context.
