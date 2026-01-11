# Getting started

To build this documentation locally, from the repository root:

```bash
sphinx-build -b html docs docs/_build/html
```

The generated HTML will live under `docs/_build/html`.

Note: the docs build prefers `myst-nb` when installed (with a fallback to
`myst-parser`).

## Widget end-to-end smoke (local)

This validates the full path: Sphinx page → web widget → `server/chat_api` → `server/model_server`.

Recommended: use the bundled runner (builds docs, starts the servers, validates CORS, and prints the final URL):

```bash
./dev/smoke_widget.sh
```

1) Start the model server (vLLM) on `8001` (see `dev/RUNBOOK_VLLM.md`).

2) Start the chat API on `8000` (enable CORS for the docs origin):

```bash
MOLSYS_AI_ENGINE_URL=http://127.0.0.1:8001 \
MOLSYS_AI_CORS_ORIGINS=http://127.0.0.1:8080 \
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
./dev/run_chat_api.sh --engine-url http://127.0.0.1:8001 --cors http://127.0.0.1:8080
```

3) Build and serve this Sphinx site:

```bash
sphinx-build -b html docs docs/_build/html
python -m http.server 8080 --directory docs/_build/html
```

4) Open:

- `http://127.0.0.1:8080/?molsys_ai_mode=backend&molsys_ai_backend_url=http://127.0.0.1:8000/v1/chat`

You should see the widget send requests to the backend and get real answers.

If you need placeholder mode (no backend calls), open:

- `http://127.0.0.1:8080/?molsys_ai_mode=placeholder`

## Public demo (uibcdf.org)

The published pilot is available at:

- `https://www.uibcdf.org/molsys-ai-server/`

The widget defaults to backend mode and automatically targets:

- `https://api.uibcdf.org/v1/chat`

To reproduce the public setup on this host:

1) Start the model engine locally (`127.0.0.1:8001`).
2) Start `chat_api` with CORS allowing the docs origin:

```bash
MOLSYS_AI_ENGINE_URL=http://127.0.0.1:8001 \
MOLSYS_AI_PROJECT_INDEX_DIR=server/chat_api/data/indexes \
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
MOLSYS_AI_CORS_ORIGINS=https://www.uibcdf.org \
./dev/run_chat_api.sh --host 127.0.0.1 --port 8000
```

3) Ensure `https://api.uibcdf.org/healthz` responds (currently via a Cloudflare tunnel).
4) Open the published page and send a question in the widget.
