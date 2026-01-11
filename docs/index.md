# MolSys-AI Documentation Pilot

This is a small Sphinx-based documentation site used to prototype the
MolSys-AI chatbot integration.

This pilot embeds the MolSys-AI chatbot into Sphinx-generated documentation.
When the backend is available, the widget answers real questions and shows
citations via a “Sources” dropdown. Placeholder mode remains available for
offline demos or UI-only work.


## AI helper (pilot)

The box below is the MolSys-AI documentation assistant:

```{raw} html
<div id="molsys-ai-chat"></div>
```


## Getting started

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
