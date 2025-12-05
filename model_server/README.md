
# MolSys-AI Model Server (MVP)

This directory contains the FastAPI-based model server for MolSys-AI.

For the MVP, the server exposes a `/v1/chat` endpoint and can use:

- a **stub backend** that echoes the last user message (default), or
- a **llama.cpp backend** (via `llama_cpp-python`) that runs a local GGUF model
  when configured.

Configuration is read from a YAML file:

- By default: `model_server/config.yaml`.
- Or from the path in the environment variable `MOLSYS_AI_MODEL_CONFIG`.

See `config.example.yaml` for the expected structure.

## Running the stub backend (no real model)

If `config.yaml` does not exist, or if it sets:

```yaml
model:
  backend: "stub"
```

the server uses the echo stub and does not load any model:

```bash
uvicorn model_server.server:app --reload
```

The endpoint will respond with:

```text
[MolSys-AI stub reply] You said: ...
```

## Using a llama.cpp backend (outline)

To use a real model via `llama_cpp-python`:

1. Install `llama_cpp-python` in your environment.
2. Download the desired GGUF model (e.g. from Hugging Face under `uibcdf/`).
3. Create `model_server/config.yaml` based on `config.example.yaml`, for example:

   ```yaml
   model:
     backend: "llama_cpp"
     local_path: "/path/to/molsys-ai-qwen2p5-7b-proto.gguf"
     device: "cuda:0"
   ```

4. Start the server:

   ```bash
   uvicorn model_server.server:app --reload
   ```

The current `llama_cpp` integration is a minimal skeleton intended to be refined
according to hardware constraints and future ADRs (context window, sampling
parameters, etc.).

The docs chatbot backend (`docs_chat/backend.py`) will talk to whichever
backend is configured here via HTTP.
