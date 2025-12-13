# MolSys-AI Model Server (MVP)

This directory contains the FastAPI-based model server for MolSys-AI.

For the MVP, the server exposes a `/v1/chat` endpoint and can use:

- a **stub backend** that echoes the last user message (default), or
- a **vLLM backend** that runs a local (or Hub) model, typically an AWQ
  quantized checkpoint (current baseline).

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

## Using the vLLM backend (current baseline)

The vLLM backend is enabled with:

```yaml
model:
  backend: "vllm"
```

For the validated, end-to-end procedure (CUDA 12.9 + `nvcc`, model download via
Hugging Face SSH + `git-lfs`, and a stable configuration for 11 GB GPUs), see:

- `dev/RUNBOOK_VLLM.md`

### Key configuration fields

- `model.local_path`
  - Local model directory (recommended for production).
  - A Hugging Face ID may also work, but local is preferred on HPC nodes.
- `model.quantization`
  - Use `"awq"` for the current baseline model.
- `model.tensor_parallel_size`
  - Use `1` for a single-GPU baseline.
  - Use `3` to spread the model across `CUDA_VISIBLE_DEVICES=0,1,2` if needed.
- `model.max_model_len`
  - Context window limit (tokens). Must be set explicitly on 11 GB GPUs.
- `model.gpu_memory_utilization`
  - Fraction of GPU memory vLLM is allowed to use.
- `model.enforce_eager`
  - Stability-first toggle. When `true`, vLLM disables `torch.compile` and CUDA
    graphs, reducing OOM risk during warmup on 11 GB GPUs.

### Important note (chat formatting)

The current MVP vLLM backend treats the last `user` message as a plain prompt.
It does not apply a model-specific chat template yet. This is sufficient for:

- smoke tests,
- RAG-style prompts (excerpts + question in a single message).

It is not yet a full multi-turn chatbot implementation.

## Legacy llama.cpp notes (deprecated)

Older ADRs describe a llama.cpp / GGUF backend. The code may still contain
placeholders for it, but the current baseline is vLLM + AWQ (see ADR-015/016/017).

The docs chatbot backend (`docs_chat/backend.py`) talks to whichever backend is
configured here via HTTP.
