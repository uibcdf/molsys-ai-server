
# MolSys-AI Constraints and Assumptions

This document lists current constraints and assumptions for the initial phase of MolSys-AI.

## Hardware

- Two nodes available:
  - Node A: has a static IP address, used for **serving** models and APIs.
  - Node B: no static IP address, used for **training** (LoRA/QLoRA) and experimentation.
- Each node has:
  - 3 × NVIDIA RTX 2080 Ti GPUs (11 GB VRAM each).
- The design should:
  - work well on this hardware,
  - allow future migration to larger GPUs without major refactors.

## Model

- Foundational base model: **Llama-3.1-8B-Instruct** (see ADR-017).
- Strategy:
  - Phase 1: use the base model with RAG + local tools (executed by the client-side agent).
  - Phase 2: add LoRA/QLoRA specialization for MolSysSuite.
- All models and LoRAs are published under the Hugging Face **organization `uibcdf`**.

## Serving

- MVP backend:
  - HTTP API using **FastAPI**.
  - Model backend: **vLLM** (see ADR-015).
  - Quantization format: **AWQ** (see ADR-016).
- Operational note:
  - A system CUDA Toolkit is required for `nvcc` so that vLLM can JIT-compile
    kernels (FlashInfer).

## RAG

- MVP index: a pickled list of documents + embeddings (see `server/rag/README.md`).
- A FAISS-based vector store is planned but not yet implemented.
- The API in `server/rag/` must allow swapping the backend later without touching the
  surrounding services.

## Environments

- Keep the vLLM inference environment minimal and isolated (pip + CUDA wheels + system `nvcc`).
- Do not mix MolSysSuite toolchains into the vLLM inference environment (some tools pull extra CUDA-related stacks).
- Local tool execution belongs to the client-side agent (`molsys-ai agent`) and should run
  in a separate environment when needed.

## Deployment reality

- The target public API is `https://api.uibcdf.org`.
- Inbound `80/443` may be blocked upstream by the data-center firewall; opening ports can take
  days and requires a request (see `dev/FIREWALL_PORT_REQUEST_TEMPLATE.md`).

## Non-goals (for the MVP)

- Training models from scratch.
- Supporting very large models (e.g. 70B) on current hardware.
- Massive multi-user scale; initial focus is on research and power-user workflows.

These constraints should guide implementation choices and are documented in more detail in the ADR files.
