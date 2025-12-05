
# MolSys-AI Constraints and Assumptions

This document lists current constraints and assumptions for the initial phase of MolSys-AI.

## Hardware

- Two nodes available:
  - Node A: has a static IP address, used for **serving** models and APIs.
  - Node B: no static IP address, used for **training** (LoRA/QLoRA) and experimentation.
- Each node has:
  - 3 × NVIDIA GTX 1080 Ti GPUs (11 GB VRAM each).
- The design should:
  - work well on this hardware,
  - allow future migration to larger GPUs without major refactors.

## Model

- Initial base model: **Qwen2.5-7B-Instruct**.
- Strategy:
  - Phase 1: use the base model with RAG + tools.
  - Phase 2: add LoRA/QLoRA specialization for the MolSys* ecosystem.
- All models and LoRAs are published under the Hugging Face **organization `uibcdf`**.

## Serving

- MVP backend:
  - HTTP API using **FastAPI**.
  - Model backend: **llama.cpp / llama-cpp-python** as a hardware-friendly option.
- Future option:
  - **vLLM** as a more powerful backend when GPUs are upgraded.

## RAG

- MVP vector store: **FAISS local**.
- The API in `rag/` must allow swapping the backend later without touching the agent logic.

## Non-goals (for the MVP)

- Training models from scratch.
- Supporting very large models (e.g. 70B) on current hardware.
- Massive multi-user scale; initial focus is on research and power-user workflows.

These constraints should guide implementation choices and are documented in more detail in the ADR files.
