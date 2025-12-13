# Checkpoint: MolSys-AI Project Status

This document summarizes the current status of the project, its objectives, the progress made, and the next steps.

## 1. Overall Goals (What we want)

The three main long-term objectives for this project are:

1.  **Model Retraining:** Retrain a language model to have a deep understanding of the MolSysSuite ecosystem tools.
2.  **Documentation Chatbot:** Integrate a chatbot into the documentation web pages (generated with Sphinx) of `molsysmt`, `molsysviewer`, etc., to answer questions about using the tools.
3.  **Autonomous CLI Agent:** Develop a command-line agent (`molsys-ai`) capable of executing autonomous workflows using the MolSysSuite tools at the user's request.

## 2. Progress Made (What we've done)

Significant progress has been made in establishing the inference stack with GPU acceleration and selecting the foundational model.

### Phases 1-6 (RAG System Validation):

- Initial project setup, first tool (`get_info`) implementation, and RAG system construction were completed.
- Initial LLM integration (Qwen3-8B-Q6_K.gguf, CPU-only due to compilation issues) and validation of the end-to-end RAG flow via `curl` were successful.

### Phase 7: Hardware Upgrade & New Inference Stack Integration

- **Hardware Upgrade:** Both inference and training nodes now feature 3x NVIDIA RTX 2080 Ti GPUs.
- **Inference Engine Pivot (ADR-015):** Decision to move from `llama-cpp-python` to `vLLM` for production, leveraging Tensor Cores and PagedAttention.
- **Model Format Standard (ADR-016):** Adoption of AWQ as the standard quantization format for `vLLM` on the new hardware.
- **Foundational Model Selection (ADR-017):** After re-evaluation based on RAG & fine-tuning strategy, `meta-llama/Llama-3.1-8B-Instruct` was selected as the optimal base model.
- **Model Deployment:** `Meta-Llama-3.1-8B-Instruct-AWQ-INT4` has been successfully duplicated to the `uibcdf` organization on Hugging Face.

### Environment Setup Challenges & Resolution

- **Initial `vLLM` Integration:** Encountered significant dependency conflicts when attempting to install `vLLM` via `conda`, primarily related to Python and PyTorch versions, and `huggingface_hub` compatibility with `transformers`.
- **Solution:** Adopted a robust hybrid strategy:
    - **`conda`:** Provides a minimal Python 3.12 environment.
    - **`pip`:** Installs `vLLM` with CUDA-enabled wheels (`cu129`).

### Smoke Test Results (vLLM + AWQ)

- **Hugging Face access:** Verified authentication via SSH (`git@hf.co`) and `git-lfs`.
- **Model download:** `uibcdf/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` cloned locally under `models/` and LFS files pulled successfully.
- **System CUDA Toolkit:** Installed CUDA 12.9 `nvcc` at system level and exported:
  - `CUDA_HOME=/usr/local/cuda`
  - `PATH=$CUDA_HOME/bin:$PATH`
  This is required for FlashInfer JIT compilation.
- **Single-GPU inference:** `model_server` + vLLM successfully loads the model and answers `/v1/chat`.
- **Multi-turn chat support:** the vLLM backend now applies the model chat template (via `transformers`), so `/v1/chat`
  accepts full `messages` history (not just the last user prompt).
- **Stable baseline config (11 GB GPU):**
  - `max_model_len=8192`
  - `gpu_memory_utilization=0.80`
  - `enforce_eager=true` (stability-first; avoids OOM during compilation/warmup)
  - Observed VRAM usage during long-prompt tests: ~9.4–9.8 GiB on GPU0.
- **Reproducible smoke runner:** `dev/smoke_vllm.sh` starts the server, performs minimal + multi-turn requests, and
  cleans up on exit (including best-effort cleanup of any stray `VLLM::EngineCore` processes).
- **Docs chatbot end-to-end (backend):** validated `docs_chat` → `model_server`:
  - `POST /v1/docs-chat` with legacy `query` returns `OK`,
  - `POST /v1/docs-chat` with `messages` history preserves context (multi-turn) and returns `Diego` in the test.
- **RAG embeddings validation:** `docs_chat` was also validated with `sentence-transformers` installed (v5.2.0), building a
  small Markdown index and answering a retrieval-backed question correctly (example: returns `1VII` for a synthetic doc).

## 3. Current Status

The foundational model (`Llama-3.1-8B-Instruct-AWQ-INT4`) is deployed on Hugging Face (`uibcdf` organization). The vLLM
backend for `model_server` is working on RTX 2080 Ti GPUs (CUDA 12.9 `nvcc` + vLLM `cu129` wheels) and supports multi-turn
chat formatting.

Repository documentation has been consolidated around:

- `AGENTS.md` as the single “how to work here” entry point,
- `dev/RUNBOOK_VLLM.md` as the inference runbook,
- a minimal `environment.yml` for general development (install Python deps via `pip install -e ".[dev]"`).

The docs chatbot stack is now validated end-to-end at the backend/API level, but
the browser/widget integration is not yet validated on a real Sphinx site:

- `docs_chat/backend.py` now accepts `messages` (recommended) in addition to legacy `query`,
- `web_widget/molsys_ai_widget.js` now keeps conversation history and sends it as `messages` when `mode: "backend"`.
  This has been validated at the backend level (HTTP requests). Browser/widget integration still needs a real docs site
  deployment decision (same-origin vs. separate service + CORS).

## 4. Next Steps

The immediate next steps focus on validating the docs chatbot and then iterating on quality:

1.  **Widget integration smoke:** Run `model_server` (vLLM) + `docs_chat` together and verify that the web widget (Sphinx)
    can talk to `POST /v1/docs-chat` in multi-turn mode (`messages` history).
2.  **Docs deployment integration:** Decide how the widget and backend will be deployed for real docs sites (same-origin vs
    separate service; CORS; rate-limiting if needed).
3.  **Improve RAG quality:**
    - **Prompt Engineering:** Refine the prompt sent to the LLM to make more effective use of the documentation context.
    - **Chunking:** Evaluate and improve the document splitting strategy (`chunking`) in `rag/build_index.py`.
4.  **Benchmarking suite:** Implement benchmark tasks (ADR-011/017) to evaluate changes in retrieval/prompting/models.
5.  **Expand agent tools:**
    - Begin implementing new real tools from the `molsysmt` ecosystem in `agent/tools/`, following the `ROADMAP.md` and ADRs.
6.  **Multi-GPU note (optional):** Only if we need more context or larger models, test `tensor_parallel_size=3` across the
    3× RTX 2080 Ti GPUs.
7.  **Prepare fine-tuning dataset (Roadmap v0.5):**
    - Start collecting and structuring data (code examples, question-answer pairs) for the future fine-tuning of the model.
