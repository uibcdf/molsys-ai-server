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
- **Stable baseline config (11 GB GPU):**
  - `max_model_len=8192`
  - `gpu_memory_utilization=0.80`
  - `enforce_eager=true` (stability-first; avoids OOM during compilation/warmup)
  - Observed VRAM usage during long-prompt tests: ~9.4–9.8 GiB on GPU0.

## 3. Current Status

The foundational model (`Llama-3.1-8B-Instruct-AWQ-INT4`) is deployed on Hugging Face (`uibcdf` organization). The `vLLM` backend for the `model_server` has been implemented. The development environment is now being set up with a clean `conda` environment using the hybrid `conda`+`pip` strategy to ensure all dependencies, especially `vLLM` and its CUDA requirements, are correctly installed.

## 4. Next Steps

The immediate next steps are focused on bringing the new inference stack online and verifying its functionality:

1.  **Environment Setup Finalization:** Complete the creation of the clean `molsys-ai-vllm` `conda` environment using the hybrid `conda`+`pip` strategy.
2.  **Document the runbook:** Consolidate the working procedure in `dev/RUNBOOK_VLLM.md`.
3.  **Verify `vLLM` Load (Multi-GPU):** Optional: test `tensor_parallel_size=3` across all RTX 2080 Ti GPUs (only if larger context/models are needed).
5.  **Benchmarking Suite Development:** Begin developing the benchmarking suite (as per ADR-011 and ADR-017) to evaluate the performance of `Llama-3.1` against other candidate models on `molsys-ai` specific tasks.
6.  **Improve RAG Quality:**
    - **Prompt Engineering:** Refine the prompt sent to the LLM to make more effective use of the documentation context.
    - **Chunking:** Evaluate and improve the document splitting strategy (`chunking`) in `rag/build_index.py`.
7.  **Expand Agent Tools:**
    - Begin implementing new real tools from the `molsysmt` ecosystem in `agent/tools/`, following the `ROADMAP.md` and ADRs.
8.  **Prepare Fine-Tuning Dataset (Roadmap v0.5):**
    - Start collecting and structuring data (code examples, question-answer pairs) for the future fine-tuning of the model.
