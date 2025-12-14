# Checkpoint: MolSys-AI Server Project Status

This document summarizes the current status of the project, its objectives, the progress made, and the next steps.

Repository naming note:

- Server repo: `molsys-ai-server` (this repo).
- Client repo (planned): `molsys-ai-client`.
- User-facing distribution + command: `molsys-ai` / `molsys-ai`.
- Server distribution (if published): `molsys-ai-server`.

See `dev/decisions/ADR-018.md`.

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
- **Single-GPU inference:** `model_server` + vLLM successfully loads the model and answers `POST /v1/engine/chat`.
- **Multi-turn chat support:** the vLLM backend now applies the model chat template (via `transformers`), so `POST /v1/engine/chat`
  accepts full `messages` history (not just the last user prompt).
- **Stable baseline config (11 GB GPU):**
  - `max_model_len=8192`
  - `gpu_memory_utilization=0.80`
  - `enforce_eager=true` (stability-first; avoids OOM during compilation/warmup)
  - Observed VRAM usage during long-prompt tests: ~9.4–9.8 GiB on GPU0.
- **Reproducible smoke runner:** `dev/smoke_vllm.sh` starts the server, performs minimal + multi-turn requests against `POST /v1/engine/chat`, and
  cleans up on exit (including best-effort cleanup of any stray `VLLM::EngineCore` processes).
- **Chat API end-to-end (backend):** validated `chat_api` → `model_server`:
  - `POST /v1/chat` with legacy `query` returns `OK`,
  - `POST /v1/chat` with `messages` history preserves context (multi-turn) and returns `Diego` in the test.
- **API naming cleanup:** public endpoint is `POST /v1/chat`; internal engine endpoint is `POST /v1/engine/chat` (no legacy `/v1/docs-chat` naming).
- **RAG embeddings validation:** `chat_api` was also validated with `sentence-transformers` installed (v5.2.0), building a
  small Markdown index and answering a retrieval-backed question correctly (example: returns `1VII` for a synthetic doc).
- **API authentication policy (production-ready):**
  - `POST /v1/engine/chat` (internal model backend) can be protected with `MOLSYS_AI_ENGINE_API_KEYS` (API key allowlist).
  - `chat_api` can authenticate to the model server via `MOLSYS_AI_ENGINE_API_KEY`.
  - `POST /v1/chat` is public by default, but can be protected with `MOLSYS_AI_CHAT_API_KEYS` if needed.
  - Smoke scripts support optional API keys via `MOLSYS_AI_ENGINE_API_KEYS` / `MOLSYS_AI_ENGINE_API_KEY` (engine) and `MOLSYS_AI_CHAT_API_KEYS` / `MOLSYS_AI_CHAT_API_KEY` (chat API).
- **CLI direction (public + lab users):**
  - The CLI is now an HTTP API client with subcommands: `login`, `chat`, `docs`.
  - `molsys-ai chat` calls `POST /v1/chat` with `client="cli"` so it can use RAG (specialist answers) and decide when
    to show sources via an LLM router call.
  - It stores a user API key locally under `~/.config/molsys-ai/config.json` (mode `0600` when possible).
  - For keys prefixed with `u1_` (lab/investigator), the CLI attempts a LAN endpoint first and falls back to the public API.
  - For keys prefixed with `u2_` (external), the CLI uses the public API only.

## 3. Current Status

The foundational model (`Llama-3.1-8B-Instruct-AWQ-INT4`) is deployed on Hugging Face (`uibcdf` organization). The vLLM
backend for `model_server` is working on RTX 2080 Ti GPUs (CUDA 12.9 `nvcc` + vLLM `cu129` wheels) and supports multi-turn
chat formatting.

Repository documentation has been consolidated around:

- `AGENTS.md` as the single “how to work here” entry point,
- `dev/RUNBOOK_VLLM.md` as the inference runbook,
- a minimal `environment.yml` for general development (install Python deps via `pip install -e ".[dev]"`).
- the stable API contract for external clients: `dev/API_CONTRACT_V1.md`.

The chat API stack is now validated end-to-end locally, including the browser/widget path:

- `server/chat_api/backend.py` now accepts `messages` (recommended) in addition to legacy `query`,
- `POST /v1/chat` now returns `sources` aligned with citations `[1]`, `[2]`, ... and (when available) deep links to
  published docs pages under `https://www.uibcdf.org/<tool>/...#Label`,
- `server/web_widget/molsys_ai_widget.js` now keeps conversation history and sends it as `messages` when `mode: "backend"`.
  This has been validated end-to-end with a local Sphinx build + CORS (see `docs/index.md` and `dev/smoke_widget.sh`).
- the widget renders a compact “Sources” dropdown for each assistant reply (using the `sources` field),
- `chat_api` supports optional CORS via `MOLSYS_AI_CORS_ORIGINS` (comma-separated), to allow serving docs and backend on
  different ports during local/widget smoke tests.
- Sphinx docs build now prefers `myst-nb` (with a fallback to `myst-parser`) and the widget can be toggled to backend mode
  via query parameters (`molsys_ai_mode` / `molsys_ai_backend_url`) in `docs/_static/molsys_ai_config.js`.
- Widget end-to-end smoke (Sphinx page → widget → chat_api → model_server) was validated via CORS preflight + cross-origin
  POST, retrieving the expected answer (`1VII`). A reproducible runner exists: `dev/smoke_widget.sh`.
- Public deployment direction: serve docs via GitHub Pages under `https://uibcdf.org/...` and run the API separately under
  `https://api.uibcdf.org` (DNS `A` record in place). A deployment guide is captured in `dev/DEPLOY_API.md`.
- If the data-center firewall keeps `80/443` blocked to this GPU host, the fallback path for an external demo is:
  external VPS + SSH reverse tunnel, documented in `dev/DEPLOY_API.md`.
- Firewall reality: `80/443` are upstream-filtered by the data-center firewall. A scan shows `8080/tcp` reachable, but a
  remote request to `http://187.141.21.243:8080/` returns a Tomcat welcome page, so it is not currently usable for the API.
  For a public demo before `443` is opened, use an external VPS + SSH reverse tunnel (see `dev/DEPLOY_API.md`).
- Reverse proxy scaffolding is now available in-repo:
  - Caddy: `dev/Caddyfile.example`
  - systemd units + env file: `dev/systemd/`, `dev/molsys-ai.env.example`

Tool execution policy and environment separation:

- The **documentation chatbot** is read-only: it answers questions and can generate scripts/snippets, but it does not
  execute MolSysSuite tools.
- Tool execution belongs to the **local agent** (`molsys-ai agent`) running on the user's machine.
- To avoid CUDA dependency conflicts, MolSysSuite toolchains should be installed in a dedicated agent environment
  (for example a conda env `molsys-agent`), separate from the server-side vLLM inference environment.

RAG corpus automation (live docs repos):

- A reproducible corpus snapshot + index build is available via `dev/sync_rag_corpus.py` (targets sibling repos:
  `../molsysmt`, `../molsysviewer`, `../pyunitwizard`, `../topomt`).
- The same script can generate an anchors map (`server/chat_api/data/anchors.json`) by extracting explicit MyST labels
  `(Label)=` from `docs/` sources, without running Sphinx or importing upstream packages.
- The generated corpus and index live under `server/chat_api/data/` by default and are intentionally ignored by git.
- An in-process end-to-end smoke (retrieve → prompt → vLLM backend) is available via `dev/smoke_chat_inprocess.py`.

## 4. Next Steps

The immediate next steps focus on validating the docs chatbot and then iterating on quality:

0.  **Deployment unblock (public demo):**
    - Send the port-opening request for `443/tcp` (and optionally `80/tcp`) using `dev/FIREWALL_PORT_REQUEST_TEMPLATE.md`.
    - Until `443` is opened, use the external VPS + SSH reverse tunnel path in `dev/DEPLOY_API.md`.
    - Once `443` is opened, deploy with Caddy + systemd using `dev/RUNBOOK_DEPLOY_443.md`.
1.  **Widget integration smoke:** Run `model_server` (vLLM) + `chat_api` together and verify that the web widget (Sphinx)
    can talk to `POST /v1/chat` in multi-turn mode (`messages` history), returning answers that cite `[1]`, `[2]` and
    sources that deep-link into `https://www.uibcdf.org/<tool>/...#Label`.
2.  **Docs deployment integration:** Decide how the widget and backend will be deployed for real docs sites (same-origin vs
    separate service; CORS; rate-limiting if needed).
    - Keep `model_server` (`/v1/engine/chat`) bound to localhost; expose `chat_api` (`/v1/chat`) for CLI + widget, with API keys
      and rate limiting as needed.
3.  **Improve RAG quality:**
    - **Prompt Engineering:** Refine the prompt sent to the LLM to make more effective use of the documentation context.
    - **Chunking:** Evaluate and improve the document splitting strategy (`chunking`) in `server/rag/build_index.py`.
4.  **Benchmarking suite:** Implement benchmark tasks (ADR-011/017) to evaluate changes in retrieval/prompting/models.
5.  **Expand local agent tools (client-side):**
    - Keep tool execution local-only (`molsys-ai agent`), and avoid server environments importing MolSysSuite toolchains.
    - Begin implementing new tools from the `molsysmt` ecosystem in `client/agent/tools/`, following the `ROADMAP.md` and ADRs.
6.  **Multi-GPU note (optional):** Only if we need more context or larger models, test `tensor_parallel_size=3` across the
    3× RTX 2080 Ti GPUs.
7.  **Prepare fine-tuning dataset (Roadmap v0.5):**
    - Start collecting and structuring data (code examples, question-answer pairs) for the future fine-tuning of the model.
