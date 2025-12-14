# MolSys-AI vLLM Runbook (CUDA 12.9 + AWQ)

This document consolidates the current, working procedure to run the MolSys-AI
`model_server` with **vLLM** on **3× RTX 2080 Ti (11 GB)** hardware.

The goal is a stable baseline for:

- local, self-hosted inference,
- RAG experiments (long prompts),
- chatbot work (multi-turn) using the model chat template.

This runbook is intentionally practical (HPC-friendly) and uses:

- a minimal **Conda** env (`python=3.12`) + **pip** for vLLM,
- **system-level** CUDA Toolkit (for `nvcc`),
- model weights downloaded via **Hugging Face SSH + git-lfs**.

## 0) Assumptions

- OS: Ubuntu 24.04.
- NVIDIA driver is installed and working (`nvidia-smi` shows the GPUs).
- You have `git` and `git-lfs` installed and configured.
- You can access Hugging Face via SSH (see section 3).

## 1) Create a clean vLLM environment (Python 3.12)

This is the minimal environment that has been validated:

```bash
conda create -n vllm python=3.12 pip -c conda-forge --strict-channel-priority
conda activate vllm

pip install -U pip
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
```

Install the MolSys-AI server dependencies (from this repo):

```bash
pip install -e ".[server]"
```

If you want to run the chat API (`server/chat_api/`) with retrieval,
also install the RAG dependencies:

```bash
pip install -e ".[rag]"
```

### Note (some HPC environments): `conda` plugins may fail

On some systems you may see a `PermissionError` from `multiprocessing.SemLock`
while running `conda` commands (often triggered by CUDA virtual package
plugins). In that case run conda with plugins disabled:

```bash
export CONDA_NO_PLUGINS=true
```

or:

```bash
conda --no-plugins <command> ...
```

Sanity check:

```bash
python - <<'PY'
import torch, vllm
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY
```

## 2) Install CUDA Toolkit (nvcc) at system level (CUDA 12.9)

vLLM will often use FlashInfer and JIT-compile kernels. This requires `nvcc`.
The minimal fix is to install **only** `nvcc` + runtime headers, not a full
desktop CUDA toolkit bundle.

### 2.1 Add NVIDIA CUDA APT repo (Ubuntu 24.04)

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
```

### 2.2 Install a minimal CUDA 12.9 toolchain

```bash
sudo apt-get install -y cuda-nvcc-12-9 cuda-cudart-dev-12-9
```

### 2.3 Ensure `nvcc` is visible

The packages install under `/usr/local/cuda-12.9` and register alternatives.
Make sure your shell can find `nvcc`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

nvcc --version
```

For a persistent setup, consider adding the exports to your shell rc or an
HPC module.

## 3) Download the model via Hugging Face SSH + git-lfs

This repo uses the model:

- `uibcdf/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`

If you authenticate to Hugging Face with SSH, clone with LFS:

```bash
mkdir -p models
cd models

GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:uibcdf/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
cd Meta-Llama-3.1-8B-Instruct-AWQ-INT4
git lfs pull
```

This produces `*.safetensors` shards in the cloned directory.

## 4) Configure the MolSys-AI model server (single GPU baseline)

Create a YAML config file (example path: `/tmp/molsys_ai_vllm.yaml`):

```yaml
model:
  backend: "vllm"
  local_path: "/ABS/PATH/TO/molsys-ai-server/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
  quantization: "awq"
  tensor_parallel_size: 1

  # Context/window limit (tokens). 8192 is a good initial baseline.
  max_model_len: 8192

  # vLLM memory control. 0.80 is conservative on 11 GB GPUs.
  gpu_memory_utilization: 0.80

  # Stability-first baseline: disable torch.compile/cudagraph.
  enforce_eager: true
```

## 5) Run the server (uvicorn)

From the repository root, with the `vllm` env active:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

CUDA_VISIBLE_DEVICES=0 \
MOLSYS_AI_MODEL_CONFIG=/tmp/molsys_ai_vllm.yaml \
uvicorn model_server.server:app --host 127.0.0.1 --port 8001
```

OpenAPI:

- http://127.0.0.1:8001/docs

## 6) Smoke tests

Recommended: use the bundled script (it starts the server, waits for readiness,
hits the API, and cleans up on exit):

```bash
./dev/smoke_vllm.sh
```

This script includes a small multi-turn request to validate that the chat
template is applied.

### 6.0 Optional API key protection (engine endpoint)

For deployments where the engine endpoint is protected, set:

- `MOLSYS_AI_ENGINE_API_KEYS` on the engine server (comma-separated allowlist),
- `MOLSYS_AI_ENGINE_API_KEY` on the client side to send `Authorization: Bearer ...`.

### 6.1 Minimal generation

```bash
curl -sS -X POST http://127.0.0.1:8001/v1/engine/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say only: smoke test OK"}]}'
```

### 6.2 RAG-style long prompt

For RAG experiments, include excerpts and the question in the same `content`
(single-message RAG). Multi-turn chat is supported as well.

### 6.3 Chat API (end-to-end, with RAG)

If you want to validate the full path (`chat_api` → `model_server`):

1. Keep `model_server` running on `http://127.0.0.1:8001`.
2. Start the chat API on `8000`:

```bash
mkdir -p /tmp/molsys_ai_docs_smoke
cat >/tmp/molsys_ai_docs_smoke/example.md <<'MD'
# Example

The example PDB id is **1VII**.
MD

MOLSYS_AI_ENGINE_URL=http://127.0.0.1:8001 \
MOLSYS_AI_DOCS_DIR=/tmp/molsys_ai_docs_smoke \
MOLSYS_AI_DOCS_INDEX=/tmp/molsys_ai_docs_smoke.pkl \
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
uvicorn chat_api.backend:app --host 127.0.0.1 --port 8000
```

3. Call it:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the example PDB id? Reply with just the id.","k":3}'
```

## 7) Multi-GPU note (optional)

If/when you need more context or larger models, you can try tensor parallelism:

```yaml
tensor_parallel_size: 3
```

and run with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 ...
```

This is not required for the current baseline.
