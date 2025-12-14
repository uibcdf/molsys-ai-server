This note captures the minimal Conda+pip environment that has been verified
to work with vLLM on this machine (Python 3.12 + CUDA-enabled PyTorch wheels).

For the full, end-to-end runbook (CUDA Toolkit / `nvcc`, model download via
Hugging Face SSH + `git-lfs`, and `uvicorn` smoke tests), see:

- `dev/RUNBOOK_VLLM.md`

## Minimal environment

You can use either a minimal conda environment or a standard Python venv.

### Option A: conda (minimal)

conda create -n vllm python=3.12 pip -c conda-forge --strict-channel-priority
conda activate vllm


pip install -U pip
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129

### Option B: venv (minimal)

python -m venv .venv-vllm
source .venv-vllm/bin/activate

pip install -U pip
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129

## System requirement: CUDA Toolkit (nvcc)

Even when using CUDA-enabled PyTorch wheels, vLLM may need `nvcc` to JIT-compile
kernels (FlashInfer). On this machine we use a minimal CUDA Toolkit install:

- `sudo apt-get install -y cuda-nvcc-12-9 cuda-cudart-dev-12-9`

See `dev/RUNBOOK_VLLM.md`.
