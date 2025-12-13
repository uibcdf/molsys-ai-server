This note captures the minimal Conda+pip environment that has been verified
to work with vLLM on this machine (Python 3.12 + CUDA-enabled PyTorch wheels).

For the full, end-to-end runbook (CUDA Toolkit / `nvcc`, model download via
Hugging Face SSH + `git-lfs`, and `uvicorn` smoke tests), see:

- `dev/RUNBOOK_VLLM.md`

## Minimal environment

conda create -n vllm python=3.12 pip -c conda-forge --strict-channel-priority
conda activate vllm


pip install -U pip
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
