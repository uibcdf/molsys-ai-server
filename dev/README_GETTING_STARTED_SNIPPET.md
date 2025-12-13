
# Getting started (development)

1. **Create and activate a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate  # Windows
   ```

2. **Install development dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

3. **Run the smoke tests**:

   ```bash
   pytest
   ```

4. **Start the model server (MVP stub)**:

   ```bash
   uvicorn model_server.server:app --reload
   ```

   Then open http://127.0.0.1:8000/docs to inspect the API.

5. **Try the CLI**:

   ```bash
   python -m cli.main
   ```

You can find more detailed information in:

- `dev/DEV_GUIDE.md`
- `dev/RUNBOOK_VLLM.md` (vLLM + CUDA 12.9 runbook)
- `dev/ARCHITECTURE.md`
- `dev/ROADMAP.md`
- `dev/CONSTRAINTS.md`
- `dev/decisions/` (ADRs)

All development documentation and code in this repository must be written in **English** (see `dev/DEV_GUIDE.md`, language conventions).
