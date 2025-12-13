
# MolSys-AI Roadmap (MVP-Oriented)

This roadmap focuses on getting a **usable MVP** while keeping the architecture clean.

## v0.1 – Repository skeleton

- Create initial repository structure.
- Add core development documentation:
  - ARCHITECTURE
  - ROADMAP
  - CONSTRAINTS
  - ADRs for key decisions
- Provide a minimal `molsys-ai` CLI that starts and prints a friendly message.

## v0.2 – Minimal agent + model server stub

- Implement `agent/core.py` with a very simple agent loop:
  - accept user input
  - send it to a model endpoint (can be a stub or a tiny model)
  - return the answer
- Implement `model_server/server.py`:
  - FastAPI app with a `/v1/chat` endpoint.
  - For MVP, it can echo or use a very small local model.
- Wire the CLI to the agent and the agent to the model server.

## v0.3 – First real tools and basic RAG

- Implement `agent/tools/molsysmt_tools.py` with one or two real tools:
  - e.g. load a system and print basic info.
- Implement `rag/build_index.py` and `rag/retriever.py`:
  - index a subset of MolSysMT docs.
  - expose a retrieval function used by the agent.

## v0.4 – Documentation chatbot backend + widget

- Implement `docs_chat/backend.py`:
  - FastAPI endpoint for documentation Q&A.
  - Use RAG + model server to answer questions.
- Implement a minimal JS widget in `web_widget/molsys_ai_widget.js`:
  - embeds a chat panel into Sphinx-generated HTML.

## v0.5 – First LoRA specialization

- Prepare a small, curated dataset of:
  - API usage examples,
  - docstring-based Q&A,
  - simple workflows.
- Train a first QLoRA/LoRA on the selected base model for MolSys-AI (see ADR-017).
- Publish the resulting model under `uibcdf/` on Hugging Face Hub.
- Update the model server to use this specialized model (vLLM + AWQ baseline).

Further versions will refine the agent autonomy, add more tools and improve robustness.
