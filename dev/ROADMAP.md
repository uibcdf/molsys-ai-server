
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

- Implement `client/agent/core.py` with a very simple agent loop:
  - accept user input
  - send it to a model endpoint (can be a stub or a tiny model)
  - return the answer
- Implement `server/model_server/server.py`:
  - FastAPI app with a `/v1/chat` endpoint.
  - For MVP, it can echo or use a very small local model.
- Wire the CLI to the agent and the agent to the model server.

## v0.3 – First real tools and basic RAG

- Implement `client/agent/tools/molsysmt_tools.py` with one or two real tools:
  - e.g. load a system and print basic info.
- Implement `server/rag/build_index.py` and `server/rag/retriever.py`:
  - index a subset of MolSysMT docs.
  - expose a retrieval function used by the agent.
- Establish the RAG corpus update workflow:
  - sync a **literal snapshot** of documentation files from the live repos,
  - rebuild the embedding index regularly (e.g. weekly).
  - See `dev/sync_rag_corpus.py`.

## v0.4 – Documentation chatbot backend + widget

- Implement `server/docs_chat/backend.py`:
  - FastAPI endpoint for documentation Q&A.
  - Use RAG + model server to answer questions.
  - Add an LLM router path so the CLI can use the same endpoint with:
    - RAG always available (specialist answers),
    - sources/citations shown only when requested or inferred.
- Add deep-linkable sources:
  - extract explicit MyST labels `(Label)=` from upstream docs snapshots,
  - return a `sources` list aligned with citations `[1]`, `[2]`, ...,
  - link to published docs pages under `https://www.uibcdf.org/<tool>/...#Label`.
- Implement a minimal JS widget in `server/web_widget/molsys_ai_widget.js`:
  - embeds a chat panel into Sphinx-generated HTML.
  - always renders a compact “Sources” dropdown for each assistant reply.
- Add an optional **derived corpus** layer for quality:
  - generate “digested” docs artifacts (summaries, FAQs, concept cards, API overviews)
    using an LLM,
  - store them as separate text sources alongside the literal snapshot (with provenance),
  - index them together with the raw docs, but keep them clearly separated so they can
    be rebuilt safely when upstream docs change.

## v0.5 – First LoRA specialization

- Prepare a small, curated dataset of:
  - API usage examples,
  - docstring-based Q&A,
  - simple workflows.
- Train a first QLoRA/LoRA on the selected base model for MolSys-AI (see ADR-017).
- Publish the resulting model under `uibcdf/` on Hugging Face Hub.
- Update the model server to use this specialized model (vLLM + AWQ baseline).

Further versions will refine the agent autonomy, add more tools and improve robustness.
