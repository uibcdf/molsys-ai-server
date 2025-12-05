
# MolSys-AI – Open Questions

This file lists topics that are intentionally left open or partially specified,
to be refined as the project evolves.

## 1. Embedding model for RAG

- We decided to use FAISS as the vector store (see ADR-004),
  but the exact embedding model is not yet fixed.
- Candidates (to be evaluated):
  - Small, CPU-friendly models for local use.
  - GPU-backed sentence-transformers or similar.
- Requirements:
  - Open license.
  - Good performance on technical/scientific text.
  - Easy integration with Python.

## 2. Exact directory structure for RAG inputs

Current draft idea:
- Use a configurable input directory, e.g. `data/docs/`,
  containing Sphinx-generated text/HTML or preprocessed markdown.

Open points:
- Final format of the ingested docs (HTML vs. markdown vs. plain text).
- Whether to keep a separate preprocessing step in `rag/build_index.py`.

## 3. Embedding/index update strategy

Questions:
- How often will the index be rebuilt?
- Will we support incremental updates (e.g. only re-index changed files)?
- Where will the FAISS index live (e.g. `rag/index/`) and how will
  it be versioned?

## 4. Backend model evolution

- MVP: llama.cpp (see ADR-003).
- Future: vLLM when hardware is upgraded.

Open point:
- Exact interface of the `ModelClient` abstraction that will allow us
  to swap llama.cpp, vLLM or other backends without changing the agent.

## 5. Level of autonomy of the agent

- Planner/executor design is still high-level.
- We need to refine:
  - how many reasoning/tool-calling iterations per request,
  - how to limit cost/latency,
  - how to log and audit tool usage.

## 6. First concrete MolSysMT/TopoMT tools

- We have placeholders under `agent/tools/`.
- The first real tools to implement are not fully fixed, but likely:
  - load a molecular system (MolSysMT) and summarise it,
  - run a simple pocket/topography analysis (TopoMT),
  - display basic information or suggested next steps.

## 7. Evaluation dataset design

- ADR-011 describes the *strategy* for benchmarks.
- The concrete test cases and success metrics are still to be curated.

## 8. Web chatbot integration details

- The widget skeleton exists under `web_widget/`.
- Open points:
  - Exact JS API and CSS styling.
  - Authentication or rate-limiting (if needed).
  - How to wire it into the existing Sphinx-based documentation themes.
