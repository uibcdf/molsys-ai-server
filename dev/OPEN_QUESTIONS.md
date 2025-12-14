
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
- Whether to keep a separate preprocessing step in `server/rag/build_index.py`.

## 3. Embedding/index update strategy

Questions:
- How often will the index be rebuilt?
- Will we support incremental updates (e.g. only re-index changed files)?
- Where will the FAISS index live (e.g. `server/rag/index/`) and how will
  it be versioned?

## 4. Derived corpus generation (LLM-digested artifacts)

We currently index a **literal snapshot** of the documentation files. A planned
quality improvement is to generate an additional **derived corpus** layer
(summaries, FAQs, concept cards, API overviews) using an LLM and index it
alongside the raw docs.

Open questions:
- What exact artifact types provide the best retrieval lift for MolSysSuite docs?
- How do we enforce provenance and traceability (sources, commit hashes, prompt, model id)?
- How do we prevent hallucinations in derived artifacts (e.g. only summarize with citations)?
- How often do we regenerate derived artifacts (weekly with the snapshot, or on-demand)?
- Do we store derived artifacts in the same corpus directory under a dedicated subtree
  (e.g. `derived/`), or in a separate corpus root?

## 4. Backend model evolution

- Current baseline: vLLM (see ADR-015).
- Legacy: llama.cpp (see ADR-003/014).

Open point:
- Exact interface of the `ModelClient` abstraction that will allow swapping
  vLLM, alternative engines (SGLang/ExLlamaV2), or remote services without
  changing the agent core.

Related open point:
- Conversation context management: the vLLM backend now applies the model chat
  template (multi-turn), but a real chatbot still needs robust context
  management (history truncation strategy, optional summarization, and/or
  external memory).

## 5. Agent Planner Evolution Strategy

The design of the agent's planner, which was previously a high-level open question, has been decided with a two-phase strategy:

- **Phase 1 (MVP):** The project will proceed with the current `SimplePlanner`. This heuristic-based planner (using keywords and regular expressions) is fast, cost-effective, and reliable for well-defined tasks. Its main purpose is to enable the development and validation of the full end-to-end agent workflow (tool execution, RAG integration, response generation).

- **Phase 2 (Post-MVP):** Once the core agent mechanics are proven stable, the `SimplePlanner` will be evolved into an advanced, LLM-based planner. This upgrade is critical for enabling flexible, natural language-based tool selection and, most importantly, the multi-step reasoning required for true agent autonomy.

The specifics of the LLM-based planner (e.g., ReAct, CoT), cost/latency limits, and auditing will be defined as part of the Phase 2 implementation.

## 6. First concrete MolSysMT/TopoMT tools

- We have placeholders under `client/agent/tools/`.
- The first real tools to implement are not fully fixed, but likely:
  - load a molecular system (MolSysMT) and summarise it,
  - run a simple pocket/topography analysis (TopoMT),
  - display basic information or suggested next steps.

## 7. Evaluation dataset design

- ADR-011 describes the *strategy* for benchmarks.
- The concrete test cases and success metrics are still to be curated.

## 8. Web chatbot integration details

- The widget skeleton exists under `server/web_widget/`.
- Open points:
  - Exact JS API and CSS styling.
  - Authentication or rate-limiting (if needed).
  - How to wire it into the existing Sphinx-based documentation themes.
