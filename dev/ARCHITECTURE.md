
# MolSys-AI Architecture (Initial Draft)

This document describes the **high-level architecture** of MolSys-AI.

The system is composed of several logical components:

- **client/agent/**: local agent loop, planning and tool execution (planned to split into its own repo).
- **client/cli/**: command line interface used by end users (planned to split into its own repo).
- **server/model_server/**: HTTP interface to the underlying language model (`/v1/chat`).
- **server/docs_chat/**: RAG-enabled chat API used by both the embedded widget and the CLI (`/v1/docs-chat`).
- **server/rag/**: retrieval-augmented generation layer used by server components.
- **server/web_widget/**: small JS widget to embed the documentation chatbot.
- **train/**: training and fine-tuning assets (configs, scripts, job launchers, notebooks).
- **dev/**: internal documentation, ADRs and design notes.
- **tests/**: smoke tests and, later, unit/integration tests.

The design is intentionally modular so that:

- The agent does **not** depend on a specific model backend (llama.cpp, vLLM, TGI, etc.).
- The RAG layer can swap storage backends (pickled MVP now; FAISS later).
- The same core can power both CLI and web documentation chat.
- Training (LoRA/QLoRA) can happen on a separate node; serving uses Hugging Face Hub as the model registry.

For more details, see the ADRs in `dev/decisions/`.

Repo split and naming is tracked in `dev/decisions/ADR-018.md`.

## Execution policy (important)

- The **documentation chatbot** (`server/docs_chat/`) must be read-only: it answers questions and generates code
  snippets, but it does not execute tools.
- Tool execution belongs to the **local agent** (`molsys-ai agent`) running on the user's machine.
  Any MolSysSuite toolchain dependencies are therefore local concerns and should not contaminate the server
  inference environment.

## RAG corpus strategy (important)

RAG quality depends heavily on the document corpus. The long-term plan is to maintain:

- a **literal snapshot** corpus (copied verbatim from the live documentation repos), and
- an optional **derived corpus** (LLM-generated summaries/FAQs/concept cards) stored as separate text sources
  with explicit provenance.

The derived corpus is additive and must never replace the literal snapshot.

For the docs chatbot UX, the snapshot pipeline also extracts explicit MyST labels `(Label)=` from upstream docs and
builds an `anchors.json` map. This allows `POST /v1/docs-chat` to return deep-linkable sources that point to published
documentation pages under `https://www.uibcdf.org/<tool>/...#Label` without compiling Sphinx HTML or executing upstream
`conf.py`.
