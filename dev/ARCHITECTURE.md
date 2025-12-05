
# MolSys-AI Architecture (Initial Draft)

This document describes the **high-level architecture** of MolSys-AI.

The system is composed of several logical components:

- **agent/**: core agent loop, planning and tool execution.
- **rag/**: retrieval-augmented generation layer (indexing and retrieval over documentation and code).
- **model_server/**: abstraction and HTTP interface to the underlying language model.
- **cli/**: command line interface used by end users.
- **docs_chat/**: backend API to power an embedded chat widget in Sphinx/GitHub Pages.
- **web_widget/**: small JS widget to embed the documentation chatbot.
- **dev/**: internal documentation, ADRs and design notes.
- **tests/**: smoke tests and, later, unit/integration tests.

The design is intentionally modular so that:

- The agent does **not** depend on a specific model backend (llama.cpp, vLLM, TGI, etc.).
- The RAG layer can swap vector stores (FAISS now, others later).
- The same core can power both CLI and web documentation chat.
- Training (LoRA/QLoRA) can happen on a separate node; serving uses Hugging Face Hub as the model registry.

For more details, see the ADRs in `dev/decisions/`.
