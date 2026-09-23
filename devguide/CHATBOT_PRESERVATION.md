# Documentation Chatbot Capability Gate

The documentation chatbot is the first operational MolSys-AI capability and must remain usable throughout repository restructuring.

This gate protects the **user-visible capability**, not the current internal implementation.

> **Preserve the chatbot capability while allowing its implementation to evolve toward the target architecture.**

## Capability invariants

During migration, users must retain:

- grounded answers about MolSysSuite documentation/software usage;
- useful source citations and source metadata;
- documentation integration;
- conversational use, including supported multi-turn behavior;
- reliable handling of documented APIs and symbols;
- an operational end-to-end path from documentation UI to an answer.

Equivalent or better replacement behavior satisfies the gate.

## Compatibility surface during migration

The current widget and `POST /v1/chat` contract should remain compatible until replacement consumers/contracts are operational and migration is deliberate.

This does **not** freeze `/v1/chat` as a permanent architectural contract.

Internals that may evolve include:

- endpoint decomposition;
- request/response schemas under new API versions;
- retrieval/RAG implementation;
- index technology;
- model backend;
- corpus representation;
- widget implementation;
- internal module/repository layout.

Compatibility facades may preserve old consumers while new knowledge/inference/assistant contracts are introduced.

## Software Knowledge Service

The RAG stack should be understood as part of the **MolSysSuite Software Knowledge Service**, not as infrastructure owned only by the chatbot.

Potential consumers include:

```text
             Software Knowledge Service
                 /        |        \
                ▼         ▼         ▼
       Documentation   MolSys-AI   future
          Chatbot        Agent     clients
```

RAG, BM25, embeddings, symbol cards, API surfaces, recipes, citations, verification, and provenance are implementation/assets of this reusable knowledge capability.

## Migration rule

Agent and client extraction should be additive first. Legacy modules remain until equivalent destination functionality and compatibility tests exist.

## Test ownership target

```text
molsys-ai-server
    software-knowledge / docs-assistant / API / model-service tests

molsys-ai-client
    transport / config / auth / schema compatibility tests

molsys-ai-agent
    planner / executor / tool / inspection / local-execution tests
```

The current mixed smoke tests should be decomposed before legacy packages are removed.

## Release principle

> **Restructuring the repositories must not turn a working chatbot into a future promise.**

This principle protects functionality while explicitly permitting architectural improvement.
