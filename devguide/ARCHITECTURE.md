# Target Server Architecture

## Mission

`molsys-ai-server` provides shared remote inference and **MolSysSuite Software Knowledge** capabilities for documentation assistants, MolSys-AI Agent, and other authorized clients.

## Main services

```text
molsys-ai-server
├── inference service
│   ├── model backend
│   ├── generation API
│   └── streaming
├── MolSysSuite Software Knowledge Service
│   ├── documentation corpus
│   ├── API surfaces and symbol cards
│   ├── recipes and tutorials
│   ├── retrieval / RAG
│   ├── citations and provenance
│   └── API-symbol guardrails
├── documentation assistants
├── authentication and quotas
└── deployment and observability
```

RAG is an implementation technique inside Software Knowledge, not the identity of the service or product.

## Consumers

Software Knowledge may serve:

- the documentation chatbot;
- MolSys-AI Agent;
- MolSys-AI Client consumers;
- future MolSysSuite-facing applications.

The documentation chatbot is the first operational consumer and its capability must remain available during restructuring, but its current endpoint/module implementation is not architecturally frozen.

## Server boundary

The server must not own active molecular systems, a user's live MolSysViewer canvas, local file access, MolSysSuite tool execution, the specialist-agent loop, or authoritative scientific project history.

Those responsibilities belong to the scientific/agent environment, primarily `molsys-ai-agent` where agent behavior is concerned.

## Confirmed principles

- Inference remains separable from MolSysSuite toolchains.
- Corpus construction is reproducible and can run offline.
- Retrieval is project-aware and code-aware.
- Symbol verification/re-reading reduces invented or misused APIs.
- Grounded answers provide sources and stable citations.
- Server contracts should be versioned and capability-discoverable.

## Documentation assistants

Documentation assistants are narrow consumers of inference + Software Knowledge. They execute no local scientific tools and require no private molecular session.

## Evolution

Current `POST /v1/chat` may remain as a compatibility facade while more explicit inference, knowledge, and documentation-assistant contracts evolve.
