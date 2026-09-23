# MolSys-AI Server Development Guide

## Mission

`molsys-ai-server` provides shared remote services for MolSys-AI:

- language-model serving;
- MolSysSuite software-knowledge services;
- public documentation assistants;
- reproducible corpus/index construction;
- grounded answers with citations and API guardrails.

The server is not the scientific execution runtime. MolSysSuite-specialist planning/execution belongs to [molsys-ai-agent](https://github.com/uibcdf/molsys-ai-agent).

## Architecture and migration

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [TRANSFORMATION.md](TRANSFORMATION.md)
- [ROADMAP.md](ROADMAP.md)

Legacy `client/agent` and `client/cli` modules remain temporarily as migration sources. Move responsibilities incrementally according to `TRANSFORMATION.md`; preserve working server behavior.

## Service contracts

- [PUBLIC_API.md](PUBLIC_API.md): versioned inference, knowledge and assistant APIs.
- [KNOWLEDGE_SERVICE.md](KNOWLEDGE_SERVICE.md): corpus lifecycle, retrieval, releases and guardrails.
- [AUTHORIZATION.md](AUTHORIZATION.md): credentials, authorization, quotas and tenant isolation.

## Operations and quality

- [DEPLOYMENT.md](DEPLOYMENT.md)
- [EVALUATION.md](EVALUATION.md)

## Guiding principle

RAG is an implementation technique inside the software-knowledge service, not the identity of MolSys-AI.

> **MolSys-AI Server knows about MolSysSuite software; MolSys-AI Agent operates MolSysSuite.**
