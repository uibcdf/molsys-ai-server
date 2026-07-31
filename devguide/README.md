# MolSys-AI Server Development Guide

> **Design status**
>
> These documents combine confirmed repository decisions, remembered ideas pending verification and new architectural proposals.

## Mission

`molsys-ai-server` provides shared remote services for MolSys-AI:

- language-model serving,
- public documentation assistants,
- static knowledge services for MolSysSuite,
- reproducible corpus and index construction,
- grounded answers with citations and API guardrails.

The server is not the scientific execution runtime. It must not execute MolSysSuite workflows on behalf of users or keep live molecular systems in memory.

## Architecture and migration

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [TRANSFORMATION.md](TRANSFORMATION.md)
- [ROADMAP.md](ROADMAP.md)

## Service contracts

- [PUBLIC_API.md](PUBLIC_API.md): versioned inference, knowledge and assistant APIs.
- [KNOWLEDGE_SERVICE.md](KNOWLEDGE_SERVICE.md): corpus lifecycle, retrieval, releases and guardrails.
- [AUTHORIZATION.md](AUTHORIZATION.md): credentials, authorization, quotas and tenant isolation.

## Operations and quality

- [DEPLOYMENT.md](DEPLOYMENT.md): deployment topology, isolation and observability.
- [EVALUATION.md](EVALUATION.md): release gates for inference, knowledge, security and operations.

## Guiding principle

RAG remains an implementation technique inside the knowledge service, not the organizing principle of MolSys-AI. The server exposes reliable knowledge and inference capabilities for documentation chatbots and the local scientific copilot.
