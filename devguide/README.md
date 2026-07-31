# MolSys-AI Server Development Guide

This directory defines the implementation direction of `molsys-ai-server`.

## Mission

`molsys-ai-server` provides shared remote services for MolSys-AI:

- language-model serving,
- public documentation assistants,
- static knowledge services for MolSysSuite,
- reproducible corpus and index construction,
- grounded answers with citations and API guardrails.

The server is not the scientific execution runtime. It must not execute MolSysSuite workflows on behalf of users or keep live molecular systems in memory.

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md): target architecture and boundaries.
- [TRANSFORMATION.md](TRANSFORMATION.md): migration from the current repository layout.
- [ROADMAP.md](ROADMAP.md): implementation sequence and milestones.

## Guiding principle

RAG remains an implementation technique inside the knowledge service, not the organizing principle of MolSys-AI. The server should expose reliable knowledge and inference capabilities that can be used by documentation chatbots and by the local scientific copilot.
