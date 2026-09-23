# MolSys-AI Server

**MolSys-AI Server provides the remote inference and MolSysSuite software-knowledge services of MolSys-AI.**

Its responsibilities include:

- model serving;
- MolSysSuite documentation/software knowledge;
- RAG and structured retrieval;
- API surfaces, symbol cards, recipes, citations, and guardrails;
- documentation assistants;
- versioned remote service APIs;
- authentication, deployment, and observability.

It is **not** the MolSysSuite scientific execution runtime and it is **not** MOLI Agent.

## Project structure

MolSys-AI is organized across:

- [molsys-ai](https://github.com/uibcdf/molsys-ai) — umbrella/architecture;
- [molsys-ai-server](https://github.com/uibcdf/molsys-ai-server) — this repository;
- [molsys-ai-client](https://github.com/uibcdf/molsys-ai-client) — typed remote-service SDK;
- [molsys-ai-agent](https://github.com/uibcdf/molsys-ai-agent) — MolSysSuite specialist agent.

## Current operational assets

The current chat API, documentation widget, vLLM stack, corpus/index pipeline, hybrid retrieval, citations, symbol verification, benchmarks, and deployment infrastructure remain strategic server assets.

The documentation chatbot is a read-only product surface backed by these services.

## Legacy migration note

This repository predates the final split and still contains `client/agent` and `client/cli` prototypes. They are retained temporarily as migration sources.

- planning/execution/tool-inspection concepts belong in `molsys-ai-agent`;
- HTTP transport/configuration/auth/typed remote contracts belong in `molsys-ai-client`;
- inference, software knowledge, RAG, documentation assistants, and remote APIs remain here.

Do not remove legacy code until replacement paths and compatibility tests are functional. See `devguide/TRANSFORMATION.md`.

## Environment boundary

The server-side inference environment should remain isolated from MolSysSuite scientific toolchains. MolSysSuite execution belongs to the agent/scientific environment.

See `devguide/` and `dev/` for architecture, operational runbooks, deployment, benchmarking, and migration details.
