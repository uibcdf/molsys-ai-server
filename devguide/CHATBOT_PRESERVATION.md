# Chatbot Preservation Gate

The documentation chatbot is the first operational MolSys-AI capability and is protected during repository restructuring.

## Required invariants

A migration change must not be accepted if it breaks:

- the public `POST /v1/chat` behavior currently used by the documentation widget;
- grounded MolSysSuite documentation retrieval;
- source citations and source metadata;
- documentation anchors/deep links;
- multi-turn messages;
- project-aware/code-aware retrieval;
- symbol verification/re-reading guardrails;
- widget → chat API → model server end-to-end operation;
- supported authentication/CORS behavior;
- current deployment path.

## Migration rule

Agent and client extraction must be additive first. Legacy server modules remain until equivalent functionality exists in the destination repositories and compatibility tests pass.

## Test ownership target

```text
molsys-ai-server
    chatbot/API/RAG/model-service tests

molsys-ai-client
    transport/config/auth/schema compatibility tests

molsys-ai-agent
    planner/executor/tool/inspection/local-execution tests
```

The current mixed smoke tests should be decomposed before legacy packages are removed.

## Release principle

> **Restructuring the repositories must not turn a working chatbot into a future promise.**
