# Transformation Plan

## Objective

Transform the repository into a focused server for inference, MolSysSuite Software Knowledge, and documentation assistants without discarding mature operational work.

## Preserve as capabilities/assets

- working documentation-chatbot capability;
- model serving and deployment;
- current ingress/deployment infrastructure while useful;
- corpus synchronization and provenance;
- project-specific indices;
- API surfaces and symbol cards;
- recipes/tutorial-derived knowledge;
- BM25, dense and hybrid retrieval;
- citations and anchors;
- symbol verification and re-reading;
- benchmark infrastructure.

Preservation of the chatbot means preservation of equivalent user-visible capability, **not permanent freezing of its current endpoint, RAG implementation, schemas, or widget internals**.

## Target interpretation

The current RAG stack becomes part of the reusable **MolSysSuite Software Knowledge Service**. The documentation chatbot is its first operational consumer, not its owner.

## Move or extract

Legacy code is classified by responsibility:

- HTTP transport/config/auth → `molsys-ai-client`;
- planner/executor/tool/API-inspection/local execution → `molsys-ai-agent`;
- inference/Software Knowledge/docs assistants → remain in server;
- mixed CLI code → split by responsibility.

## Deprecate gradually

- server-owned local MolSysSuite execution;
- agent state/orchestration inside server;
- arbitrary local shell execution in server;
- user-facing agent packaging from server;
- assumption that one chat endpoint represents the complete MolSys-AI product.

## Compatibility strategy

1. Establish chatbot capability baseline.
2. Introduce stable inference/knowledge contracts alongside current interfaces.
3. Keep `/v1/chat` compatible while existing consumers need it.
4. Implement typed client functionality incrementally.
5. Establish MolSys-AI Agent independently.
6. Migrate mixed legacy code by responsibility.
7. Decompose mixed tests by repository ownership.
8. Remove legacy server modules only after replacement paths and compatibility gates pass.

A compatibility facade may outlive the internal implementation it originally exposed.
