# Transformation Plan

## Objective

Transform the current repository into a focused server for inference, knowledge and public documentation assistants without discarding mature work already present.

## Preserve

The following areas are considered strategic assets:

- model serving and deployment,
- FastAPI service boundaries,
- corpus synchronization and provenance,
- project-specific indices,
- API surfaces,
- symbol cards,
- notebook, test, docstring and Markdown recipes,
- BM25 and hybrid retrieval,
- source citations and anchors,
- symbol verification and re-reading,
- benchmark infrastructure,
- hardware and offline-build assumptions.

## Move or extract

Code under legacy client and agent directories should be reviewed and classified:

- generic HTTP transport may move to `molsys-ai-client`,
- CLI concepts may move to `molsys-ai`,
- planner, executor and tool prototypes may inform the new design but should not be copied unchanged,
- schemas shared across the boundary must receive explicit ownership and versioning.

## Deprecate

The server should gradually deprecate:

- user-facing CLI packaging,
- local MolSysSuite execution,
- agent state and orchestration,
- arbitrary shell tools,
- assumptions that one chat endpoint represents the complete MolSys-AI product.

## Compatibility strategy

1. Keep current documentation-chat behavior working.
2. Introduce stable inference and knowledge contracts alongside existing endpoints.
3. Move client functionality incrementally.
4. Mark legacy modules clearly before removal.
5. Add compatibility tests for `molsys-ai-client`.
6. Remove legacy client/agent code only after replacement paths are functional.

## Repository archaeology register

During migration, maintain a table of recovered ideas with evidence and status:

| Idea | Evidence | Status | Destination |
|---|---|---|---|
| Server/client split | ADR-018 | Confirmed | all repositories |
| Local scientific execution | constraints/architecture | Confirmed | `molsys-ai` |
| Code-aware knowledge corpus | ADR-019/021 | Confirmed | server |
| API-symbol verification | ADR-020 | Confirmed | server |
| Interactive CLI like Codex CLI | pending search | Remembered — verify | `molsys-ai` |
| Tokenized user profiles | pending search | Remembered — verify | client/server |
| Multiple environment profiles | pending search | Remembered — verify | client/core |

This register should be updated as repository history is inspected.
