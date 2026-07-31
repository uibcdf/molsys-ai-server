# Server Roadmap

## Phase 0 — Baseline and archaeology

- Treat the existing documentation chatbot as a maintained product.
- Inventory endpoints, generated artifacts and deployment assumptions.
- Verify remembered profile, token, CLI and authentication ideas against repository history.
- Record accepted server boundaries in current ADRs or replacement design records.

## Phase 1 — Stabilize service contracts

- Separate inference contracts from knowledge-query contracts.
- Add explicit API versioning and capability reporting.
- Define typed request, response, citation and error models.
- Preserve compatibility with current chat consumers.

## Phase 2 — Formalize the knowledge service

- Expose project-scoped retrieval.
- Expose symbol cards and recipes as first-class results where useful.
- Report corpus versions, source repository commits and index metadata.
- Retain symbol verification and re-read guardrails.
- Add evaluation cases for MolSysViewer, TopoMT and PharmacophoreMT documentation.

## Phase 3 — Documentation assistants

- Define a reusable configuration for embedding assistants in each MolSysSuite documentation site.
- Support per-project prompts, filters and presentation rules without duplicating the backend.
- Add PharmacophoreMT when its documentation corpus becomes available.

## Phase 4 — Client extraction

- Move or rewrite generic transport in `molsys-ai-client`.
- Move interactive CLI and agent concepts to `molsys-ai`.
- Mark remaining server-side client/agent modules as legacy.
- Remove them only after compatibility tests pass.

## Phase 5 — Authentication and operations

- Define opaque-token authentication and authorization.
- Add quotas, redaction, audit logging and operational metrics.
- Keep client-side profile representation independent from server internals.

## Phase 6 — Scale only when required

- Upgrade storage or reranking only after benchmark and latency evidence.
- Preserve offline corpus building and deploy generated artifacts to inference hosts.
- Avoid coupling server stability to MolSysSuite runtime dependencies.
