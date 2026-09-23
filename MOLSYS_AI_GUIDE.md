<!--
CANONICAL MOLSYS-AI GUIDE — child-repository copies should remain synchronized.
Canonical source: https://github.com/uibcdf/molsys-ai/blob/main/MOLSYS_AI_GUIDE.md
-->

# MolSys-AI subsystem guide

MolSys-AI is the AI subsystem specialized in understanding and operating MolSysSuite.

## Governance chain

```text
MOLI
  ↓ shared platform / engineering governance
MolSysSuite
  ↓ modeling-domain governance
MolSys-AI
  ↓ subsystem contracts and coordination
├── molsys-ai-server
├── molsys-ai-client
└── molsys-ai-agent
```

Child repositories do not need separate copies of `MOLI_GUIDE.md` or `MOLSYSSUITE_GUIDE.md`. This guide is their single governance ambassador and must preserve the inherited rules.

## Repository ownership

- `molsys-ai` — subsystem architecture, cross-repository contracts, governance, migration coordination, and shared subsystem decisions.
- `molsys-ai-server` — remote inference, MolSysSuite Software Knowledge, documentation assistants, corpus/index construction, and server deployment.
- `molsys-ai-client` — lightweight typed transport SDK for remote MolSys-AI services.
- `molsys-ai-agent` — local-first MolSysSuite specialist agent, tool planning/execution, state inspection, recovery, and agent-owned integrations.

The working documentation chatbot is a regression anchor. Preserve its capability while implementation ownership is migrated.

## Reporting rule

> A concern is governed at the lowest level that owns the shared contract it affects.

Examples:

- server-only RAG defect → `molsys-ai-server`;
- Client transport bug → `molsys-ai-client`;
- Agent tool-planning bug → `molsys-ai-agent`;
- Server ↔ Client contract → `molsys-ai`;
- Client ↔ Agent subsystem contract → `molsys-ai`;
- MolSys-AI ↔ MolSysSuite contract → `molsyssuite`;
- MolSys-AI ↔ MOLI Agent/platform contract → `moli`.

Use issue-backed durable reports for multi-session analysis. Archive meaningful resolved history.

## Inherited engineering governance

MolSys-AI repositories inherit applicable MOLI engineering policy through MolSysSuite. MolSysSuite may add stricter modeling-domain profiles.

A child repository may add stricter local requirements but must not silently weaken inherited policy. Applicability depends on the repository's actual capabilities; do not claim Python-package or release obligations before implementation exists.

## Architectural boundaries

- MolSys-AI ≠ MOLI Agent.
- Server inference/Software Knowledge ≠ MolSysSuite execution.
- Client SDK ≠ specialist agent.
- Agent may use Client/Server but does not require that topology for every operation.
- Client remains usable without MolSysSuite installed.
- Agent does not own/import server RAG internals.
- The umbrella repository coordinates; it does not absorb child implementation.
- Legacy code migrates by responsibility, not by directory.

## Cross-repository changes

When migration reveals a provider limitation, report it to the owning child repository and cross-link consumer work. Escalate to `molsys-ai` when a subsystem contract changes, to MolSysSuite when a modeling-domain contract changes, and to MOLI only at the platform boundary.

Do not edit synchronized child copies of this guide directly.
