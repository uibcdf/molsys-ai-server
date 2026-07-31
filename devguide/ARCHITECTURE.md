# Target Server Architecture

> **Design status**
>
> This document combines confirmed repository decisions with the intended future role of the server. Any remembered feature not yet verified must remain explicitly marked as provisional.

## Mission

`molsys-ai-server` provides shared remote capabilities that can be consumed by the local scientific copilot and by public documentation assistants.

## Main services

```text
molsys-ai-server
├── inference service
│   ├── model backend
│   ├── generation API
│   └── streaming
├── knowledge service
│   ├── documentation corpus
│   ├── API surfaces and symbol cards
│   ├── recipes and tutorials
│   ├── hybrid retrieval
│   ├── citations
│   └── API-symbol guardrails
├── documentation assistants
│   ├── MolSysMT
│   ├── MolSysViewer
│   ├── TopoMT
│   ├── PharmacophoreMT
│   └── future MolSysSuite tools
├── authentication and quotas
└── deployment and observability
```

## Confirmed principles inherited from the current repository

- The inference environment remains isolated from MolSysSuite toolchains.
- Corpus construction is reproducible and can run offline.
- API information can be extracted with AST without importing upstream packages.
- Retrieval is project-aware and code-aware.
- Symbol verification and symbol re-reading reduce invented or misused APIs.
- Documentation answers should provide sources and stable citations.

## Server boundary

The server must not own:

- active molecular systems,
- a user's live MolSysViewer canvas,
- local file access,
- MolSysSuite tool execution,
- the scientific agent loop,
- reproducible local project history.

Those belong to `molsys-ai` in the user's environment.

## Knowledge-service role

RAG remains useful, but it is an internal retrieval technique rather than the identity of the product. The service should expose grounded knowledge through stable contracts such as:

- answer a documentation question,
- retrieve relevant source fragments,
- retrieve a symbol card,
- retrieve recipes for a capability,
- validate whether a documented symbol exists,
- report corpus versions and provenance.

## Documentation assistants

Each MolSysSuite project should be able to embed a specialized chatbot in its documentation. These assistants use the shared server but apply project filters, presentation rules and tool-specific sources.

They are not reduced versions of the local copilot. They are public knowledge products with a narrower permission model:

- no local scientific execution,
- no private molecular data,
- no persistent molecular session,
- grounded explanations and code examples.

## Authentication and user profiles

**Remembered — verify:** earlier plans may have included tokenized profiles or per-user access tokens.

The server should support opaque credentials and server-side policy without depending on a particular client profile format. Profile definitions belong to the client side; authentication and authorization belong here.
