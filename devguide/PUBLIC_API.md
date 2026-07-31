# Public API

## Status

**Proposed contract groups.** Exact paths and schemas will be finalized together with `molsys-ai-client`.

## Principles

- Versioned, typed and capability-discoverable.
- Inference and knowledge are separate contracts.
- Scientific execution and live molecular state are excluded.
- Streaming events use the same semantic result model as non-streaming calls.

## API groups

### Health and capabilities

Report service health, API versions, available model roles, knowledge projects, corpus versions, streaming support, authentication requirements and request limits.

### Inference

Accept messages or structured task context and return model output, usage, model identity, finish reason and safety/policy metadata. Structured-output requests declare a schema or supported output type.

### Knowledge

Support project-scoped retrieval, grounded answers, symbol lookup, recipe retrieval, symbol validation and corpus metadata.

### Documentation assistants

Provide a configured question-answering surface for each MolSysSuite project while retaining citations and project boundaries.

### Streaming

Events include generation deltas, retrieval status, citations, usage, completion and structured errors.

## Errors

Errors use stable codes for authentication, authorization, quota, validation, unsupported capability, corpus unavailable, model unavailable, timeout, overload and internal failure. Secrets and internal stack traces are never returned.

## Compatibility

Clients negotiate API major version and optional capabilities. New optional fields are additive; incompatible changes require a new major version. Deprecations publish replacement contracts and removal windows.

## Idempotency and tracing

Requests may carry client request IDs and idempotency keys where appropriate. Server trace IDs are returned for operational support without exposing sensitive payloads.