# API Contract v1 (stability target)

This document defines the minimal stable HTTP contract that external clients
(`molsys-ai` CLI) depend on.

Backends (vLLM model, RAG implementation, infra, deployment) may evolve without
breaking this contract. Breaking changes must be introduced as a new version
(e.g. `/v2/...`) or be kept backwards-compatible.

## 1) Authentication

### 1.1 API key header (standard)

When an endpoint is protected by API keys, clients authenticate with:

- `Authorization: Bearer <api_key>`

Alternative header (also accepted):

- `X-API-Key: <api_key>`

### 1.2 Key tiers (`u1_` / `u2_`)

Key prefixes are a **client-side convention**:

- `u1_...`: lab/investigator key (CLI may try LAN-first then fall back to public API)
- `u2_...`: external key (CLI uses public API only)

Important: server-side authorization/rate limits are always enforced by the full
key value, not by the prefix alone.

## 2) Endpoints

### 2.1 `POST /v1/chat`

Purpose: public chat endpoint for end users (CLI + docs widget).

Request JSON:

```json
{
  "messages": [
    {"role": "system", "content": "..." },
    {"role": "user", "content": "..." }
  ],
  "k": 5,
  "client": "cli",
  "rag": "auto",
  "sources": "auto",
  "debug": false,
  "rag_config": {
    "bm25_weight": 0.25,
    "hybrid_weight": 0.15
  }
}
```

Response JSON:

```json
{
  "answer": "...",
  "sources": [
    {"id": 1, "path": "molsysmt/docs/...", "section": "# ...", "label": "Some_Label", "url": "https://www.uibcdf.org/molsysmt/...#Some_Label"}
  ],
  "debug": {
    "index_used": "project:molsysmt",
    "retrieval_ms": 12.3
  }
}
```

Notes:

- `messages` is required and represents the full conversation history.
- `k` is optional (default server-side) and controls how many snippets are retrieved when RAG is enabled.
- `client` is optional (`"widget"` | `"cli"`). If omitted, treat as `"widget"`.
- `rag` is optional (`"on"` | `"off"` | `"auto"`). `"auto"` uses an LLM router to decide.
- `sources` is optional (`"on"` | `"off"` | `"auto"`). When enabled, `sources` align with citations `[1]`, `[2]`, ...
- `debug` is optional. When enabled, the server may include a `debug` field in the response. This is intended for
  internal benchmarking and is typically disabled in public deployments.
- `rag_config` is optional. It is an internal/benchmarking override for retrieval weights/knobs and is typically ignored
  unless explicitly enabled server-side.

Errors:

- `400`: malformed request (e.g. missing messages)
- `401`: invalid/missing API key (when enabled)
- `500`: server-side failure

### 2.2 `POST /v1/engine/chat`

Purpose: internal model engine endpoint used by the chat API for generation and router calls.

Request JSON:

```json
{
  "messages": [
    {"role": "system", "content": "..." },
    {"role": "user", "content": "..." }
  ],
  "generation": {"max_tokens": 128, "temperature": 0.0, "top_p": 1.0}
}
```

Response JSON:

```json
{
  "content": "..."
}
```

Notes:

- `messages` is required and represents the full conversation history.
- `content` is the assistant reply.
- `generation` is optional and allows callers to override sampling parameters (useful for router/classifier calls).

Errors:

- `400`: malformed request (e.g. missing messages)
- `401`: invalid/missing API key (if enabled)
- `500`: server-side failure

## 3) Backwards-compatible evolution rules

- Adding new optional request fields is allowed.
- Adding new response fields is allowed.
- Changing field names, types, or required/optional status is breaking.
- Breaking changes require a new API version (e.g. `/v2/...`) or a compatibility layer.
