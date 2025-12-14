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

Purpose: low-level chat completion for internal services (model backend).

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
- `content` is a plain string with the assistant reply.
- `generation` is optional and allows callers to override sampling parameters (useful for router/classifier calls).

Errors:

- `400`: malformed request (e.g. missing messages)
- `401`: invalid/missing API key (when enabled)
- `500`: server-side failure

### 2.2 `POST /v1/docs-chat`

Purpose: documentation chatbot endpoint used by the web widget and optionally
by the CLI. This is the recommended endpoint for end-user chat, since it can
use RAG and optionally return sources.

Request JSON (single-turn):

```json
{
  "query": "How do I ...?",
  "k": 5
}
```

Request JSON (multi-turn; optional for clients that support it):

```json
{
  "messages": [
    {"role": "user", "content": "..." }
  ],
  "k": 5,
  "client": "cli",
  "rag": "auto",
  "sources": "auto"
}
```

Response JSON:

```json
{
  "answer": "...",
  "sources": [
    {"id": 1, "path": "molsysmt/docs/...", "section": "# ...", "label": "Some_Label", "url": "https://www.uibcdf.org/molsysmt/...#Some_Label"}
  ]
}
```

Notes:

- `sources` is optional and may be an empty list (for example, if no snippets were retrieved).
- Source ids align with bracketed citations in the answer (`[1]`, `[2]`, ...).
- `client` is optional (`"widget"` | `"cli"`). If omitted, treat as `"widget"`.
- `rag` is optional (`"on"` | `"off"` | `"auto"`). For CLI, `"auto"` uses an LLM router to decide; widget uses `"on"`.
- `sources` is optional (`"on"` | `"off"` | `"auto"`). For CLI, `"auto"` decides whether to show citations/sources.

Errors:

- `400`: malformed request (missing query/messages)
- `401`: invalid/missing API key (if enabled)
- `500`: server-side failure

## 3) Backwards-compatible evolution rules

- Adding new optional request fields is allowed.
- Adding new response fields is allowed.
- Changing field names, types, or required/optional status is breaking.
- Breaking changes require a new API version (e.g. `/v2/...`) or a compatibility layer.
