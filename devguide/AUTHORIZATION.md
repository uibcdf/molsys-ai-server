# Authentication, Authorization and Quotas

## Boundary

Clients may organize credentials through named profiles, but the server owns identity verification, authorization and quota enforcement.

## Credentials

Initial deployments may use opaque bearer tokens. The API contract must permit later replacement by institutional identity or delegated authorization without changing scientific clients.

Tokens represent access credentials, not user preferences or scientific memory.

## Authorization model

Authorization may consider:

- authenticated subject,
- organization or project,
- service and model role,
- documentation corpus,
- request limits,
- administrative capability.

The server does not authorize local MolSysSuite tool execution; that remains local policy in `molsys-ai`.

## Quotas

Quotas may apply to requests, generated tokens, concurrent streams, model classes or time windows. Limit responses include stable error codes and reset information without leaking other users’ activity.

## Token lifecycle

Support creation, rotation, revocation and expiration. Store only protected token representations server-side. Raw tokens must not appear in logs, metrics or exception reports.

## Audit events

Record authentication failures, token lifecycle changes, authorization denials, quota changes and administrative actions. Avoid retaining user prompts as part of security audit records.

## Multi-tenant isolation

Corpus access, quotas and operational records must be scoped explicitly. Cached inference or retrieval results must not leak content across tenants or private configurations.