# Deployment and Operations

## Topology

Separate build environments from inference environments. Corpus and index artifacts are built offline or in controlled CI, validated, versioned and then deployed to serving hosts.

Typical components:

- public ingress through Cloudflare Tunnel or a conventional reverse proxy,
- FastAPI application,
- inference backend such as vLLM or another replaceable provider,
- knowledge-service workers,
- immutable corpus/index artifacts,
- authentication and quota store,
- metrics, logs and traces.

## Existing Cloudflare Tunnel infrastructure

**Confirmed and reusable.** The current deployment already exposes `https://api.uibcdf.org` through a Cloudflare Tunnel while inbound ports `80` and `443` remain filtered by the upstream data-center network.

The existing topology is:

```text
public clients and documentation widgets
        │
        ▼
https://api.uibcdf.org
        │
        ▼
Cloudflare Tunnel
        │
        ▼
FastAPI public service on 127.0.0.1:8000
        │
        ▼
private model service on 127.0.0.1:8001
```

This infrastructure should be preserved during the server transformation. It is not tied conceptually to the legacy `/v1/chat` endpoint. The tunnel should become the public ingress for the modular `molsys-ai-server` API, including health, capabilities, inference, knowledge and documentation-assistant routes.

The model backend must remain bound to localhost and must never be exposed directly through the tunnel. Scientific MolSysSuite execution also remains outside the server and therefore outside this ingress path.

The current operational runbook remains in `dev/DEPLOY_API.md`. The development guide records the target architecture; the runbook records host-specific commands and current deployment details.

## Migration strategy for the public API

The existing deployment can evolve without interrupting the published documentation chatbot:

1. Keep `https://api.uibcdf.org/v1/chat` working for current documentation widgets.
2. Add versioned health and capability endpoints.
3. Introduce inference, knowledge and assistant APIs alongside the legacy chat route.
4. Move internal implementation behind stable service boundaries.
5. Deprecate legacy routes only after clients and documentation widgets have migrated.

A single tunnel should normally expose the public FastAPI gateway rather than creating a separate tunnel for each internal service.

## Cloudflare responsibilities

Cloudflare may provide:

- public TLS termination and hostname routing,
- rate limiting and abuse controls,
- optional WAF rules,
- traffic-level metrics,
- protection without opening inbound ports on the serving host.

Cloudflare does not replace application authorization. FastAPI remains responsible for endpoint permissions, token validation, quotas, policy enforcement and structured audit events.

The documentation assistant may remain public with strict rate limiting. Copilot inference and other non-public capabilities should require authenticated access. The internal model endpoint is never public.

## Tunnel configuration and secrets

Repository documentation should include a redacted configuration template, for example:

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: /etc/cloudflared/<TUNNEL_UUID>.json

ingress:
  - hostname: api.uibcdf.org
    service: http://127.0.0.1:8000
  - service: http_status:404
```

The following must not be committed:

- tunnel credential JSON files,
- API tokens,
- origin certificates or private keys,
- host-specific secrets.

The operational documentation should eventually cover installation, systemd management, restart behavior, health verification, credential rotation and disaster recovery for `cloudflared`.

## Isolation

Inference hosts must not import MolSysSuite scientific runtimes. Documentation builds and API extraction occur in dedicated environments. Public documentation assistants never receive private molecular project data.

All public application services should bind to localhost when reached through Cloudflare Tunnel. Only explicitly required internal services may listen on other protected interfaces.

## Configuration

Configuration is environment-specific and external to source code. Secrets use protected deployment facilities. Startup validation checks model availability, API compatibility, tunnel-facing application readiness and knowledge-release integrity.

## Health

Expose liveness, readiness and dependency status separately. Readiness should fail when required models or selected knowledge releases are unavailable.

The public health check should verify the FastAPI gateway without disclosing sensitive deployment details. Internal checks may separately verify the model backend, knowledge release and tunnel service.

## Observability

Record request counts, latency, error codes, model utilization, retrieval latency, corpus release, quota events and saturation. Payload content is excluded from routine metrics and redacted from logs.

Operational monitoring should distinguish:

- Cloudflare ingress failures,
- tunnel connectivity failures,
- public FastAPI failures,
- model-backend failures,
- knowledge-service failures.

## Reliability

- graceful shutdown and request draining,
- bounded queues and timeouts,
- overload responses rather than unbounded waiting,
- rollback to previous model or knowledge releases,
- compatibility tests before deployment,
- backups for configuration and authorization state,
- supervised `cloudflared` operation and restart after host reboot,
- verification that public routing still targets the intended local service.

## Alternative ingress

A conventional Caddy or nginx deployment on inbound `443` remains a valid future option if the data-center firewall changes. The application architecture should not depend on Cloudflare-specific request semantics.

Cloudflare Tunnel is therefore the current preferred ingress, not an irreversible product dependency.

## Hardware profiles

Document supported CPU-only, single-GPU and multi-GPU deployments. Model loading, quantization and batching choices remain deployment policy rather than public API behavior.

## Release procedure

1. Build and test application artifacts.
2. Build or select immutable knowledge release.
3. Run API, retrieval and safety benchmarks.
4. Deploy to staging and verify internal services.
5. Verify the public route through Cloudflare Tunnel.
6. Promote with rollback references.
7. Monitor regressions, tunnel health and resource saturation.
