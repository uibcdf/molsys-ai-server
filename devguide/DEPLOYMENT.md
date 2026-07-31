# Deployment and Operations

## Topology

Separate build environments from inference environments. Corpus and index artifacts are built offline or in controlled CI, validated, versioned and then deployed to serving hosts.

Typical components:

- reverse proxy and TLS,
- FastAPI application,
- inference backend such as vLLM or another replaceable provider,
- knowledge-service workers,
- immutable corpus/index artifacts,
- authentication and quota store,
- metrics, logs and traces.

## Isolation

Inference hosts must not import MolSysSuite scientific runtimes. Documentation builds and API extraction occur in dedicated environments. Public documentation assistants never receive private molecular project data.

## Configuration

Configuration is environment-specific and external to source code. Secrets use protected deployment facilities. Startup validation checks model availability, API compatibility and knowledge-release integrity.

## Health

Expose liveness, readiness and dependency status separately. Readiness should fail when required models or selected knowledge releases are unavailable.

## Observability

Record request counts, latency, error codes, model utilization, retrieval latency, corpus release, quota events and saturation. Payload content is excluded from routine metrics and redacted from logs.

## Reliability

- graceful shutdown and request draining,
- bounded queues and timeouts,
- overload responses rather than unbounded waiting,
- rollback to previous model or knowledge releases,
- compatibility tests before deployment,
- backups for configuration and authorization state.

## Hardware profiles

Document supported CPU-only, single-GPU and multi-GPU deployments. Model loading, quantization and batching choices remain deployment policy rather than public API behavior.

## Release procedure

1. Build and test application artifacts.
2. Build or select immutable knowledge release.
3. Run API, retrieval and safety benchmarks.
4. Deploy to staging and verify capabilities.
5. Promote with rollback references.
6. Monitor regressions and resource saturation.