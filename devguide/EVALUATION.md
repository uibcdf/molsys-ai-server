# Server Evaluation

## Evaluation gates

Server releases are evaluated independently across inference, knowledge and operations.

### Inference

- availability and latency,
- streaming correctness,
- structured-output conformance,
- model identity and usage reporting,
- timeout and overload behavior.

### Knowledge

- retrieval recall and precision,
- symbol-card accuracy,
- citation correctness,
- documented-version alignment,
- unsupported-question handling,
- recipe usefulness.

### Documentation assistants

Maintain project-specific golden questions for MolSysMT, MolSysViewer, TopoMT and PharmacophoreMT. Test both conceptual explanations and code-oriented answers.

### Security and operations

- authentication and authorization enforcement,
- quota correctness,
- secret redaction,
- tenant isolation,
- health and rollback behavior,
- knowledge-release integrity.

## Release policy

A deployment candidate must report application commit, model configuration, API version and knowledge release. Regressions are reviewed by category; improved prose does not compensate for worse citation or API-symbol accuracy.