# Knowledge Service

## Mission

Provide grounded, project-aware access to MolSysSuite documentation, public APIs, recipes and source provenance.

## Knowledge assets

- Markdown and Sphinx documentation,
- public API surfaces extracted without importing upstream packages,
- symbol cards,
- tutorials and examples,
- notebook, test and docstring recipes,
- source anchors and repository commits,
- project and release metadata.

## Build lifecycle

```text
source synchronization
→ normalization
→ API and recipe extraction
→ chunk and card generation
→ validation
→ index construction
→ benchmark evaluation
→ immutable release publication
```

Builds should run offline from pinned source revisions. Inference hosts consume generated releases rather than importing MolSysSuite.

## Retrieval

The service may combine lexical, dense and structured retrieval. Project, version, symbol type and source filters are first-class. Reranking is introduced only when benchmarks justify its cost.

## Guardrails

- Validate referenced symbols against API surfaces.
- Re-read authoritative symbol cards before producing code.
- Distinguish documented behavior from inferred guidance.
- Return citations with stable source identifiers and anchors.
- Report corpus and source revisions with every grounded answer.

## Contracts

First-class operations include:

- retrieve source fragments,
- answer with citations,
- fetch a symbol card,
- search symbols,
- validate symbol existence,
- retrieve recipes,
- report corpus metadata and capabilities.

## Releases

Knowledge releases are immutable and identified by project, semantic release number, source commits and artifact checksums. Rollback selects an earlier release; it does not mutate an existing one.

## Evaluation

Maintain benchmark sets for documentation questions, symbol accuracy, recipe usefulness, citation precision and unsupported-question handling across each MolSysSuite project.