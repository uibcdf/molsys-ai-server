# chat_api data directory

This directory is used for **generated** artifacts needed by the MolSys-AI chat API:

- `docs/`: a snapshot of documentation text files (`*.md`, `*.rst`, `*.txt`, `*.ipynb`) used for RAG.
- `rag_index.pkl`: the built embedding index.
- `rag_index.bm25.pkl`: optional BM25 sidecar for lexical retrieval (when built).
- `anchors.json`: extracted explicit MyST labels `(Label)=` used to build deep links to `https://www.uibcdf.org/<tool>/...#Label`.

These artifacts are intentionally not committed to git (see `.gitignore`).

## How to generate/update the corpus

From the `molsys-ai-server` repo root:

```bash
python dev/sync_rag_corpus.py --clean --build-api-surface --build-index --build-project-indices --build-anchors
```

If you want to control, per project, which directories/files are included in the snapshot, use a
corpus selection config:

```bash
python dev/sync_rag_corpus.py --corpus-config dev/corpus_config.toml --clean --build-index
```

Example config:

- `dev/corpus_config.toml.example`

By default this snapshots `docs/`, `doc/`, and `devguide/` from each upstream repo. Upstream
`examples/` directories are **excluded by default** because they can be stale and may reinforce
legacy APIs/aliases. If you explicitly want them:

```bash
python dev/sync_rag_corpus.py --include-examples ...
```

Optional quality layers (recommended as we move toward “code-aware” RAG):

- `--build-symbol-cards`: generates per-symbol cards under `docs/<project>/symbol_cards/` (AST; no imports).
- `--build-recipes`: extracts “recipes” from notebook code cells and upstream tests under `docs/<project>/recipes/`.
- `dev/digest_recipes_llm.py`: optional offline step to generate compact `docs/<project>/recipe_cards/` using the local engine.

Notes on notebook recipes:

- Notebook recipe extraction is **stateful across cells**: once a notebook imports the project (e.g. `import molsysmt as msm`),
  subsequent code cells are included even if they don't repeat the import (so tutorials are captured beyond the first cell).
- In addition to per-cell recipes under `recipes/notebooks/`, the builder also writes **section-level** recipes under
  `recipes/notebooks_sections/` that group multiple code cells under the same markdown heading (better context for workflows).
- The builder also writes **tutorial-level** recipes under `recipes/notebooks_tutorials/`:
  - one `tutorial.md` per notebook,
  - includes an outline of headings and a minimal “setup/bootstrap” snippet (imports + an entry cell when available),
  - section recipes include a `Tutorial:` pointer back to the tutorial recipe for that notebook.
- Section recipes are made more self-contained by stitching a small “preamble” when needed:
  - imports are collected from earlier cells,
  - a small number of earlier cells are included if they define variables used later in the section,
  - the header records which cells were stitched (`Preamble cells: ...`).
- Note: some historical MolSysMT notebooks use the alias `msmt` for `molsysmt`. Derived recipes normalize this to the
  canonical `msm` alias to avoid reinforcing a common LLM hallucination (`import msmt`).

Other recipe sources:

- `recipes/docstrings/`: doctest-style and fenced-code examples extracted from docstrings (AST; no imports).
- `recipes/markdown_snippets/`: fenced code blocks extracted from Markdown pages in the snapshot.

Large notebooks and long Markdown pages are included by default:

- text files larger than `--max-bytes` are **truncated** (instead of skipped),
- notebooks are compacted (outputs stripped) and bounded by `--max-bytes-ipynb`.

You can switch the size policy with:

- `--include-large-text skip` (skip files larger than the size limits)

The API-surface snapshot includes docstring excerpts (default: unlimited) and does not
impose artificial module/symbol caps unless you set:

- `--api-surface-max-modules`
- `--api-surface-max-symbols`

When `--build-api-surface` is enabled, a machine-readable symbol registry is also written:

- `server/chat_api/data/docs/_symbols.json`

To speed up index building on a multi-GPU host, use sharded indexing:

```bash
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
python dev/sync_rag_corpus.py --clean --build-api-surface --build-index --build-project-indices --build-index-parallel --index-devices 0,1,2 --build-anchors
```

Notes:

- This uses one process per GPU and merges partial indices into a single `rag_index.pkl`.
- For runtime (serving) it is recommended to keep embeddings on CPU:
  - set `MOLSYS_AI_EMBEDDINGS_DEVICE=cpu` in the service environment.

If you want stronger identifier matching, also build a BM25 sidecar:

```bash
python dev/sync_rag_corpus.py --clean --build-index --build-bm25
```

If your environment has restricted network access, and the embedding model is already cached,
set offline flags to avoid long retry loops:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python dev/sync_rag_corpus.py --clean --build-index --build-anchors
```

By default this syncs documentation content from the sibling repos:

- `../molsysmt`
- `../molsysviewer`
- `../pyunitwizard`
- `../topomt`

and writes a manifest to:

- `server/chat_api/data/docs/_manifest.json`

To audit coverage (and verify whether anything was excluded due to limits), run:

```bash
python dev/audit_rag_corpus.py --rescan-sources
```

Design decision record:

- `dev/decisions/ADR-019.md`
- `dev/decisions/ADR-020.md`
- `dev/decisions/ADR-021.md`

## Indexing derived layers

If you run `dev/digest_recipes_llm.py` to generate `recipe_cards/`, rebuild the indices so the new
documents are embedded and retrievable:

```bash
MOLSYS_AI_EMBEDDINGS=sentence-transformers \
python dev/sync_rag_corpus.py --build-index --build-project-indices --build-index-parallel --index-devices 0,1
```

Note on recipe-card generation:

- `dev/digest_recipes_llm.py` **purges** existing `docs/<project>/recipe_cards/` by default, so stale cards cannot linger
  after upstream docs/recipes move or disappear.
- Use `--no-purge` only for incremental experimentation (and then `--overwrite` controls whether existing outputs are replaced).

For production deployments, you may prefer storing these artifacts under a system
directory (e.g. `/var/lib/molsys-ai/...`) and set:

- `MOLSYS_AI_DOCS_DIR`
- `MOLSYS_AI_DOCS_INDEX`

in the service environment file.
