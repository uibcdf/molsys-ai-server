# chat_api data directory

This directory is used for **generated** artifacts needed by the MolSys-AI chat API:

- `docs/`: a snapshot of documentation text files (`*.md`, `*.rst`, `*.txt`, `*.ipynb`) used for RAG.
- `rag_index.pkl`: the built embedding index.
- `anchors.json`: extracted explicit MyST labels `(Label)=` used to build deep links to `https://www.uibcdf.org/<tool>/...#Label`.

These artifacts are intentionally not committed to git (see `.gitignore`).

## How to generate/update the corpus

From the `molsys-ai-server` repo root:

```bash
python dev/sync_rag_corpus.py --clean --build-index --build-anchors
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

For production deployments, you may prefer storing these artifacts under a system
directory (e.g. `/var/lib/molsys-ai/...`) and set:

- `MOLSYS_AI_DOCS_DIR`
- `MOLSYS_AI_DOCS_INDEX`

in the service environment file.
