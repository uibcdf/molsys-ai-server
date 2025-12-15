
# MolSys-AI RAG Layer

This directory holds the Retrieval-Augmented Generation (RAG) components.

The goal is to allow the agent to:
- consult documentation,
- read code/docstrings,
- retrieve relevant snippets,
- and feed those into the model as context.

## Modules

- `build_index.py`:
  - Will contain functions to:
    - read documents (e.g. Sphinx-generated text/HTML or markdown),
    - split them into chunks,
    - embed those chunks using an embedding model,
    - store them in a FAISS index.

- `retriever.py`:
  - Will expose a simple interface such as:

    ```python
    def retrieve(query: str, k: int = 5) -> list[Document]:
        ...
    ```

  - `Document` is expected to be a small dataclass or similar with:
    - `content: str`
    - `metadata: dict` (e.g. source file, section, etc.).

- `embeddings.py`:
  - Wraps the embedding model used to turn text into vectors.
  - The choice of model is documented in ADR-009 / OPEN_QUESTIONS.
  - This module should hide backend details so that we can swap
    embedding models later without changing the rest of the code.

## Index location

- The current MVP stores a pickled list of `Document` objects on disk
  (default: `data/rag_index.pkl`).
- A FAISS-based index is planned (see ADR-004), but not implemented yet.

## MVP behaviour and quickstart

## Corpus inputs (literal vs derived)

The RAG index is built from a **corpus directory** (configured via `MOLSYS_AI_DOCS_DIR`).

There are two complementary kinds of corpus content:

1. **Literal snapshot (current default)**:
   - plain documentation files copied verbatim from the live repos (`*.md`, `*.rst`, `*.txt`).
   - this is what `dev/sync_rag_corpus.py` produces.

2. **Derived corpus (planned)**:
   - additional “digested” artifacts generated from the literal snapshot (e.g. summaries, FAQs,
     concept cards, API overviews).
   - these files are still plain text (so they can be indexed like any other doc), but must include
     provenance in their content/metadata (what sources they summarize, when, and with which model/prompt).

The derived corpus is planned because it often improves retrieval quality and answer structure,
but it must never replace the literal snapshot; it is an additive layer.

- `build_index.build_index(source_dir, index_path)` (MVP implementation):
  - Reads `*.md`, `*.rst`, and `*.txt` files under `source_dir`.
  - For notebooks (`*.ipynb`), it indexes markdown and (by default) code cell sources
    to capture executable API examples. You can disable code cells with:
    - `MOLSYS_AI_RAG_IPYNB_INCLUDE_CODE=0`
  - Splits content into small chunks with a simple Markdown-aware heuristic:
    - tracks headings as section context,
    - merges paragraphs into chunks by character size.
  - Chunking can be tuned with:
    - `MOLSYS_AI_RAG_CHUNK_MAX_CHARS` (default: 1600)
    - `MOLSYS_AI_RAG_CHUNK_MIN_CHARS` (default: 400)
  - Embeds chunks using a sentence-transformers model
    (see `server/rag/embeddings.py`, default: `all-MiniLM-L6-v2`).
  - For offline smoke tests (or minimal environments), embeddings can fall back
    to a lightweight hashing baseline via `MOLSYS_AI_EMBEDDINGS=hashing`. This
    keeps the system runnable but will significantly reduce retrieval quality.
  - Stores the resulting `Document` list (including embeddings) as a pickle
    file at `index_path`.

- `retriever.retrieve(query, k=5)`:
  - Embeds the query using the same embedding model.
  - Scores documents by cosine similarity against stored embeddings.
  - Optional hybrid rerank:
    - `MOLSYS_AI_RAG_HYBRID_WEIGHT` (default: 0.15) mixes in a lightweight lexical score
      to boost exact identifier matches (e.g. function names like `molsysmt.structure.get_rmsd`).
  - To reduce repetition, retrieval limits how many chunks can be returned from a single
    source file (default: 3). Tune with `MOLSYS_AI_RAG_MAX_CHUNKS_PER_SOURCE`.
  - Returns up to `k` documents ordered by similarity.

- The chat API (`server/chat_api/backend.py`) calls `build_index(...)`
  automatically on startup:
  - By default it looks for `.md` files under `server/chat_api/data/docs`.
  - You can override the source directory with the environment variable
    `MOLSYS_AI_DOCS_DIR`.
  - `MOLSYS_AI_DOCS_INDEX` controls where the pickled index is stored.

To experiment quickly:

1. Place a few `.md` documents under `server/chat_api/data/docs` (or set
   `MOLSYS_AI_DOCS_DIR` to another directory).
2. Run the chat API, e.g.:

   ```bash
   uvicorn chat_api.backend:app --reload
   ```

3. Ensure the model server is running (stub or real) on
   `http://127.0.0.1:8001` (or set `MOLSYS_AI_ENGINE_URL`).

4. Call the endpoint:

   ```bash
   curl -X POST http://127.0.0.1:8000/v1/chat \
        -H "Content-Type: application/json" \
        -d '{"query": "MolSysMT", "k": 5}'
   ```

The chat API will:
- build an embedding index from the `.md` documents,
- retrieve simple matches for the query via `rag.retriever.retrieve`,
- send a prompt with the retrieved excerpts to the configured model server,
- and return the model's reply.

## Design goals

- Keep the RAG logic modular and independent from the agent.
- Allow experimentation with:
  - different chunking strategies,
  - different embedding models,
  - different retrieval heuristics.

For the MVP, the RAG layer can start very simple (small subset of docs,
basic embedding model) and grow over time.
