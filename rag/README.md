
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

- `build_index.build_index(source_dir, index_path)` (MVP implementation):
  - Reads `*.md` files under `source_dir`.
  - Splits content into small chunks (currently by paragraph).
  - Embeds chunks using a sentence-transformers model
    (see `rag/embeddings.py`, default: `all-MiniLM-L6-v2`).
  - For offline smoke tests (or minimal environments), embeddings can fall back
    to a lightweight hashing baseline via `MOLSYS_AI_EMBEDDINGS=hashing`. This
    keeps the system runnable but will significantly reduce retrieval quality.
  - Stores the resulting `Document` list (including embeddings) as a pickle
    file at `index_path`.

- `retriever.retrieve(query, k=5)`:
  - Embeds the query using the same embedding model.
  - Scores documents by cosine similarity against stored embeddings.
  - Returns up to `k` documents ordered by similarity.

- The `docs_chat` backend (`docs_chat/backend.py`) calls `build_index(...)`
  automatically on startup:
  - By default it looks for `.md` files under `docs_chat/data/docs`.
  - You can override the source directory with the environment variable
    `MOLSYS_AI_DOCS_DIR`.
  - `MOLSYS_AI_DOCS_INDEX` controls where the pickled index is stored.

To experiment quickly:

1. Place a few `.md` documents under `docs_chat/data/docs` (or set
   `MOLSYS_AI_DOCS_DIR` to another directory).
2. Run the docs-chat backend, e.g.:

   ```bash
   uvicorn docs_chat.backend:app --reload
   ```

3. Ensure the model server is running (stub or real) on
   `http://127.0.0.1:8001` (or set `MOLSYS_AI_MODEL_SERVER_URL`).

4. Call the endpoint:

   ```bash
   curl -X POST http://127.0.0.1:8000/v1/docs-chat \
        -H "Content-Type: application/json" \
        -d '{"query": "MolSysMT", "k": 5}'
   ```

The docs-chat backend will:
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
