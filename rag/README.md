
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

- The FAISS index will likely be stored on disk under a directory
  such as `rag/index/`.
- The exact location and naming will be configurable later.

## Design goals

- Keep the RAG logic modular and independent from the agent.
- Allow experimentation with:
  - different chunking strategies,
  - different embedding models,
  - different retrieval heuristics.

For the MVP, the RAG layer can start very simple (small subset of docs,
basic embedding model) and grow over time.
