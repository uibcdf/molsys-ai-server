
"""Index building utilities for MolSys-AI RAG.

In the MVP this will:
- read Sphinx-generated HTML/text,
- chunk it,
- embed chunks using a chosen embedding model,
- store them in a FAISS index.
"""
