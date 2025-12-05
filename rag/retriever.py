
"""Retrieval utilities for MolSys-AI.

The retriever exposes a simple interface:

    retrieve(query: str, k: int = 5) -> List[Document]

where Document is a small dataclass with content + metadata.
"""
