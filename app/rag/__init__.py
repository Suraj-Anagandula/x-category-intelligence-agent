"""Ask Intelligence: retrieval-augmented memory/search over previously
collected X posts.

    X Posts -> Clean/Normalize -> Embeddings -> Vector Store -> Retriever
    -> Relevant X Posts -> Groq -> Evidence-backed Answer

Every module here is lazily-imports its own heavy dependency
(`sentence-transformers`/`chromadb`, the `rag` optional-dependency group in
pyproject.toml) so the CLI, the core pipeline, and every other UI page keep
working with zero RAG dependencies installed - see `app/rag/embeddings.py`
and `app/rag/vector_store.py`.
"""

from __future__ import annotations
