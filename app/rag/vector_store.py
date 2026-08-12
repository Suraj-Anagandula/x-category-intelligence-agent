"""ChromaDB-backed persistent vector store wrapper for Ask Intelligence.

Deliberately never lets Chroma compute its own embeddings (no
`embedding_function` registered on the collection) - every call site here
passes embeddings explicitly, computed by the same `Embedder` instance for
both indexing (`app/rag/indexer.py`) and querying (`app/rag/retriever.py`),
so there is never a model mismatch between index-time and query-time
vectors. This also sidesteps Chroma's own default embedding function
(onnxruntime-based), which this design doesn't need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.exceptions import RAGError

#: One shared collection across all categories - `category` is a metadata
#: field filtered via Chroma's `where` clause, which is simpler and more
#: flexible than per-category collections (a question can legitimately span
#: more than one category).
COLLECTION_NAME = "tweets"


class Hit:
    """One raw result row from a vector-store query - not yet filtered by
    similarity threshold or reranked (see `app/rag/retriever.py`)."""

    __slots__ = ("id", "document", "metadata", "distance")

    def __init__(self, id: str, document: str, metadata: dict, distance: float) -> None:
        self.id = id
        self.document = document
        self.metadata = metadata
        self.distance = distance


class VectorStore:
    """Thin wrapper around one persistent Chroma collection."""

    def __init__(self, persist_dir: Path | str, collection_name: str = COLLECTION_NAME) -> None:
        self._persist_dir = str(persist_dir)
        self._collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    def _ensure_collection(self) -> Any:
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RAGError(
                    "chromadb is not installed. "
                    'Run `pip install -e ".[rag]"` (or `pip install -r requirements.txt`) '
                    "to enable Ask Intelligence."
                ) from exc
            try:
                self._client = chromadb.PersistentClient(path=self._persist_dir)
                # Explicit cosine distance, set at creation time: without this,
                # Chroma defaults to squared L2 over whatever scale the
                # embedder happens to produce, which is meaningless to turn
                # into a "similarity" - cosine distance is always `1 -
                # cosine_similarity`, so `retriever.py` can convert it
                # directly regardless of whether the embedder's vectors are
                # unit-normalized.
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name, metadata={"hnsw:space": "cosine"}
                )
            except Exception as exc:  # noqa: BLE001 - any chromadb failure is non-fatal upstream
                raise RAGError(f"Failed to open the vector store: {exc}") from exc
        return self._collection

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Idempotent on `ids` - re-upserting an existing id overwrites its
        document/metadata/embedding rather than duplicating it, which is
        exactly what makes same-day-rerun re-indexing safe with zero
        hand-rolled dedup logic."""
        if not ids:
            return
        collection = self._ensure_collection()
        try:
            collection.upsert(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )
        except Exception as exc:  # noqa: BLE001
            raise RAGError(f"Failed to upsert into the vector store: {exc}") from exc

    def query(
        self, query_embedding: list[float], top_k: int, where: dict | None = None
    ) -> list[Hit]:
        """Returns `[]` (never raises) when the store is empty - an empty
        index is a normal, expected state (fresh install, nothing indexed
        yet), not an error."""
        collection = self._ensure_collection()
        count = collection.count()
        if count == 0:
            return []

        try:
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                where=where,
            )
        except Exception as exc:  # noqa: BLE001
            raise RAGError(f"Vector store query failed: {exc}") from exc

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[Hit] = []
        for id_, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            hits.append(Hit(id=id_, document=document, metadata=metadata or {}, distance=distance))
        return hits

    def count(self) -> int:
        collection = self._ensure_collection()
        return collection.count()
