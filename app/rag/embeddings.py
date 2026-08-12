"""Local embedding model wrapper for Ask Intelligence (RAG).

Lazily imports `sentence_transformers` inside `_ensure_model()`, exactly
like `app/llm.py`'s provider `_ensure_client()` pattern - so this module
stays importable, and every non-RAG code path never pays the import/model-
load cost, even when `sentence-transformers`/`torch` aren't installed at all.
"""

from __future__ import annotations

from typing import Any

from app.exceptions import RAGError

#: all-MiniLM-L6-v2: 384-dim, ~80MB, fast on CPU, well-suited to short
#: informal text (tweets) - the standard "good enough, cheap enough"
#: default at this scale (thousands, not millions, of vectors). Chosen over
#: a larger model (e.g. all-mpnet-base-v2) since this app has no GPU
#: dependency anywhere else and inference needs to stay fast on CPU.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Wraps a single sentence-transformers model, loaded once on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RAGError(
                    "sentence-transformers is not installed. "
                    'Run `pip install -e ".[rag]"` (or `pip install -r requirements.txt`) '
                    "to enable Ask Intelligence."
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` into vectors, one per input string, in order.

        Returns `[]` for an empty input without loading the model at all -
        callers with nothing to embed never pay the model-load cost.
        """
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(list(texts), convert_to_numpy=True)
        return [[float(component) for component in vector] for vector in vectors]
