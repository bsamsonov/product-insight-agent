from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-m3"


class BgeM3Embedder:
    """Local multilingual embedder using BGE-M3 (dimension=1024).

    Downloads model on first use; cached in HuggingFace cache dir.
    """

    DIMENSION = 1024

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | None = None) -> None:
        _log.info("Loading sentence-transformer model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: np.ndarray = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()
