"""
Wrapper around sentence-transformers for generating embeddings.
Everything runs locally — no external API calls.

Dependency: sentence-transformers
"""
import logging
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from ..config.config import Config

logger = logging.getLogger(__name__)

class Embedder:
    """
    Wraps SentenceTransformer for document and query embedding.

    Notes:
    - BGE models benefit from a model-specific query prefix.
    - All embeddings are L2-normalised, making dot product equivalent
      to cosine similarity.
    """

    def __init__(self, config: "Config") -> None:
        self.config = config
        logger.info(f"Loading embedding model '{config.embedding_model}' (device={config.embedding_device})")
        self._model = SentenceTransformer(
            config.embedding_model,
            device=config.embedding_device,
        )
        self._dim: int = self._model.get_embedding_dimension()
        logger.info(f"Embedding model ready — dimension: {self._dim}")

    # *** Methods ******************************
    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts.

        Returns:
            np.ndarray of shape (N, dim), dtype float32, L2-normalised.
        """
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a user query.
        BGE and E5 models use a dedicated query prefix distinct from
        the passage prefix, which improves retrieval quality.
        """
        q = self._apply_query_prefix(query)
        return self._model.encode(
            q,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    # *** Helpers ******************************
    def _apply_query_prefix(self, query: str) -> str:
        """Add the recommended query prefix for BGE / E5 models."""
        model_name = self.config.embedding_model.lower()
        if "bge" in model_name:
            return f"Represent this sentence for searching relevant passages: {query}"
        if "e5" in model_name:
            return f"query: {query}"
        return query

    # *** Properties ******************************
    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def sentence_model(self) -> SentenceTransformer:
        """Direct access to the underlying SentenceTransformer (used by the chunker)."""
        return self._model
