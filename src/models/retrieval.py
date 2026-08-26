"""
RetrievalHit: a typed result returned by the Retriever.

Replaces the untyped dict[str, Any] that was used previously.
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalHit:
    """A single chunk returned by a similarity search."""
    text: str
    metadata: dict[str, Any]    # keys: page, chunk_index, chunk_type,
                                #       figure_label, token_count, source
    distance: float             # cosine distance (lower = closer)
    score: float                # 1 - distance (higher = more relevant)
