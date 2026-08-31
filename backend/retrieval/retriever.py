"""
Retriever: turns a text query into a ranked list of relevant chunks.

Steps:
  1. Embed the query (with BGE/E5 query prefix when applicable).
  2. Search ChromaDB by cosine similarity.
  3. Optionally filter by chunk type (text / image_description).
  4. Format results as a context string for the LLM prompt.
"""
import logging
from typing import TYPE_CHECKING

from .embedder import Embedder
from .vector_store import VectorStore
from ..models.retrieval import RetrievalHit

if TYPE_CHECKING:
    from ..config.config import Config

logger = logging.getLogger(__name__)


class Retriever:
    """
    Orchestrates the retrieval step: query → embeddings → ChromaDB → context.
    """

    def __init__(
        self,
        config: "Config",
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self.config = config
        self._embedder = embedder
        self._store = vector_store

    # *** Methods ******************************
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        include_images: bool = True,
    ) -> list[RetrievalHit]:
        """
        Return the top-k most relevant chunks for a query.

        Args:
            query         : user question (plain text)
            top_k         : number of chunks to return (default: config.top_k)
            include_images: if False, image description chunks are excluded

        Returns:
            List of RetrievalHit objects sorted by descending score.
        """
        k = top_k or self.config.top_k
        query_emb = self._embedder.embed_query(query)

        # Fetch extra candidates when we will filter some out
        fetch_k = k * 2 if not include_images else k
        raw_hits = self._store.query(query_embedding=query_emb, n_results=fetch_k)

        hits = [
            RetrievalHit(
                text=h["text"],
                metadata=h["metadata"],
                distance=h["distance"],
                score=h["score"],
            )
            for h in raw_hits
        ]

        if not include_images:
            hits = [
                h for h in hits
                if h.metadata.get("chunk_type") != "image_description"
            ]

        hits = hits[:k]
        logger.debug(f"Retrieved {len(hits)} chunks for query '{query[:60]}...' (include_images={include_images})")
        return hits

    def format_context(self, hits: list[RetrievalHit]) -> str:
        """
        Format retrieved chunks into a context block ready for the LLM prompt.
        Each chunk is prefixed with a header indicating its source and page.
        """
        parts: list[str] = []
        for hit in hits:
            meta = hit.metadata
            ctype = meta.get("chunk_type", "text")
            page = meta.get("page", "?")
            label = meta.get("figure_label", "")

            if ctype == "image_description":
                header = f"[{label or 'Figure'} — page {page}]"
            else:
                header = f"[Page {page}]"

            parts.append(f"{header}\n{hit.text}")

        return "\n\n---\n\n".join(parts)

    def retrieve_and_format(
        self,
        query: str,
        top_k: int | None = None,
        include_images: bool = True,
    ) -> tuple[str, list[RetrievalHit]]:
        """
        Convenience method: returns both the formatted context string
        and the raw list of hits.
        """
        hits = self.retrieve(query, top_k=top_k, include_images=include_images)
        context = self.format_context(hits)
        return context, hits
