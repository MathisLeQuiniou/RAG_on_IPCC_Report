"""
ChromaDB interface for local, persistent vector storage.

Each chunk is stored with:
  - Its text (document field)
  - Its embedding (provided explicitly — no ChromaDB embedding function)
  - Metadata: page, chunk type, figure label, token count, source filename

Dependency: chromadb
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from chromadb import PersistentClient
from chromadb.config import Settings

from ..models.chunk import Chunk

if TYPE_CHECKING:
    from ..config.config import Config

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Wraps a persistent ChromaDB collection.

    Chunk metadata schema:
        page         : int  — page number in the source PDF
        chunk_index  : int  — sequential index within its source block
        chunk_type   : str  — "text" | "image_description"
        figure_label : str  — figure label if chunk_type == "image_description"
        token_count  : int  — number of tokens in the chunk
        source       : str  — source filename (e.g. "IPCC_AR6_SYR_FullVolume.pdf")
    """

    def __init__(self, config: "Config") -> None:
        self.config = config
        db_path = str(config.chroma_db_path)
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._client = PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready at '{db_path}' — collection '{config.chroma_collection_name}' ({self._collection.count()} existing chunks)")

    # *** Write ******************************
    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 256,
        id_offset: int = 0,
    ) -> None:
        """
        Insert chunks and their embeddings into the collection.

        Args:
            chunks     : list of Chunk objects
            embeddings : np.ndarray of shape (N, dim), float32, normalised
            batch_size : insertion batch size (avoids OOM on large datasets)
            id_offset  : ID offset for resuming partial indexing
        """
        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]
            batch_embs = embeddings[start:end]

            ids = [f"chunk_{id_offset + start + i}" for i in range(len(batch_chunks))]
            documents = [c.text for c in batch_chunks]
            metadatas = [
                {
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "chunk_type": c.chunk_type,
                    "figure_label": c.figure_label,
                    "token_count": c.token_count,
                    "source": c.source,
                }
                for c in batch_chunks
            ]
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=batch_embs.tolist(),
            )
            logger.debug(f"Inserted batch [{start}:{end}]")

        logger.info(f"Added {total} chunks — total in collection: {self._collection.count()}")

    # *** Read / Retrieval ******************************
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 6,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """
        Find the closest chunks to a query embedding.

        Args:
            query_embedding : 1D normalised float32 vector
            n_results       : number of results to return
            chunk_type      : optional filter ("text" | "image_description")

        Returns:
            List of dicts {"text", "metadata", "distance", "score"}
            sorted by descending score (score = 1 − cosine distance).
        """
        where = {"chunk_type": {"$eq": chunk_type}} if chunk_type else None

        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1.0 - dist,
            })

        return hits

    # *** Collection management ******************************
    def reset(self) -> None:
        """Delete and recreate the collection (for full re-indexing)."""
        self._client.delete_collection(self.config.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self.config.chroma_collection_name}' has been reset.")

    def count(self) -> int:
        return self._collection.count()
