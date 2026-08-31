"""
GET /api/chunks
Returns all chunks indexed in the vector store, sorted by page and chunk_index.
"""
from fastapi import APIRouter, HTTPException

from backend.api.schemas import ChunkItem, ChunksResponse

router = APIRouter()

# Injected by main.py at startup
get_pipeline = None


@router.get("/chunks", response_model=ChunksResponse)
def get_chunks():
    """
    Retrieve every chunk stored in the vector store.

    Returns:
        Total chunk count and the full list, ordered by page then chunk_index.
    """
    if get_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not available")

    pipeline = get_pipeline()
    raw = pipeline.vector_store.get_all()

    chunks = [
        ChunkItem(
            id=item["id"],
            text=item["text"],
            page=item["metadata"].get("page", 0),
            chunk_index=item["metadata"].get("chunk_index", 0),
            chunk_type=item["metadata"].get("chunk_type", "text"),
            figure_label=item["metadata"].get("figure_label", ""),
            token_count=item["metadata"].get("token_count", 0),
            source=item["metadata"].get("source", ""),
        )
        for item in raw
    ]

    return ChunksResponse(total=len(chunks), chunks=chunks)
