"""
POST /api/query
Runs the RAG pipeline and returns the generated answer along with the retrieved chunks.
"""
from fastapi import APIRouter, HTTPException

from backend.api.schemas import QueryRequest, QueryResponse, ChunkResult

router = APIRouter()

# Injected by main.py at startup
get_pipeline = None


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Run the full RAG pipeline for a user question.

    Args:
        request : question, optional top_k and include_images flag

    Returns:
        Generated answer and the list of chunks used as context.
    """
    if get_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not available")

    pipeline = get_pipeline()
    answer, hits = pipeline.query_with_hits(
        question=request.question,
        top_k=request.top_k,
        include_images=request.include_images,
    )

    chunks = [
        ChunkResult(
            text=h.text,
            page=h.metadata.get("page", 0),
            chunk_index=h.metadata.get("chunk_index", 0),
            chunk_type=h.metadata.get("chunk_type", "text"),
            score=h.score,
        )
        for h in hits
    ]

    return QueryResponse(answer=answer, chunks=chunks)
