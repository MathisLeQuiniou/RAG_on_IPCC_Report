"""
Pydantic schemas for the RAG API.

All request and response models are defined here and imported by the routes.
This keeps the API contract in one place, separate from the internal domain
models (plain dataclasses) in backend/models/.
"""
from pydantic import BaseModel

# *** /api/query ******************************
class QueryRequest(BaseModel):
    question: str
    top_k: int = 6
    include_images: bool = True

class ChunkResult(BaseModel):
    """A single retrieved chunk, as returned by the query endpoint."""
    text: str
    page: int
    chunk_index: int
    chunk_type: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkResult]

# *** /api/chunks ******************************
class ChunkItem(BaseModel):
    """A single chunk as stored in the vector store, as returned by the chunks endpoint."""
    id: str
    text: str
    page: int
    chunk_index: int
    chunk_type: str
    figure_label: str
    token_count: int
    source: str

class ChunksResponse(BaseModel):
    total: int
    chunks: list[ChunkItem]
