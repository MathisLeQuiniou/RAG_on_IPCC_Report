"""
FastAPI application — entry point for the RAG API.

Usage:
    uvicorn backend.api.main:app --reload --port 8000

The pipeline (embedder + ChromaDB + Ollama) is initialised once at startup
via FastAPI's lifespan mechanism and shared across all requests.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.config import Config
from backend.pipeline import RAGPipeline
from backend.api.routes import query, chunks, document

logger = logging.getLogger(__name__)

# Pipeline instance shared accross requests
pipeline: RAGPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the pipeline on startup and release it on shutdown."""
    global pipeline
    logger.info("Initialising RAG pipeline...")
    pipeline = RAGPipeline(Config())
    logger.info("Pipeline ready — API is accepting requests.")
    yield
    pipeline = None

app = FastAPI(
    title="RAG on IPCC Report",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow requests from the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_pipeline() -> RAGPipeline:
    if pipeline is None:
        raise RuntimeError("Pipeline not initialised")
    return pipeline

# Inject the getter into routers before registering them
query.get_pipeline = get_pipeline
chunks.get_pipeline = get_pipeline

app.include_router(query.router, prefix="/api")
app.include_router(chunks.router, prefix="/api")
app.include_router(document.router, prefix="/api")

# health endpoint
@app.get("/api/health")
def health():
    return {"status": "ok"}
