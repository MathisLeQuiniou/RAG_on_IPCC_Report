"""
IPCC RAG — public surface.

Typical usage:
    from src import RAGPipeline, Config

    pipeline = RAGPipeline(Config())
    answer = pipeline.query("What are the projected sea level rises by 2100?")
"""
from .config.config import Config
from .pipeline import RAGPipeline
from .models import TextBlock, ImageBlock, Chunk, ImageDescription, RetrievalHit

__all__ = [
    "Config",
    "RAGPipeline",
    "TextBlock",
    "ImageBlock",
    "Chunk",
    "ImageDescription",
    "RetrievalHit",
]
