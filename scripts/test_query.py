"""
Script that allows to test the RAGPipeline against a single query
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow importing src/ from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import RAGPipeline, Config
from backend.utils import setup_logging

logger = logging.getLogger(__name__)

def main() -> None:
    # *** RAGPipeline initialization ******************************
    setup_logging(logging.INFO)
    pipeline = RAGPipeline(Config())

    # *** Query processing ******************************
    user_query = "What are the key risks of climate change?"
    logger.info(f"       user query: {user_query}")
    for token in pipeline.query(
        "What is the structure of the document ?",
        top_k=6,
        stream=True,
        verbose=True
    ):
        print(token, end="", flush=True)
    print()  # newline after stream ends
    
if __name__ == "__main__":
    main()
