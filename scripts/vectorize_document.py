"""
Document vectorisation script for the IPCC RAG system.

Steps performed:
  1. Load the PDF (text blocks + images)
  2. Describe images via LLaVA (local multimodal model)
  3. Semantic token-based chunking
  4. Embed all chunks
  5. Insert into ChromaDB

Usage:
    # Full vectorisation
    python scripts/vectorize_document.py

    # Re-index from scratch (drops the existing collection)
    python scripts/vectorize_document.py --reset

    # Quick test on the first 20 pages, no images
    python scripts/vectorize_document.py --pages 20 --skip-images

    # Custom PDF path
    python scripts/vectorize_document.py --pdf /path/to/file.pdf
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow importing src/ from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils import setup_logging, get_elapsed_time
from backend.config.config import Config
from backend.ingestion.loader import PDFLoader
from backend.ingestion.image_describer import ImageDescriber
from backend.ingestion.chunker import SemanticTokenChunker
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

def main() -> None:
    # *** Arguments Parsing ******************************
    parser = argparse.ArgumentParser(
        description="Vectorise the IPCC PDF and store chunks in ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing ChromaDB collection before indexing.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip figure description step (faster, for testing).",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the PDF to index (overrides the config default).",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N pages (useful for quick tests).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Embedding batch size (default: 64).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    # *** Configuration ******************************
    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    config = Config()
    if args.pdf:
        config.input_document = Path(args.pdf).resolve()

    logger.info("=" * 56)
    logger.info("IPCC RAG — Document Vectorisation")
    logger.info("=" * 56)
    for line in config.display().splitlines():
        logger.info(line)

    if not config.input_document.exists():
        logger.error(f"PDF not found: {config.input_document}")
        sys.exit(1)

    total_start = time.time()
    logger.info("=" * 56)
    
    # *** Component initialisation ******************************
    logger.info("[Init] Loading embedder...")
    embedder = Embedder(config)

    logger.info("[Init] Connecting to ChromaDB...")
    store = VectorStore(config)

    if args.reset:
        logger.info("[Init] Resetting collection...")
        store.reset()

    # *** Step 1 - Load PDF ******************************
    logger.info(f"[1/4] Loading PDF: {config.input_document.name}")
    t0 = time.time()

    page_range = (1, args.pages) if args.pages else None

    with PDFLoader(config.input_document) as loader:
        logger.info(f"       {loader.n_pages} pages total")
        text_blocks = loader.extract_text_blocks(page_range=page_range)
        image_blocks = (
            [] if args.skip_images
            else loader.extract_images(page_range=page_range)
        )

    logger.info(
        f"[1/4] Done — {len(text_blocks)} text blocks, {len(image_blocks)} "\
        f"images [{get_elapsed_time(t0)}]"
    )
    
    # *** Step 2 — Describe images (LLaVA) ******************************
    image_pairs: list[tuple] = []

    if image_blocks and not args.skip_images:
        logger.info(
            f"[2/4] Describing {len(image_blocks)} figures "\
            f"via {config.multimodal_model}...",
        )
        t0 = time.time()
        describer = ImageDescriber(config)
        descriptions = describer.describe_batch(image_blocks)
        image_pairs = list(zip(image_blocks, descriptions))
        n_ok = sum(1 for _, d in image_pairs if d.success)
        logger.info(
            f"[2/4] Done — {n_ok}/{len(image_pairs)} descriptions succeeded  "\
            f"[{get_elapsed_time(t0)}]",
        )
    else:
        logger.info("[2/4] Image description skipped (--skip-images).")

    # *** Step 3 — Semantic chunking ******************************
    logger.info(
        f"[3/4] Chunking text blocks (max {config.max_tokens_per_chunk} "\
        f"tokens/chunk)..."
    )
    t0 = time.time()

    chunker = SemanticTokenChunker(config, embedder.sentence_model)
    all_chunks = []
    source_name = config.input_document.name

    for i, block in enumerate(text_blocks):
        if (i + 1) % 200 == 0:
            logger.debug(f"  Chunking block {i+1}/{len(text_blocks)}...")
        chunks = chunker.chunk_text_block(
            text=block.text,
            page=block.page,
            source=source_name,
        )
        all_chunks.extend(chunks)

    # Add image description chunks
    n_img_chunks = 0
    for img_block, img_desc in image_pairs:
        img_chunk = chunker.chunk_image_description(
            description=img_desc.description,
            page=img_block.page,
            figure_label=img_block.figure_label,
            source=source_name,
        )
        all_chunks.append(img_chunk)
        n_img_chunks += 1

    n_text_chunks = len(all_chunks) - n_img_chunks
    logger.info(
        f"[3/4] Done — {len(all_chunks)} chunks total ({n_text_chunks} text, "\
        f"{n_img_chunks,} image)  [{get_elapsed_time(t0)}]",
    )

    if not all_chunks:
        logger.error("No chunks produced. Check the PDF and parameters.")
        sys.exit(1)
    
    # *** Step 4 — Embed + insert into ChromaDB ******************************
    logger.info(f"[4/4] Embedding {len(all_chunks)} chunks...")
    t0 = time.time()

    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed(texts, batch_size=args.batch_size, show_progress=True)
    logger.info(f"       Embedding done  [{get_elapsed_time(t0)}]")

    logger.info("       Inserting into ChromaDB...")
    t0 = time.time()
    store.add_chunks(all_chunks, embeddings)
    logger.info(f"       Insertion done  [{get_elapsed_time(t0)}]")

    # *** Summary ******************************
    logger.info("=" * 56)
    logger.info(f"Vectorisation complete  [total: {get_elapsed_time(total_start)}]")
    logger.info(f"  Chunks in collection : {store.count()}")
    logger.info(f"  ChromaDB path        : {config.chroma_db_path}")
    logger.info("=" * 56)
    logger.info("To query the system:")
    logger.info("  from backend import RAGPipeline, Config")
    logger.info("  pipeline = RAGPipeline(Config())")
    logger.info('  print(pipeline.query("What are the key risks of climate change?"))')
    
if __name__ == "__main__":
    main()
