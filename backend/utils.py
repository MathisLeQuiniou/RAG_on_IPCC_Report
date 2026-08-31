"""
Utils script for helper functions used in the project
"""
import logging
import time

def setup_logging(level: int = logging.INFO) -> None:
    """Configure a minimal, human-readable logger for the script."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers
    for noisy in ("chromadb", "sentence_transformers", "transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_elapsed_time(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"