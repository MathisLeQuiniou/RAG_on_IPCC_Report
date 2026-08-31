"""
Chunk: a semantically coherent unit of text ready to be embedded and indexed.
"""
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    token_count: int
    page: int
    chunk_index: int
    chunk_type: str = "text"            # "text" | "image_description"
    figure_label: str = ""
    source: str = ""                    # source filename
