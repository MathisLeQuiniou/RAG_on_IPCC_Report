"""
ImageDescription: the result of a multimodal LLM describing a PDF figure.
"""
from dataclasses import dataclass

@dataclass
class ImageDescription:
    figure_label: str
    page: int
    description: str
    success: bool = True
    error: str = ""
