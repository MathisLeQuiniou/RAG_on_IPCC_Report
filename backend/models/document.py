"""
Raw content extracted from a PDF document.

TextBlock  : a continuous text block from a page.
ImageBlock : an image with its surrounding context and detected figure label.
"""
from dataclasses import dataclass

@dataclass
class TextBlock:
    """A continuous text block extracted from a PDF page."""
    text: str
    page: int
    block_type: str = "text"

@dataclass
class ImageBlock:
    """An image extracted from the PDF, along with its textual context."""
    image_bytes: bytes
    image_base64: str           # Base64-encoded for the Ollama API
    page: int
    surrounding_text: str       # ~400 characters of nearby text
    figure_label: str = ""      # e.g. "Figure 1.1", "Box 2.3"
    block_type: str = "image"
    width: int = 0
    height: int = 0
