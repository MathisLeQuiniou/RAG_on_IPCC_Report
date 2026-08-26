"""
PDF loading and content extraction.

Extracts:
  - Text blocks (with page number metadata)
  - Images (with surrounding text context and detected figure label)

Dependency: pymupdf  (`pip install pymupdf`)
"""
import base64
import logging
import re
from pathlib import Path
import pymupdf

from ..models.document import TextBlock, ImageBlock

logger = logging.getLogger(__name__)


# *** Variables ******************************
# Images smaller than this (bytes) are ignored (icons, decorators, etc.)
_MIN_IMAGE_BYTES = 8_000

# Regex patterns for detecting figure labels in surrounding text
_FIGURE_PATTERNS = [
    r"(Figure\s+\d+[\.\-]\d*[a-zA-Z]?)",
    r"(Fig\.\s*\d+[\.\-]\d*[a-zA-Z]?)",
    r"(Box\s+\d+[\.\-]\d*)",
    r"(Panel\s+[A-Z])",
    r"(Table\s+\d+[\.\-]\d*)",
    r"(Infographic\s+\d+)",
]

# *** Loader Class ******************************
class PDFLoader:
    """
    Loads a PDF and exposes:
      - extract_text_blocks() -> list[TextBlock]
      - extract_images()      -> list[ImageBlock]
    """

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path
        self._doc: pymupdf.Document = pymupdf.open(str(pdf_path))
        self._n_pages = len(self._doc)
        logger.info(f"Opened PDF '{pdf_path.name}' ({self._n_pages} pages)")

    # *** Methods ******************************
    def extract_text_blocks(
        self,
        min_chars: int = 30,
        page_range: tuple[int, int] | None = None,
    ) -> list[TextBlock]:
        """
        Extract text blocks from all pages (or a given range).

        Args:
            min_chars:   Minimum text length to keep a block.
            page_range:  (start_page, end_page) in 1-based index, inclusive.

        Returns:
            List of TextBlock objects.
        """
        blocks: list[TextBlock] = []
        pages = list(self._page_iter(page_range))

        for page_num, page in pages:
            raw_blocks = page.get_text("blocks")
            for blk in raw_blocks:
                # blk = (x0, y0, x1, y1, text, block_no, block_type)
                # block_type 0 = text, 1 = image
                if blk[6] != 0:
                    continue
                text = blk[4].strip()
                # Clean up internal line breaks
                text = re.sub(r"-\n", "", text)       # hyphenated word wrap
                text = re.sub(r"\n", " ", text)
                text = re.sub(r"\s{2,}", " ", text)
                if len(text) >= min_chars:
                    blocks.append(TextBlock(text=text, page=page_num))

        logger.info(f"Extracted {len(blocks)} text blocks")
        return blocks

    def extract_images(
        self,
        page_range: tuple[int, int] | None = None,
    ) -> list[ImageBlock]:
        """
        Extract significant images from the PDF along with their metadata.

        For each image:
          - image_bytes / image_base64 : raw image data
          - surrounding_text           : text from the same page
          - figure_label               : detected label (Figure X.Y, Box Y, …)

        Returns:
            List of ImageBlock objects.
        """
        images: list[ImageBlock] = []
        pages = list(self._page_iter(page_range))

        for page_num, page in pages:
            page_text = page.get_text("text")
            img_list = page.get_images(full=True)

            for img_info in img_list:
                xref = img_info[0]
                try:
                    base_img = self._doc.extract_image(xref)
                except Exception as exc:
                    logger.debug(f"Could not extract image xref={xref} on page {page_num}: {exc}")
                    continue

                img_bytes = base_img["image"]
                if len(img_bytes) < _MIN_IMAGE_BYTES:
                    continue

                w = base_img.get("width", 0)
                h = base_img.get("height", 0)
                if w < 100 or h < 100:
                    continue  # too small to be a meaningful figure

                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                surrounding = self._surrounding_text(page_text, max_chars=400)
                label = self._detect_figure_label(page_text) or f"Image p.{page_num}"

                images.append(ImageBlock(
                    image_bytes=img_bytes,
                    image_base64=img_b64,
                    page=page_num,
                    surrounding_text=surrounding,
                    figure_label=label,
                    width=w,
                    height=h,
                ))

        logger.info(f"Extracted {len(images)} images")
        return images

    # *** Helpers ******************************
    def _page_iter(self, page_range: tuple[int, int] | None):
        """Iterate over pages (1-based index)."""
        start = (page_range[0] - 1) if page_range else 0
        end   = (page_range[1])     if page_range else self._n_pages
        for i in range(start, min(end, self._n_pages)):
            yield i + 1, self._doc[i]

    @staticmethod
    def _surrounding_text(page_text: str, max_chars: int = 400) -> str:
        """Return a representative excerpt of the page text."""
        words = page_text.split()
        return " ".join(words[:80])[:max_chars]

    @staticmethod
    def _detect_figure_label(text: str) -> str:
        """Search for a figure label in the text (e.g. 'Figure 2.3')."""
        for pattern in _FIGURE_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return ""

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> "PDFLoader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # *** Properties ******************************
    @property
    def n_pages(self) -> int:
        return self._n_pages
