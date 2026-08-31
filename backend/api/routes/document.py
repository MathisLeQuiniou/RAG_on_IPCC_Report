"""
GET /api/document/page/{page_num}
Returns a single PDF page rendered as a PNG image.
Used by the frontend PDF viewer to display and highlight retrieved chunks.

Dependency: pymupdf (pip install pymupdf)
"""
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import fitz  # pymupdf

logger = logging.getLogger(__name__)

router = APIRouter()

_PDF_PATH = Path(__file__).parent.parent.parent.parent / "input_document" / "IPCC_AR6_SYR_FullVolume.pdf"

# Cache the open document to avoid re-opening on every request
_pdf_doc: fitz.Document | None = None


def _get_doc() -> fitz.Document:
    global _pdf_doc
    if _pdf_doc is None:
        if not _PDF_PATH.exists():
            raise HTTPException(status_code=404, detail=f"PDF not found: {_PDF_PATH}")
        _pdf_doc = fitz.open(str(_PDF_PATH))
        logger.info(f"Opened PDF '{_PDF_PATH.name}' ({_pdf_doc.page_count} pages)")
    return _pdf_doc


@router.get("/document/page/{page_num}")
def get_page(page_num: int, scale: float = 2.0):
    """
    Render page `page_num` (1-indexed) as a PNG image.

    Args:
        page_num : page number, 1-indexed to match chunk metadata
        scale    : rendering resolution factor (2.0 = high-res, readable)

    Returns:
        PNG image bytes.
    """
    doc = _get_doc()

    # Convert 1-indexed to 0-indexed
    idx = page_num - 1
    if idx < 0 or idx >= doc.page_count:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_num} out of range (document has {doc.page_count} pages)",
        )

    page = doc[idx]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")

    return Response(content=png_bytes, media_type="image/png")


@router.get("/document/info")
def get_info():
    """Return basic document metadata (page count and filename)."""
    doc = _get_doc()
    return {
        "page_count": doc.page_count,
        "filename": _PDF_PATH.name,
    }
