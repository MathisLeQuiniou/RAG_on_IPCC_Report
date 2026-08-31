"""
GET /api/document/page/{page_num}
Returns a single PDF page rendered as a PNG image.
An optional `highlight` query parameter can be passed to have PyMuPDF
search for that text on the page and draw a yellow highlight over it
before rendering.

Dependency: pymupdf (pip install pymupdf)
"""
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import pymupdf

logger = logging.getLogger(__name__)

router = APIRouter()

_PDF_PATH = Path(__file__).parent.parent.parent.parent / "input_document" / "IPCC_AR6_SYR_FullVolume.pdf"

# Cache the open document to avoid re-opening on every request
_pdf_doc: pymupdf.Document | None = None


def _get_doc() -> pymupdf.Document:
    global _pdf_doc
    if _pdf_doc is None:
        if not _PDF_PATH.exists():
            raise HTTPException(status_code=404, detail=f"PDF not found: {_PDF_PATH}")
        _pdf_doc = pymupdf.open(str(_PDF_PATH))
        logger.info(f"Opened PDF '{_PDF_PATH.name}' ({_pdf_doc.page_count} pages)")
    return _pdf_doc


@router.get("/document/page/{page_num}")
def get_page(page_num: int, scale: float = 2.0, highlight: str | None = None):
    """
    Render page `page_num` (1-indexed) as a PNG image.

    Args:
        page_num  : page number, 1-indexed to match chunk metadata
        scale     : rendering resolution factor (2.0 = high-res, readable)
        highlight : optional text to search for and highlight in yellow on
                    the page. A temporary copy of the page is used so the
                    cached document is never modified.

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

    mat = pymupdf.Matrix(scale, scale)

    if highlight:
        # Work on a temporary single-page copy so the cached document is
        # never modified by annotation calls.
        tmp = pymupdf.open()
        tmp.insert_pdf(doc, from_page=idx, to_page=idx)
        page = tmp[0]

        # Normalise the search string: collapse newlines, limit length
        search_text = " ".join(highlight.split())
        rects = page.search_for(search_text)

        if rects:
            annot = page.add_highlight_annot(rects)
            # Bright yellow — (R, G, B) in [0, 1]
            annot.set_colors(stroke=(1.0, 0.9, 0.0))
            annot.update()

        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        tmp.close()
    else:
        page = doc[idx]
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
