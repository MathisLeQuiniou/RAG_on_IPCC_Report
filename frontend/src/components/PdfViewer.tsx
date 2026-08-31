/**
 * PdfViewer — shared component for displaying a PDF page with the chunk text
 * highlighted directly on the document image, and the raw chunk text shown
 * below for reference.
 * Used by both the AskYourDoc and VectorStore tabs.
 *
 * Props:
 *   page          — page number to display (null triggers the empty state)
 *   text          — chunk text; passed to the backend as a highlight query
 *                   param so PyMuPDF draws a yellow highlight on the image,
 *                   and displayed as-is below the image
 *   figureLabel   — optional label shown next to the page number (VectorStore)
 *   siblingPages  — optional sorted list of all pages that contain a chunk,
 *                   shown in the header when more than one (AskYourDoc)
 *   emptyMessage  — placeholder text shown when page is null
 *   highlightLabel— label above the text block below the image
 */
import { documentPageUrl } from "../api"

interface Props {
  page: number | null
  text?: string
  figureLabel?: string
  siblingPages?: number[]
  emptyMessage?: string
  highlightLabel?: string
}

export default function PdfViewer({
  page,
  text,
  figureLabel,
  siblingPages,
  emptyMessage = "Select a chunk to see its page in the document.",
  highlightLabel = "Extracted passage",
}: Props) {
  if (page === null) {
    return (
      <div className="pdf-viewer pdf-viewer--empty">
        <p>{emptyMessage}</p>
      </div>
    )
  }

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer__header">
        Page {page}
        {figureLabel && (
          <span className="pdf-viewer__pages"> · {figureLabel}</span>
        )}
        {siblingPages && siblingPages.length > 1 && (
          <span className="pdf-viewer__pages">
            {" "}· chunks on pages {siblingPages.join(", ")}
          </span>
        )}
      </div>

      <div className="pdf-viewer__image-wrap">
        <img
          src={documentPageUrl(page, text)}
          alt={`Page ${page}`}
          className="pdf-viewer__image"
        />
      </div>

      {text && (
        <div className="pdf-viewer__highlight">
          <span className="highlight-label">{highlightLabel}</span>
          <p className="highlight-text">{text}</p>
        </div>
      )}
    </div>
  )
}
