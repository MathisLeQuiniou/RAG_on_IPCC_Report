/**
 * PdfViewer (AskYourDoc) — displays a PDF page as an image and highlights
 * the selected chunk text below it.
 *
 * Since pages are served as PNG images by the backend, in-image text
 * highlighting is not possible without coordinates. Instead the selected
 * chunk text is shown with a yellow highlight below the page image,
 * giving the document context alongside the extracted content.
 */
import { documentPageUrl } from "../../api"
import type { ChunkResult } from "../../types"

interface Props {
  chunks: ChunkResult[]
  selected: ChunkResult | null
}

export default function PdfViewer({ chunks, selected }: Props) {
  if (chunks.length === 0) return (
    <div className="pdf-viewer pdf-viewer--empty">
      <p>The document will appear here after your first query.</p>
    </div>
  )

  const active = selected ?? chunks[0]

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer__header">
        Page {active.page}
        {chunks.length > 1 && (
          <span className="pdf-viewer__pages">
            {" "}· chunks on pages {[...new Set(chunks.map(c => c.page))].sort((a,b)=>a-b).join(", ")}
          </span>
        )}
      </div>

      <div className="pdf-viewer__image-wrap">
        <img
          src={documentPageUrl(active.page)}
          alt={`Page ${active.page}`}
          className="pdf-viewer__image"
        />
      </div>

      <div className="pdf-viewer__highlight">
        <span className="highlight-label">Extracted passage</span>
        <p className="highlight-text">{active.text}</p>
      </div>
    </div>
  )
}
