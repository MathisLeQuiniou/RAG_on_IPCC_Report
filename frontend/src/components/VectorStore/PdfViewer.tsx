/**
 * PdfViewer (VectorStore) — displays the PDF page corresponding to the
 * selected chunk, with the chunk text highlighted below the page image.
 */
import { documentPageUrl } from "../../api"
import type { ChunkItem } from "../../types"

interface Props {
  selected: ChunkItem | null
}

export default function PdfViewer({ selected }: Props) {
  if (!selected) return (
    <div className="pdf-viewer pdf-viewer--empty">
      <p>Select a chunk on the left to see its page in the document.</p>
    </div>
  )

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer__header">
        Page {selected.page}
        {selected.figure_label && (
          <span className="pdf-viewer__pages"> · {selected.figure_label}</span>
        )}
      </div>

      <div className="pdf-viewer__image-wrap">
        <img
          src={documentPageUrl(selected.page)}
          alt={`Page ${selected.page}`}
          className="pdf-viewer__image"
        />
      </div>

      <div className="pdf-viewer__highlight">
        <span className="highlight-label">Chunk text</span>
        <p className="highlight-text">{selected.text}</p>
      </div>
    </div>
  )
}
