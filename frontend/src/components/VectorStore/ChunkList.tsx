/**
 * ChunkList — scrollable list of all chunks indexed in the vector store.
 * Clicking a chunk selects it and updates the PDF viewer.
 */
import type { ChunkItem } from "../../types"

interface Props {
  chunks: ChunkItem[]
  selected: ChunkItem | null
  onSelect: (chunk: ChunkItem) => void
}

export default function ChunkList({ chunks, selected, onSelect }: Props) {
  return (
    <section className="panel chunk-list-panel">
      <h3>Vector store — {chunks.length} chunks</h3>
      <ul className="chunk-list">
        {chunks.map((chunk) => (
          <li
            key={chunk.id}
            className={`chunk-item ${selected?.id === chunk.id ? "chunk-item--active" : ""}`}
            onClick={() => onSelect(chunk)}
          >
            <span className="chunk-meta">
              Page {chunk.page}
              {chunk.chunk_type === "image_description" && (
                <span className="chunk-badge">image</span>
              )}
              · {chunk.token_count} tokens
            </span>
            <p className="chunk-text">{chunk.text.slice(0, 180)}…</p>
          </li>
        ))}
      </ul>
    </section>
  )
}
