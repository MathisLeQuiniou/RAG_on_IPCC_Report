/**
 * ChunksPanel — list of chunks retrieved by the RAG pipeline.
 * Clicking a chunk selects it and updates the PDF viewer.
 */
import type { ChunkResult } from "../../types"

interface Props {
  chunks: ChunkResult[]
  selected: ChunkResult | null
  onSelect: (chunk: ChunkResult) => void
}

export default function ChunksPanel({ chunks, selected, onSelect }: Props) {
  if (chunks.length === 0) return null
  return (
    <section className="panel chunks-panel">
      <h3>Retrieved chunks ({chunks.length})</h3>
      <ul className="chunk-list">
        {chunks.map((chunk, i) => (
          <li
            key={i}
            className={`chunk-item ${selected === chunk ? "chunk-item--active" : ""}`}
            onClick={() => onSelect(chunk)}
          >
            <span className="chunk-meta">
              Page {chunk.page} · {chunk.chunk_type} · score {chunk.score.toFixed(3)}
            </span>
            <p className="chunk-text">{chunk.text.slice(0, 200)}…</p>
          </li>
        ))}
      </ul>
    </section>
  )
}
