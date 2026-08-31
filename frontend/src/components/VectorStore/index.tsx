/**
 * VectorStore — loads all chunks on mount and lets the user browse them
 * alongside the corresponding PDF page.
 */
import { useEffect, useState } from "react"
import { fetchAllChunks } from "../../api"
import type { ChunkItem } from "../../types"
import ChunkList from "./ChunkList"
import PdfViewer from "./PdfViewer"

export default function VectorStore() {
  const [chunks, setChunks]     = useState<ChunkItem[]>([])
  const [selected, setSelected] = useState<ChunkItem | null>(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState<string | null>(null)

  useEffect(() => {
    fetchAllChunks()
      .then((res) => setChunks(res.chunks))
      .catch((e) => setError(e instanceof Error ? e.message : "Unknown error"))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <p className="status-message">Loading chunks from vector store…</p>
  if (error)   return <p className="error-message">{error}</p>

  return (
    <div className="tab-layout">
      <div className="tab-layout__left">
        <ChunkList chunks={chunks} selected={selected} onSelect={setSelected} />
      </div>
      <div className="tab-layout__right">
        <PdfViewer selected={selected} />
      </div>
    </div>
  )
}
