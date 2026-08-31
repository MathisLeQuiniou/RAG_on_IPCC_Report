/**
 * AskYourDoc — manages state for the query tab.
 * Orchestrates QueryBar → API call → AnswerPanel + ChunksPanel + PdfViewer.
 */
import { useState } from "react"
import { queryRag } from "../../api"
import type { ChunkResult } from "../../types"
import QueryBar from "./QueryBar"
import AnswerPanel from "./AnswerPanel"
import ChunksPanel from "./ChunksPanel"
import PdfViewer from "../PdfViewer"

export default function AskYourDoc() {
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const [answer, setAnswer]   = useState("")
  const [chunks, setChunks]   = useState<ChunkResult[]>([])
  const [selected, setSelected] = useState<ChunkResult | null>(null)

  async function handleQuery(question: string) {
    setLoading(true)
    setError(null)
    setAnswer("")
    setChunks([])
    setSelected(null)
    try {
      const res = await queryRag(question)
      setAnswer(res.answer)
      setChunks(res.chunks)
      setSelected(res.chunks[0] ?? null)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  const active = selected ?? chunks[0] ?? null
  const siblingPages = [...new Set(chunks.map(c => c.page))].sort((a, b) => a - b)

  return (
    <div className="tab-layout">
      <div className="tab-layout__left">
        <QueryBar onSubmit={handleQuery} loading={loading} />
        {error && <p className="error-message">{error}</p>}
        <AnswerPanel answer={answer} />
        <ChunksPanel chunks={chunks} selected={selected} onSelect={setSelected} />
      </div>
      <div className="tab-layout__right">
        <PdfViewer
          page={active?.page ?? null}
          text={active?.text}
          siblingPages={siblingPages}
          emptyMessage="The document will appear here after your first query."
          highlightLabel="Extracted passage"
        />
      </div>
    </div>
  )
}
