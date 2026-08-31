/**
 * API client — thin wrappers around the FastAPI backend endpoints.
 * Base URL can be overridden via the VITE_API_BASE environment variable.
 */
import type { QueryResponse, ChunksResponse } from "./types"

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000"

export async function queryRag(
  question: string,
  topK = 6,
  includeImages = true,
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, include_images: includeImages }),
  })
  if (!res.ok) throw new Error(`Query failed: ${res.status} ${res.statusText}`)
  return res.json()
}

export async function fetchAllChunks(): Promise<ChunksResponse> {
  const res = await fetch(`${API_BASE}/api/chunks`)
  if (!res.ok) throw new Error(`Chunks fetch failed: ${res.status} ${res.statusText}`)
  return res.json()
}

/** Returns the URL of a PDF page rendered as a PNG image (1-indexed). */
export function documentPageUrl(pageNum: number): string {
  return `${API_BASE}/api/document/page/${pageNum}`
}
