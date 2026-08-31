/**
 * Shared TypeScript types matching the Pydantic schemas defined in backend/api/schemas.py.
 */

// POST /api/query
export interface ChunkResult {
  text: string
  page: number
  chunk_index: number
  chunk_type: string
  score: number
}

export interface QueryResponse {
  answer: string
  chunks: ChunkResult[]
}

// GET /api/chunks
export interface ChunkItem {
  id: string
  text: string
  page: number
  chunk_index: number
  chunk_type: string
  figure_label: string
  token_count: number
  source: string
}

export interface ChunksResponse {
  total: number
  chunks: ChunkItem[]
}
