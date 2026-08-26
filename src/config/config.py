"""
Centralised configuration for the IPCC RAG project.
Edit this file to tune models, paths, and hyperparameters.
"""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # *** Config values ******************************
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    input_document: Path | None = None
    chroma_db_path: Path | None = None
    images_cache_path: Path | None = None

    # Embedding model (HuggingFace / sentence-transformers, runs locally)
    # Recommended options:
    #   "BAAI/bge-large-en-v1.5"                  (best quality, ~1.3 GB)
    #   "sentence-transformers/all-MiniLM-L6-v2"  (fast, ~90 MB)
    #   "intfloat/multilingual-e5-large"           (multilingual support)
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "mps"  # "cpu" | "cuda" | "mps" (Apple Silicon)

    # Ollama models (served locally via `ollama serve`)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    multimodal_model: str = "llava:13b"

    # llm parameters
    temperature: float = 0.1
    max_new_tokens: int = 1024

    # Chunking parameters
    max_tokens_per_chunk: int = 512
    chunk_overlap_tokens: int = 64
    min_chunk_tokens: int = 40
    # Cosine similarity threshold below which a new chunk is started.
    # Lower value → shorter, more thematically homogeneous chunks.
    semantic_similarity_threshold: float = 0.45

    # rag
    chroma_collection_name: str = "ipcc_rag"
    top_k: int = 6

    # *** Post-init: resolve relative paths ******************************
    def __post_init__(self) -> None:
        root = self.project_root
        if self.input_document is None:
            self.input_document = root / "input_document" / "IPCC_AR6_SYR_FullVolume.pdf"
        if self.chroma_db_path is None:
            self.chroma_db_path = root / "db" 
        if self.images_cache_path is None:
            self.images_cache_path = root / "images_cache"

    # *** Helper functions ******************************
    def display(self) -> str:
        lines = [
            f"  PDF              : {self.input_document}",
            f"  ChromaDB         : {self.chroma_db_path}",
            f"  Embedding model  : {self.embedding_model} ({self.embedding_device})",
            f"  LLM model        : {self.llm_model}",
            f"  Multimodal model : {self.multimodal_model}",
            f"  Max tokens/chunk : {self.max_tokens_per_chunk}",
            f"  Overlap tokens   : {self.chunk_overlap_tokens}",
            f"  Sem. threshold   : {self.semantic_similarity_threshold}",
            f"  Top-k retrieval  : {self.top_k}",
        ]
        return "\n".join(lines)
