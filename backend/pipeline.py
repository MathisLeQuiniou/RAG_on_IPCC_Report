"""
End-to-end RAG pipeline: query → retrieval → generation.

This is the main entry point for querying the system in production.

Minimal usage:
    from backend import RAGPipeline, Config

    pipeline = RAGPipeline(Config())
    answer = pipeline.query("What are the main drivers of climate change?")
    print(answer)

Streaming usage:
    for token in pipeline.query("...", stream=True):
        print(token, end="", flush=True)
"""
import logging
from typing import Iterator

from .config.config import Config
from .retrieval.embedder import Embedder
from .retrieval.vector_store import VectorStore
from .retrieval.retriever import Retriever
from .generation.llm import LLM
from .models.retrieval import RetrievalHit

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Assembles all RAG components:
      Embedder → VectorStore → Retriever → LLM

    The pipeline is stateful: components (embedding model, ChromaDB
    connection, Ollama client) are instantiated once and reused across
    all queries.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        logger.info(f"Initialising RAG pipeline\n{config.display()}")

        logger.info("[1/3] Loading embedder...")
        self._embedder = Embedder(config)

        logger.info("[2/3] Connecting to ChromaDB...")
        self._store = VectorStore(config)

        logger.info("[3/3] Initialising LLM (Ollama)...")
        self._llm = LLM(config)

        self._retriever = Retriever(config, self._embedder, self._store)
        logger.info("Pipeline ready.")

    # *** Methods ******************************
    def query(
        self,
        question: str,
        top_k: int | None = None,
        include_images: bool = True,
        stream: bool = False,
        verbose: bool = False,
    ) -> str | Iterator[str]:
        """
        Ask a question to the RAG system.

        Args:
            question       : natural-language question
            top_k          : number of chunks to retrieve (default: config.top_k)
            include_images : include figure descriptions in the context
            stream         : if True, returns an Iterator[str] of tokens
            verbose        : log retrieved chunks at INFO level

        Returns:
            Answer str, or Iterator[str] if stream=True.
        """
        # 1. Retrieval
        context, hits = self._retriever.retrieve_and_format(
            query=question,
            top_k=top_k,
            include_images=include_images,
        )

        if verbose:
            self._log_hits(hits)

        # 2. Prompt assembly
        prompt = self._llm.build_rag_prompt(context=context, question=question)

        # 3. Generation
        return self._llm.generate(prompt, stream=stream)


    def query_with_hits(
        self,
        question: str,
        top_k: int | None = None,
        include_images: bool = True,
    ) -> tuple[str, list[RetrievalHit]]:
        """
        Like query(), but also returns the retrieved hits alongside the answer.
        Intended for API consumers that need both the answer and the source chunks.

        Args:
            question       : natural-language question
            top_k          : number of chunks to retrieve (default: config.top_k)
            include_images : include figure descriptions in the context

        Returns:
            Tuple of (answer string, list of RetrievalHit).
        """
        # 1. Retrieval
        context, hits = self._retriever.retrieve_and_format(
            query=question,
            top_k=top_k,
            include_images=include_images,
        )

        # 2. Prompt assembly
        prompt = self._llm.build_rag_prompt(context=context, question=question)

        # 3. Generation
        answer = self._llm.generate(prompt, stream=False)

        return answer, hits

    # *** Properties ******************************
    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    # *** Helpers ******************************
    @staticmethod
    def _log_hits(hits: list[RetrievalHit]) -> None:
        logger.info(f"Retrieved {len(hits)} chunks:")
        for i, h in enumerate(hits, 1):
            meta = h.metadata
            snippet = h.text[:80].replace("\n", " ")
            logger.info(f"  {i}. [score={h.score:.3f}] page={meta['page']} type={meta['chunk_type']} | {snippet}...")
