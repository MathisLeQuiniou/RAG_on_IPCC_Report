"""
Semantic token-based text chunking.

Algorithm:
  1. Split the text into sentences (regex, handles common abbreviations).
  2. Encode all sentences in a single batch pass.
  3. Compute cosine similarity between consecutive sentence embeddings.
  4. Start a new chunk when:
       - similarity drops below the threshold (topic boundary)
         AND the current chunk has reached min_chunk_tokens,
       OR
       - adding the next sentence would exceed max_tokens_per_chunk.
  5. Apply token overlap: the last N tokens of the previous chunk are
     prepended to the next one to preserve continuity.

Image descriptions are wrapped as a single "image_description" chunk
without further splitting (their length is already controlled by the LLM).

Dependencies: sentence-transformers, transformers (for the tokenizer)
"""
import logging
import re
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..models.chunk import Chunk

if TYPE_CHECKING:
    from ..config.config import Config

logger = logging.getLogger(__name__)

# *** Variables ******************************
# # Common abbreviations found in scientific reports (prevent false splits)
_ABBREVS = (
    r"e\.g|i\.e|vs|Fig|Dr|Mr|Mrs|Prof|et al|approx|ca|No|Vol|"
    r"Sec|Eq|Ref|al|pp|cf|viz|resp|est|max|min|avg"
)
_ABBREV_RE = re.compile(rf"\b({_ABBREVS})\.", re.IGNORECASE)

# *** SemanticTokenChunker Class ******************************
class SemanticTokenChunker:
    """
    Segments a text block into semantically coherent chunks
    while respecting a token budget.
    """

    def __init__(self, config: "Config", embedding_model: SentenceTransformer) -> None:
        self.config = config
        self._model = embedding_model
        # Use the same tokenizer as the embedding model for accurate token counts
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.embedding_model
        )
        logger.debug(f"SemanticTokenChunker initialised (model={config.embedding_model})")

    # *** Methods ******************************
    def chunk_text_block(
        self,
        text: str,
        page: int,
        source: str = "",
    ) -> list[Chunk]:
        """
        Split a raw text block into semantic chunks.

        Returns:
            List of Chunk objects (may be empty if the text is too short).
        """
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        # Encode all sentences in one batch (more efficient than one-by-one)
        embeddings: np.ndarray = self._model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        chunks: list[Chunk] = []
        current_sents: list[str] = []
        current_tokens: int = 0
        chunk_idx: int = 0

        for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
            sent_tokens = self._count_tokens(sent)

            should_split = False
            if current_sents:
                # Token budget exceeded
                if current_tokens + sent_tokens > self.config.max_tokens_per_chunk:
                    should_split = True
                # Semantic boundary (only if current chunk is large enough)
                elif current_tokens >= self.config.min_chunk_tokens:
                    sim = self._cosine_sim(embeddings[i - 1], emb)
                    if sim < self.config.semantic_similarity_threshold:
                        should_split = True

            if should_split:
                chunk_text = " ".join(current_sents)
                if self._count_tokens(chunk_text) >= self.config.min_chunk_tokens:
                    chunks.append(Chunk(
                        text=chunk_text,
                        token_count=self._count_tokens(chunk_text),
                        page=page,
                        chunk_index=chunk_idx,
                        source=source,
                    ))
                    chunk_idx += 1

                # Overlap: prepend the last N tokens of the previous chunk
                overlap = self._build_overlap(current_sents)
                current_sents = overlap
                current_tokens = sum(self._count_tokens(s) for s in overlap)

            current_sents.append(sent)
            current_tokens += sent_tokens

        # Flush remaining sentences
        if current_sents:
            chunk_text = " ".join(current_sents)
            if self._count_tokens(chunk_text) >= self.config.min_chunk_tokens:
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=self._count_tokens(chunk_text),
                    page=page,
                    chunk_index=chunk_idx,
                    source=source,
                ))

        return chunks

    def chunk_image_description(
        self,
        description: str,
        page: int,
        figure_label: str,
        source: str = "",
    ) -> Chunk:
        """
        Wrap an image description as a single chunk of type 'image_description'.
        No further splitting is applied.
        """
        return Chunk(
            text=description,
            token_count=self._count_tokens(description),
            page=page,
            chunk_index=0,
            chunk_type="image_description",
            figure_label=figure_label,
            source=source,
        )

    # *** Private helpers ******************************
    @staticmethod
    def split_into_sentences(text: str) -> list[str]:
        """
        Split text into sentences.
        Handles abbreviations and internal line breaks.
        """
        # Protect abbreviation dots
        text = _ABBREV_RE.sub(lambda m: m.group(1) + "\x00", text)
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Restore protected dots
        sentences = [s.replace("\x00", ".").strip() for s in sentences]
        return [s for s in sentences if s]

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two already-normalised vectors."""
        return float(np.dot(a, b))

    def _build_overlap(self, sentences: list[str]) -> list[str]:
        """
        Build the overlap by taking the last sentences of the chunk
        within the chunk_overlap_tokens budget.
        """
        budget = self.config.chunk_overlap_tokens
        overlap: list[str] = []
        used = 0
        for s in reversed(sentences):
            t = self._count_tokens(s)
            if used + t > budget:
                break
            overlap.insert(0, s)
            used += t
        return overlap
