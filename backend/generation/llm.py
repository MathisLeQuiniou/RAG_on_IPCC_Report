"""
LLM interface via Ollama (local inference).

Supports:
  - Standard generation (returns a str)
  - Token-by-token streaming (returns an Iterator[str])

Dependency: ollama  (`pip install ollama`)
"""
import logging
from typing import Iterator, TYPE_CHECKING

import ollama

if TYPE_CHECKING:
    from ..config.config import Config

logger = logging.getLogger(__name__)


# *** Prompts ******************************

RAG_SYSTEM_PROMPT = """\
You are an expert assistant on climate science, specializing in the IPCC AR6 \
Synthesis Report. Answer questions accurately and concisely based solely on the \
context provided below. If the context does not contain sufficient information \
to answer the question, say so explicitly — do not invent facts. \
When referencing specific information, cite the page number(s) in parentheses, \
e.g. (p. 42).\
"""

RAG_PROMPT_TEMPLATE = """\
Context extracted from the IPCC AR6 Synthesis Report:

{context}

---

Question: {question}

Answer (based only on the context above):\
"""


# *** Main LLM class ******************************

class LLM:
    """
    Ollama wrapper for text generation.

    Usage:
        llm = LLM(config)
        answer = llm.generate(prompt)              # returns str
        for token in llm.generate(prompt, stream=True):
            print(token, end="", flush=True)       # streaming
    """

    def __init__(self, config: "Config") -> None:
        self.config = config
        self._client = ollama.Client(host=config.ollama_base_url)
        logger.info(f"LLM client initialised (model={config.llm_model}, url={config.ollama_base_url})")

    def generate(
        self,
        prompt: str,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Generate a response for the given prompt.

        Args:
            prompt : complete prompt (context + question already formatted)
            stream : if True, returns an Iterator[str] of tokens

        Returns:
            Full str response if stream=False, Iterator[str] otherwise.
        """
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        options = {
            "temperature": self.config.temperature,
            "num_predict": self.config.max_new_tokens,
        }

        if stream:
            return self._stream(messages, options)

        logger.debug(f"Sending request to Ollama ({self.config.llm_model})")
        response = self._client.chat(
            model=self.config.llm_model,
            messages=messages,
            options=options,
        )
        return response.message.content

    def _stream(self, messages: list[dict], options: dict) -> Iterator[str]:
        """Token streaming generator."""
        response = self._client.chat(
            model=self.config.llm_model,
            messages=messages,
            options=options,
            stream=True,
        )
        for chunk in response:
            content = chunk.message.content
            if content:
                yield content

    def build_rag_prompt(self, context: str, question: str) -> str:
        """Assemble the standard RAG prompt from context and question."""
        return RAG_PROMPT_TEMPLATE.format(context=context, question=question)
