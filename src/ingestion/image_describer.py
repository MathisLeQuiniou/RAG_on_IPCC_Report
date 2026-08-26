"""
Automatic description of PDF figures using a local multimodal LLM
(LLaVA via Ollama).

The generated text is treated as a regular chunk: it is embedded and
stored in ChromaDB so the retriever can surface figures via semantic search.
"""
import logging
from typing import TYPE_CHECKING

import ollama

from ..models.document import ImageBlock
from ..models.image import ImageDescription

if TYPE_CHECKING:
    from ..config.config import Config

logger = logging.getLogger(__name__)


# *** Prompts ******************************
_SYSTEM_PROMPT = (
    "You are an expert scientific assistant specializing in climate science "
    "and IPCC reports. When shown a figure, describe it thoroughly and precisely: "
    "what it represents, the axes or dimensions involved, key trends or data ranges, "
    "labels, color coding, and the main scientific conclusions that can be drawn. "
    "Focus on information useful for answering questions about climate change. "
    "Be factual and concise."
)

_USER_PROMPT_TEMPLATE = (
    "This is {label} from the IPCC AR6 Synthesis Report (page {page}).\n"
    "Context from the surrounding text: {context}\n\n"
    "Describe this figure in detail."
)

# *** ImageDescriber Class ******************************
class ImageDescriber:
    """
    Sends each image to a multimodal Ollama model (e.g. LLaVA)
    and returns a textual description.

    Usage:
        describer = ImageDescriber(config)
        results = describer.describe_batch(image_blocks)
        for r in results:
            print(r.description)
    """

    def __init__(self, config: "Config") -> None:
        self.config = config
        self._client = ollama.Client(host=config.ollama_base_url)

    # *** Methods ******************************
    def describe(self, block: ImageBlock) -> ImageDescription:
        """
        Describe a single image. Never raises — returns a fallback on error.
        """
        label = block.figure_label or f"Image on page {block.page}"
        context = block.surrounding_text

        user_content = _USER_PROMPT_TEMPLATE.format(
            label=label,
            page=block.page,
            context=context,
        )

        try:
            response = self._client.chat(
                model=self.config.multimodal_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_content,
                        "images": [block.image_base64],
                    },
                ],
            )
            text = response.message.content.strip()
            logger.debug(f"Described '{label}' (page {block.page}): {len(text)} chars")
            return ImageDescription(figure_label=label, page=block.page, description=text)

        except Exception as exc:
            logger.warning(f"Failed to describe '{label}' (page {block.page}): {exc}")
            fallback = f"[{label}, page {block.page} — description unavailable: {exc}]"
            return ImageDescription(
                figure_label=label,
                page=block.page,
                description=fallback,
                success=False,
                error=str(exc),
            )

    def describe_batch(
        self,
        blocks: list[ImageBlock],
    ) -> list[ImageDescription]:
        """
        Describe a list of images sequentially.
        (LLaVA via Ollama does not yet support native batching.)
        """
        results: list[ImageDescription] = []
        total = len(blocks)

        for i, block in enumerate(blocks, 1):
            label = block.figure_label or f"image {i}"
            logger.info(f"Describing image {i}/{total}: {label} (page {block.page}, {block.width}x{block.height} px)")
            result = self.describe(block)
            results.append(result)

        n_ok = sum(1 for r in results if r.success)
        logger.info(f"Image description complete: {n_ok}/{total} succeeded")
        return results
