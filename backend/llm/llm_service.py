"""LLM yanıt üretimi ve istem hazırlama servisi."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm.client import LLMClient
from llm.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from models.scene import StructuredScene


class LLMService:
    """Yapılandırılmış sahne ve OCR metninden LLM yanıtı üreten servis."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_builder: type[PromptBuilder] = PromptBuilder,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.prompt_builder = prompt_builder

    def generate_answer(
        self,
        question: str,
        scene: StructuredScene | None = None,
        ocr_text: str | None = None,
    ) -> str:
        """Soru, sahne ve metin bağlamını PromptBuilder'a devredip yanıt üretir."""
        try:
            messages = self.prompt_builder.build(
                question=question,
                scene=scene,
                ocr_text=ocr_text,
            )
            return self.llm_client.generate(messages)
        except Exception:
            return ""
