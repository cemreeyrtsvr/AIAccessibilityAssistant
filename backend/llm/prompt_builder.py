"""
Prompt Builder Module.

Builds prompts for the Local LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)

if TYPE_CHECKING:
    from models.scene import StructuredScene


class PromptBuilder:
    """
    Builds prompts for VisionVoice AI.
    """

    @staticmethod
    def build(
        question: str = "",
        scene: StructuredScene | None = None,
        ocr_text: str | None = None,
        objects: list[str] | None = None,
        text: str = "",
    ) -> list[dict[str, str]]:
        """
        Build prompts for the Local LLM.

        Args:
            question: User question string.
            scene: Optional StructuredScene instance.
            ocr_text: OCR detected text.
            objects: Legacy list of detected object strings.
            text: Legacy OCR detected text string.

        Returns:
            OpenAI-compatible messages list.
        """
        formatted_objects: list[str] = []

        if scene is not None and scene.objects:
            for obj in scene.objects:
                dir_str = (
                    obj.direction.value
                    if hasattr(obj.direction, "value")
                    else str(obj.direction)
                )
                if obj.distance is not None:
                    formatted_objects.append(f"{dir_str} {obj.label} ({obj.distance:.1f}m)")
                else:
                    formatted_objects.append(f"{dir_str} {obj.label}")
        elif objects:
            formatted_objects = list(objects)

        if not formatted_objects:
            formatted_objects = ["None"]

        final_ocr = (ocr_text if ocr_text is not None else text).strip()
        if not final_ocr:
            final_ocr = "None"

        user_prompt = USER_PROMPT.format(
            objects="\n".join(formatted_objects),
            text=final_ocr,
        )

        if question and question.strip():
            user_prompt += f"\n\nUser Question:\n{question.strip()}"

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]