"""
Prompt Builder Module.

Builds prompts for the Local LLM.
"""

from llm.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)


class PromptBuilder:
    """
    Builds prompts for VisionVoice AI.
    """

    @staticmethod
    def build(
        objects: list[str] | None = None,
        text: str = "",
    ) -> list[dict[str, str]]:
        """
        Build prompts for the Local LLM.

        Args:
            objects: List of detected objects.
            text: OCR detected text.

        Returns:
            OpenAI-compatible messages list.
        """

        if not objects:
            objects = ["None"]

        if not text.strip():
            text = "None"

        user_prompt = USER_PROMPT.format(
            objects="\n".join(objects),
            text=text,
        )

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