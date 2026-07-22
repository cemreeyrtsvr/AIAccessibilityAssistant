"""
Local LLM Client.

Communicates with Azure AI Foundry Local.
"""

from openai import OpenAI

from config.settings import (
    FOUNDRY_API_KEY,
    FOUNDRY_BASE_URL,
    MAX_RESPONSE_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
)


class LLMClient:
    """
    Client for communicating with the local Foundry model.
    """

    def __init__(self) -> None:

        self.client = OpenAI(
            base_url=FOUNDRY_BASE_URL,
            api_key=FOUNDRY_API_KEY,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """
        Generate a response from the Local LLM.
        """

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS,
        )

        return response.choices[0].message.content.strip()