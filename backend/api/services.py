"""Business logic for VisionVoice AI."""

import numpy as np

from assistant.assistant_service import AssistantService
from models.response import Alerts


class AIService:
    """Main AI pipeline."""

    def __init__(self, assistant_service: AssistantService | None = None) -> None:
        self.assistant = assistant_service or AssistantService()

    def analyze(self, frame: np.ndarray) -> Alerts:
        """Analyze an image frame through the live backend pipeline.

        Args:
            frame: OpenCV BGR image.

        Returns:
            Structured Alerts object.
        """
        return self.assistant.process_frame(frame)