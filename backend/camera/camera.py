"""
Camera module.

Handles webcam initialization, frame capture and resource cleanup.
"""

import cv2
from typing import Optional

from config.settings import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
)


class Camera:
    """Camera controller."""

    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open the camera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

    def capture_frame(self) -> Optional[cv2.typing.MatLike]:
        """
        Capture a single frame from the camera.

        Returns:
            Captured frame if successful, otherwise None.
        """

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def is_opened(self) -> bool:
        """
        Check whether the camera is available.
        """

        return self.cap.isOpened()

    def release(self) -> None:
        """
        Release camera resources.
        """

        if self.cap.isOpened():
            self.cap.release()