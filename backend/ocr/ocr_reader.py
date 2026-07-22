"""
OCR Reader Module.
"""

from typing import Optional

import cv2
import pytesseract

from camera.image_preprocessing import grayscale
from config.settings import (
    OCR_LANGUAGE,
    TESSERACT_PATH,
)
from ocr.text_cleaner import clean_text


class OCRReader:
    """
    Reads text from images using Tesseract OCR.
    """

    def __init__(self) -> None:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    def read_text(
        self,
        frame: cv2.typing.MatLike,
    ) -> str:
        """
        Extract text from an image.

        Args:
            frame: OpenCV image.

        Returns:
            Cleaned OCR text.
        """

        processed = grayscale(frame)

        try:

            text = pytesseract.image_to_string(
                processed,
                lang=OCR_LANGUAGE,
                config="--oem 3 --psm 6",
            )

        except Exception:

            return ""

        return clean_text(text)