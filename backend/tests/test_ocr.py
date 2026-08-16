"""
OCR integration tests.
"""

from camera.camera import Camera
from ocr.ocr_reader import OCRReader


def test_ocr_reader_runs() -> None:
    """
    Verify that OCR can process a captured frame without crashing.
    """

    camera = Camera()
    ocr = OCRReader()

    try:
        assert camera.is_opened()

        frame = camera.capture_frame()

        assert frame is not None

        text = ocr.read_text(frame)

        # OCR text may legitimately be empty depending on the scene.
        assert text is None or isinstance(text, str)

    finally:
        camera.release()