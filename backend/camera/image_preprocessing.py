"""
Image preprocessing utilities.

These functions improve image quality before OCR
or computer vision processing.
"""

import cv2
from typing import Any


def grayscale(frame: Any):
    """
    Convert image to grayscale.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def gaussian_blur(frame: Any, kernel_size: int = 5):
    """
    Apply Gaussian Blur.
    """
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def threshold(frame: Any):
    """
    Apply binary threshold.
    """

    gray = grayscale(frame)

    _, thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY,
    )

    return thresh


def adaptive_threshold(frame: Any):
    """
    Apply adaptive threshold.
    """

    gray = grayscale(frame)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def resize(frame: Any, scale: float = 2.0):
    """
    Resize image.
    """

    return cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )


def denoise(frame: Any):
    """
    Remove image noise.
    """

    return cv2.fastNlMeansDenoising(frame)


def equalize_histogram(frame: Any):
    """
    Improve image contrast.
    """

    gray = grayscale(frame)

    return cv2.equalizeHist(gray)