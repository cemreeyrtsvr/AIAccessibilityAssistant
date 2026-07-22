"""
Text cleaning utilities.

Provides helper functions for cleaning raw OCR output.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean raw OCR text.

    Args:
        text: Raw text returned by OCR.

    Returns:
        Cleaned text.
    """

    if not text:
        return ""

    # Windows ve Unix satır sonlarını normalize et
    text = text.replace("\r", "\n")

    # Birden fazla boşluğu teke indir
    text = re.sub(r"[ \t]+", " ", text)

    # Birden fazla boş satırı teke indir
    text = re.sub(r"\n{2,}", "\n", text)

    # Baştaki ve sondaki boşlukları temizle
    text = text.strip()

    return text