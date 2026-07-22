"""API yanıtları ve sistem içi mesajlaşma modelleri."""

from __future__ import annotations

from pydantic import BaseModel, Field

from assistant.intent_classifier import UserIntent
from decision.notification_engine import Alert, Alerts
from models.scene import StructuredScene


class AskModeResult(BaseModel):
    """Soru-Cevap (Ask Mode) modunun orkestrasyon yanıt modeli."""

    intent: UserIntent
    success: bool = True
    scene: StructuredScene | None = None
    ocr_text: str | None = None
    answer: str | None = None


class LiveModeResult(BaseModel):
    """Canlı Mod (Live Mode) işleme hattı sonuç modeli."""

    spoken: bool
    scene: StructuredScene
    spoken_text: str | None = None


__all__ = ["Alert", "Alerts", "AskModeResult", "LiveModeResult"]
