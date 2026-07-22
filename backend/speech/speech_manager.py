"""Sesli etkileşimler için merkezi yönetici servisi."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from speech.sentence_builder import SentenceBuilder
from speech.stt import SpeechRecognizer
from speech.tts import TextToSpeechService

if TYPE_CHECKING:
    from models.response import Alert, Alerts
    from models.scene import SceneObject, StructuredScene


class SpeechManager:
    """TTS, STT ve cümle oluşturucu modüllerini birleştiren merkezi ses yöneticisi."""

    def __init__(
        self,
        sentence_builder: SentenceBuilder | None = None,
        tts_service: TextToSpeechService | None = None,
        speech_recognizer: SpeechRecognizer | None = None,
    ) -> None:
        self.sentence_builder = sentence_builder or SentenceBuilder()
        self.tts_service = tts_service or TextToSpeechService()
        self.speech_recognizer = speech_recognizer or SpeechRecognizer()

    def speak_scene(
        self,
        items: Sequence[SceneObject] | StructuredScene | list[Alert] | Alerts | Any,
    ) -> str | None:
        """Sahne veya uyarı nesnelerini cümleleştirip seslendirir ve metni döndürür."""
        if items is None:
            return None

        speech_text = self.sentence_builder.build_text(items)
        if speech_text and speech_text.strip():
            self.tts_service.speak(speech_text)
            return speech_text
        return None

    def speak_alerts(self, alerts: Alerts | list[Alert] | Any) -> None:
        """Alerts / Alert nesnelerini Türkçe cümleye dönüştürüp seslendirir."""
        self.speak_scene(alerts)

    def speak(self, text: str) -> None:
        """Düz Türkçe metni seslendirir."""
        if text:
            self.tts_service.speak(text)

    def listen(self, duration_seconds: float = 3.0) -> str:
        """Mikrofondan ses dinler ve tanınan Türkçe metni döndürür."""
        return self.speech_recognizer.listen_and_recognize(
            duration_seconds=duration_seconds
        )

    def stop(self) -> None:
        """Ses motorlarını durdurur."""
        self.tts_service.stop()
