"""Çevrimdışı Türkçe Konuşmayı Metne Dönüştürme (STT) servisi."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import ClassVar

import vosk

from speech.microphone import MicrophoneStream


class SpeechRecognizer:
    """Vosk tabanlı çevrimdışı Türkçe ses tanıma servisi."""

    _models: ClassVar[dict[str, vosk.Model]] = {}
    _model_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        model_path: str | Path = "models/vosk-model-small-tr",
        sample_rate: int = 16000,
    ) -> None:
        self.model_path = str(model_path)
        self.sample_rate = sample_rate
        self._model: vosk.Model | None = None

    @classmethod
    def _get_model(cls, model_path: str) -> vosk.Model | None:
        """Vosk modelini bir kez yükler ve önbelleğe alır."""
        with cls._model_lock:
            if model_path not in cls._models:
                if not Path(model_path).exists():
                    return None
                try:
                    cls._models[model_path] = vosk.Model(model_path)
                except Exception:
                    return None
            return cls._models[model_path]

    def _ensure_model(self) -> vosk.Model | None:
        """Model yüklenmemişse getirir."""
        if self._model is None:
            self._model = self._get_model(self.model_path)
        return self._model

    def recognize_audio_bytes(self, audio_data: bytes) -> str:
        """PCM ses verisini çözümleyerek tanınan metni döndürür."""
        model = self._ensure_model()
        if model is None or not audio_data:
            return ""

        try:
            rec = vosk.KaldiRecognizer(model, self.sample_rate)
            rec.AcceptWaveform(audio_data)
            result_json = rec.FinalResult()
            result_dict = json.loads(result_json)
            return str(result_dict.get("text", "")).strip()
        except Exception:
            return ""

    def listen_and_recognize(self, duration_seconds: float = 3.0) -> str:
        """Mikrofondan ses dinler ve tanınan Türkçe metni döndürür."""
        model = self._ensure_model()
        if model is None:
            return ""

        stream = MicrophoneStream(sample_rate=self.sample_rate)
        try:
            stream.start()
            rec = vosk.KaldiRecognizer(model, self.sample_rate)
            chunks_to_read = int((self.sample_rate / stream.chunk_size) * duration_seconds)

            for _ in range(max(1, chunks_to_read)):
                chunk = stream.read_chunk()
                if not chunk:
                    break
                rec.AcceptWaveform(chunk)

            result_json = rec.FinalResult()
            result_dict = json.loads(result_json)
            return str(result_dict.get("text", "")).strip()
        except Exception:
            return ""
        finally:
            stream.stop()
