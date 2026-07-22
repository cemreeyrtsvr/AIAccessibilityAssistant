"""Çevrimdışı Türkçe Metinden Sese (TTS) servisi."""

from __future__ import annotations

import queue
import threading

import pyttsx3

from config.settings import VOICE_RATE, VOICE_VOLUME


class TextToSpeechService:
    """pyttsx3 tabanlı çevrimdışı Metinden Sese (TTS) dönüştürme servisi."""

    def __init__(
        self,
        rate: int = VOICE_RATE,
        volume: float = VOICE_VOLUME,
        language: str = "tr-TR",
    ) -> None:
        self.rate = rate
        self.volume = volume
        self.language = language
        self._speech_queue: queue.Queue[str | None] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._is_running = False
        self._start_worker()

    def _start_worker(self) -> None:
        """Sesi arka planda engellemesiz yürütmek için iş parçacığı başlatır."""
        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._process_speech_queue,
            daemon=True,
        )
        self._worker_thread.start()

    def _process_speech_queue(self) -> None:
        """Kuyruktaki metinleri pyttsx3 motoruyla sırayla seslendirir."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self._select_turkish_voice(engine, self.language)
        except Exception:
            engine = None

        while self._is_running:
            try:
                text = self._speech_queue.get(timeout=0.5)
                if text is None:
                    break
                if engine is not None and text.strip():
                    engine.say(text)
                    engine.runAndWait()
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                continue

    def speak(self, text: str) -> None:
        """Düz metni sıraya ekleyerek çevrimdışı seslendirir."""
        if not text or not isinstance(text, str):
            return
        self._speech_queue.put(text)

    def stop(self) -> None:
        """Seslendirme iş parçacığını durdurur."""
        self._is_running = False
        self._speech_queue.put(None)

    @staticmethod
    def _select_turkish_voice(engine: pyttsx3.Engine, language: str) -> None:
        """Sistemde mevcut Türkçe sesi arar ve seçer."""
        try:
            voices = engine.getProperty("voices")
            target_langs = {"tr", "tr-tr", "tr_tr", "turkish", "tur"}
            for voice in voices:
                voice_name = getattr(voice, "name", "").casefold()
                voice_id = getattr(voice, "id", "").casefold()
                voice_langs = [
                    str(lang).casefold()
                    for lang in getattr(voice, "languages", [])
                ]

                if (
                    any(lang in voice_name or lang in voice_id for lang in target_langs)
                    or any(lang in voice_langs for lang in target_langs)
                ):
                    engine.setProperty("voice", voice.id)
                    return
        except Exception:
            pass
