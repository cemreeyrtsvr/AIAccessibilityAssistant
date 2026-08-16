"""Çevrimdışı/Yerel Microsoft Edge Neural TTS (edge-tts) servisi.

Yüksek kaliteli Türkçe sinirsel sesler (tr-TR-EmelNeural, tr-TR-AhmetNeural) ile
kuyruk tabanlı, kesintisiz ve geçici dosya bırakmayan seslendirme sağlar.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import queue
import tempfile
import threading
from typing import Optional

import edge_tts

from config.settings import VOICE_RATE, VOICE_VOLUME


class TextToSpeechService:
    """Microsoft Edge Neural TTS motoruyla yüksek kaliteli seslendirme servisi."""

    PRIMARY_VOICE = "tr-TR-EmelNeural"
    FALLBACK_VOICE = "tr-TR-AhmetNeural"

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
        self._worker_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._start_worker()

    def _start_worker(self) -> None:
        """Seslendirmeyi ana iş parçacığını engellemeden yürütmek için arka plan iş parçacığı başlatır."""
        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._run_async_worker,
            daemon=True,
        )
        self._worker_thread.start()

    def _run_async_worker(self) -> None:
        """Arka plan iş parçacığında asyncio olay döngüsü çalıştırır."""
        asyncio.run(self._speech_loop())

    async def _speech_loop(self) -> None:
        """Kuyruktaki metinleri edge-tts ile dönüştürüp sırayla seslendirir."""
        loop = asyncio.get_running_loop()

        while self._is_running:
            try:
                # Kuyruktan veri alma işlemini engellemesiz çalıştır
                text = await loop.run_in_executor(
                    None, self._get_next_queue_item
                )
                if text is None:
                    break
                if text.strip():
                    await self._generate_and_play(text.strip())
                self._speech_queue.task_done()
            except Exception:
                continue

    def _get_next_queue_item(self) -> str | None:
        """Kuyruktan bir sonraki metni güvenli şekilde alır."""
        try:
            return self._speech_queue.get(timeout=0.5)
        except queue.Empty:
            return ""

    async def _generate_and_play(self, text: str) -> None:
        """Metni edge-tts ile MP3 dosyasına yazıp çalar ve ardından dosyayı siler."""
        temp_file_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                temp_file_path = tmp.name

            # Birincil ses (EmelNeural) dene, hata verirse ikincil ses (AhmetNeural) kullan
            selected_voice = self.PRIMARY_VOICE
            try:
                communicate = edge_tts.Communicate(text, selected_voice)
                await communicate.save(temp_file_path)
            except Exception:
                selected_voice = self.FALLBACK_VOICE
                communicate = edge_tts.Communicate(text, selected_voice)
                await communicate.save(temp_file_path)

            # Windows MCI API ile sesi oynat
            await asyncio.to_thread(self._play_mp3_mci, temp_file_path)

        except Exception as e:
            print(f"[WARNING] Edge TTS generation/playback error: {e}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

    @staticmethod
    def _play_mp3_mci(file_path: str) -> None:
        """Windows MCI (Media Control Interface) yerel API'si ile MP3 dosyasını çalar."""
        try:
            winmm = ctypes.windll.winmm
            alias = "edge_tts_playback"

            # Varsa önceki oturumu kapat
            winmm.mciSendStringW(f"close {alias}", None, 0, 0)

            open_cmd = f'open "{file_path}" type mpegvideo alias {alias}'
            res_open = winmm.mciSendStringW(open_cmd, None, 0, 0)

            if res_open == 0:
                winmm.mciSendStringW(f"play {alias} wait", None, 0, 0)
                winmm.mciSendStringW(f"close {alias}", None, 0, 0)
        except Exception as err:
            print(f"[WARNING] MCI Audio Playback error: {err}")

    def speak(self, text: str) -> None:
        """Düz metni sıraya ekleyerek çevrimdışı/yerel ve sıralı şekilde seslendirir."""
        if not text or not isinstance(text, str):
            return
        self._speech_queue.put(text)

    def stop(self) -> None:
        """Seslendirme iş parçacığını ve kuyruğu güvenli şekilde durdurur."""
        self._is_running = False
        self._speech_queue.put(None)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)