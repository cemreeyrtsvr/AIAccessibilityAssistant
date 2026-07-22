"""Mikrofon ses yakalama araçları."""

from __future__ import annotations


class MicrophoneStream:
    """Çevrimdışı ses tanıma için mikrofon veri akışı yöneticisi."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 4000) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._pyaudio = None
        self._stream = None

    def start(self) -> None:
        """Mikrofon akışını başlatır."""
        try:
            import pyaudio

            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            self._stream.start_stream()
        except Exception:
            self.stop()

    def read_chunk(self) -> bytes:
        """Mikrofondan tek bir ses parçası okur."""
        if self._stream is None:
            return b""
        try:
            return self._stream.read(self.chunk_size, exception_on_overflow=False)
        except Exception:
            return b""

    def stop(self) -> None:
        """Mikrofon akışını ve kaynaklarını kapatır."""
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None
