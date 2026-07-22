"""Zaman tabanlı kare sınırlayıcı (FrameThrottle) bileşeni."""

from __future__ import annotations

from time import monotonic


class FrameThrottle:
    """Belirli bir hedef FPS oranına göre karelerin işlenip işlenmeyeceğini belirleyen sınırlayıcı."""

    def __init__(self, target_fps: float = 10.0) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps sıfırdan büyük olmalıdır.")
        self.target_fps = target_fps
        self.min_interval = 1.0 / target_fps
        self._last_processed_time: float | None = None

    def should_process(self) -> bool:
        """Geçerli karenin işlenmesi gerekip gerekmediğini zaman kontrolüyle belirler.

        İlk kare her zaman işlenir.
        """
        current_time = monotonic()

        if self._last_processed_time is None:
            self._last_processed_time = current_time
            return True

        elapsed = current_time - self._last_processed_time
        if elapsed >= self.min_interval:
            self._last_processed_time = current_time
            return True

        return False

    def reset(self) -> None:
        """Zamanlayıcıyı sıfırlar, bir sonraki kare işlenir."""
        self._last_processed_time = None
