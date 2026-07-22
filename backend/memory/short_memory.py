"""Konuşulmuş uyarılar için hafif, süreç içi kısa süreli bellek."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Any

from models.detection import Direction

if TYPE_CHECKING:
    from models.detection import Detection
    from models.scene import SceneObject


@dataclass(frozen=True)
class AlertMemoryKey:
    """Tekrarlama kontrolünde kullanılan yaklaşık konumlu uyarı kimliği."""

    object_label: str
    direction: Direction
    horizontal_position: int
    vertical_position: int


class ShortTermMemory:
    """Aynı nesne-konum uyarısının kısa sürede yeniden iletilmesini engeller."""

    def __init__(
        self,
        expiration_seconds: float = 5.0,
        position_bucket_pixels: int = 120,
    ) -> None:
        if expiration_seconds <= 0:
            raise ValueError("expiration_seconds sıfırdan büyük olmalıdır.")
        if position_bucket_pixels < 1:
            raise ValueError("position_bucket_pixels en az 1 olmalıdır.")

        self.expiration_seconds = expiration_seconds
        self.position_bucket_pixels = position_bucket_pixels
        self._spoken_alerts: dict[AlertMemoryKey, float] = {}
        self._lock = Lock()

    def should_speak(self, item: Detection | SceneObject | Any) -> bool:
        """Algılama veya nesne yeni ise kaydedip uyarının iletilebileceğini bildirir.

        Bu işlem tek bir kilit altında yapıldığı için eşzamanlı isteklerde aynı
        nesne-konum uyarısının birden fazla kez iletilmesine izin vermez.
        """
        alert_key = self._build_key(item)
        current_time = monotonic()

        with self._lock:
            self._expire_entries(current_time)
            if alert_key in self._spoken_alerts:
                return False

            self._spoken_alerts[alert_key] = current_time
            return True

    def remember_spoken(self, item: Detection | SceneObject | Any) -> None:
        """Bir algılamanın uyarı olarak iletildiğini kaydeder veya zamanını yeniler."""
        alert_key = self._build_key(item)
        current_time = monotonic()

        with self._lock:
            self._expire_entries(current_time)
            self._spoken_alerts[alert_key] = current_time

    def has_recent_alert(self, item: Detection | SceneObject | Any) -> bool:
        """Aynı nesne-konum uyarısının yakın zamanda iletilip iletilmediğini döndürür."""
        alert_key = self._build_key(item)
        current_time = monotonic()

        with self._lock:
            self._expire_entries(current_time)
            return alert_key in self._spoken_alerts

    def clear(self) -> None:
        """Bellekteki tüm geçici uyarı kayıtlarını siler."""
        with self._lock:
            self._spoken_alerts.clear()

    @property
    def size(self) -> int:
        """Süresi dolmamış kayıt sayısını döndürür."""
        current_time = monotonic()
        with self._lock:
            self._expire_entries(current_time)
            return len(self._spoken_alerts)

    def _expire_entries(self, current_time: float) -> None:
        """Süresi dolmuş kayıtları bellekten kaldırır."""
        expired_keys = [
            alert_key
            for alert_key, spoken_at in self._spoken_alerts.items()
            if current_time - spoken_at >= self.expiration_seconds
        ]
        for alert_key in expired_keys:
            del self._spoken_alerts[alert_key]

    def _build_key(self, item: Detection | SceneObject | Any) -> AlertMemoryKey:
        """Algılamayı veya SceneObject nesnesini yön ve konum bilgisiyle anahtara dönüştürür."""
        if hasattr(item, "label"):
            raw_label = getattr(item, "label")
        elif hasattr(item, "object"):
            raw_label = getattr(item, "object")
        else:
            raw_label = str(item)

        direction = getattr(item, "direction", Direction.CENTER)
        center_x = getattr(item, "center_x", 0)
        center_y = getattr(item, "center_y", 0)

        return AlertMemoryKey(
            object_label=" ".join(str(raw_label).casefold().split()),
            direction=direction,
            horizontal_position=center_x // self.position_bucket_pixels,
            vertical_position=center_y // self.position_bucket_pixels,
        )
