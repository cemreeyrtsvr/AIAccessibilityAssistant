"""Anlamsal sahne değişikliği algılayıcı (SceneChangeDetector) bileşeni."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.scene import StructuredScene


class SceneChangeDetector:
    """Ardışık StructuredScene nesnelerini anlamsal nesne-yön-mesafe seviyesinde karşılaştırır."""

    def __init__(self, distance_bucket_meters: float = 1.0) -> None:
        self.distance_bucket_meters = distance_bucket_meters
        self._last_fingerprint: set[tuple[str, str, int | None]] | None = None

    def has_changed(self, scene: StructuredScene) -> bool:
        """Yeni sahnenin önceki sahneye göre anlamsal olarak değişip değişmediğini belirler."""
        if scene is None:
            return False

        current_fingerprint = self._build_fingerprint(scene)

        if self._last_fingerprint is None:
            self._last_fingerprint = current_fingerprint
            return True

        if current_fingerprint != self._last_fingerprint:
            self._last_fingerprint = current_fingerprint
            return True

        return False

    def reset(self) -> None:
        """Değişiklik algılayıcı geçmişini sıfırlar."""
        self._last_fingerprint = None

    def _build_fingerprint(
        self, scene: StructuredScene
    ) -> set[tuple[str, str, int | None]]:
        """Sahnedeki nesneleri (etiket, yön, mesafe kova indeksi) parmak izi kümesine çevirir."""
        fingerprint: set[tuple[str, str, int | None]] = set()

        for obj in scene.objects:
            normalized_label = " ".join(obj.label.casefold().split())
            dir_str = (
                obj.direction.value
                if hasattr(obj.direction, "value")
                else str(obj.direction)
            ).casefold()

            distance_bucket: int | None = None
            if obj.distance is not None:
                distance_bucket = round(obj.distance / self.distance_bucket_meters)

            fingerprint.add((normalized_label, dir_str, distance_bucket))

        return fingerprint
