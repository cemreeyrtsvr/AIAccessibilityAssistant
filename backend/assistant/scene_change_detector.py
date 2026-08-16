"""Anlamsal sahne değişikliği algılayıcı — gürültülü kareler için streak doğrulama,
yön değişiklikleri için anında geçiş."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.scene import StructuredScene


# Takip edilen nesne başına (etiket, yön) çifti
_Fingerprint = frozenset[tuple[str, str, int | None]]


class SceneChangeDetector:
    """Ardışık StructuredScene nesnelerini karşılaştırır.

    Kararlılık politikası
    ─────────────────────
    • Sahne ilk kez görülüyorsa: anında geç.
    • **Mevcut** etiketlerin yön değişikliği: anında geç (düşük gecikme zorunluluğu).
    • Yeni etiket eklenmesi / kaldırılması: ``stability_frames`` ardışık kare onayla.
    • Tek gürültülü kare: yoksay.
    """

    def __init__(
        self,
        distance_bucket_meters: float = 1.0,
        stability_frames: int = 2,
    ) -> None:
        self.distance_bucket_meters = distance_bucket_meters
        self.stability_frames = stability_frames

        self._confirmed_fp: _Fingerprint | None = None
        self._candidate_fp: _Fingerprint | None = None
        self._candidate_streak: int = 0

    # ─── Genel API ─────────────────────────────────────────────────────────

    def has_changed(self, scene: StructuredScene) -> bool:
        """Sahne değişikliğini düşük gecikmeli yön algılama ile bildirir."""
        if scene is None:
            return False

        current_fp = self._build_fingerprint(scene)

        # İlk kare — her zaman değişmiş say
        if self._confirmed_fp is None:
            self._confirmed_fp = current_fp
            self._candidate_fp = current_fp
            self._candidate_streak = 0
            return True

        # Onaylı sahneyle aynı — değişiklik yok
        if current_fp == self._confirmed_fp:
            self._candidate_fp = current_fp
            self._candidate_streak = 0
            return False

        # ── Yön değişikliği mi yoksa etiket değişikliği mi? ─────────────────
        # Etiket kümesi aynı kalıyor ama yön farklılaşıyorsa → anında onayla.
        confirmed_labels = self._label_set(self._confirmed_fp)
        current_labels = self._label_set(current_fp)

        if current_labels == confirmed_labels:
            # Sadece yön/mesafe farklı — anında onayla
            self._confirmed_fp = current_fp
            self._candidate_fp = current_fp
            self._candidate_streak = 0
            return True

        # ── Etiket değişikliği — streak ile onayla ──────────────────────────
        if current_fp == self._candidate_fp:
            self._candidate_streak += 1
        else:
            self._candidate_fp = current_fp
            self._candidate_streak = 1

        if self._candidate_streak >= self.stability_frames:
            self._confirmed_fp = current_fp
            self._candidate_streak = 0
            return True

        return False

    def reset(self) -> None:
        """Değişiklik algılayıcı geçmişini sıfırlar."""
        self._confirmed_fp = None
        self._candidate_fp = None
        self._candidate_streak = 0

    # ─── Yardımcılar ───────────────────────────────────────────────────────

    @staticmethod
    def _label_set(fp: _Fingerprint) -> frozenset[str]:
        """Parmak izindeki etiketleri bir küme olarak döndürür."""
        return frozenset(label for label, *_ in fp)

    def _build_fingerprint(self, scene: StructuredScene) -> _Fingerprint:
        """Sahnedeki nesneleri (etiket, yön, mesafe kovası) parmak izi olarak döndürür."""
        entries: list[tuple[str, str, int | None]] = []

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

            entries.append((normalized_label, dir_str, distance_bucket))

        return frozenset(entries)
