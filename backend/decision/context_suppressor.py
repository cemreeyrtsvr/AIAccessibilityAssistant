"""Bağlama göre gereksiz ve yinelenen erişilebilirlik uyarılarını azaltır."""

from __future__ import annotations

from pydantic import BaseModel, Field

from decision.priority_engine import PrioritizedDetection
from decision.rules import AlertSeverity, normalize_label


class ContextSuppressionConfig(BaseModel):
    """Bağlamsal uyarı baskılamasının değiştirilebilir kurallarını tanımlar."""

    overlap_iou_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    safe_labels: set[str] = Field(
        default_factory=lambda: {
            "bench",
            "chair",
            "potted plant",
            "suitcase",
            "table",
        }
    )
    dangerous_labels: set[str] = Field(default_factory=lambda: {"person"})


class ContextAwareSuppressor:
    """Aynı kategorideki ve çakışan güvenli nesnelerde tek uyarı bırakır."""

    def __init__(self, config: ContextSuppressionConfig | None = None) -> None:
        self.config = config or ContextSuppressionConfig()
        self._safe_labels = {normalize_label(label) for label in self.config.safe_labels}
        self._dangerous_labels = {
            normalize_label(label) for label in self.config.dangerous_labels
        }

    def suppress(
        self,
        prioritized_detections: list[PrioritizedDetection],
    ) -> list[PrioritizedDetection]:
        """En anlamlı algılamaları öncelik sırasını koruyarak döndürür."""
        unique_categories = self._keep_highest_priority_per_category(
            prioritized_detections
        )
        return [
            item
            for item in unique_categories
            if not self._is_safe_object_overlapped_by_dangerous(item, unique_categories)
        ]

    @staticmethod
    def _keep_highest_priority_per_category(
        prioritized_detections: list[PrioritizedDetection],
    ) -> list[PrioritizedDetection]:
        """Aynı nesne kategorisinin yalnızca en yüksek öncelikli örneğini korur."""
        selected: list[PrioritizedDetection] = []
        seen_categories: set[str] = set()

        for item in sorted(
            prioritized_detections,
            key=lambda prioritized: prioritized.priority,
            reverse=True,
        ):
            category = normalize_label(item.detection.label)
            if category not in seen_categories:
                selected.append(item)
                seen_categories.add(category)

        return selected

    def _is_safe_object_overlapped_by_dangerous(
        self,
        candidate: PrioritizedDetection,
        candidates: list[PrioritizedDetection],
    ) -> bool:
        """Güvenli bir nesne tehlikeli nesneyle çakışıyorsa bastırılacağını belirler."""
        if not self._is_safe(candidate):
            return False

        return any(
            other is not candidate
            and self._is_dangerous(other)
            and self._intersection_over_union(candidate, other)
            >= self.config.overlap_iou_threshold
            for other in candidates
        )

    def _is_safe(self, item: PrioritizedDetection) -> bool:
        """Nesnenin yapılandırılmış güvenli sınıflardan biri olup olmadığını döndürür."""
        return normalize_label(item.detection.label) in self._safe_labels

    def _is_dangerous(self, item: PrioritizedDetection) -> bool:
        """Nesnenin kural veya önem seviyesine göre tehlikeli olduğunu belirler."""
        return (
            normalize_label(item.detection.label) in self._dangerous_labels
            or item.danger.severity in {AlertSeverity.HIGH, AlertSeverity.CRITICAL}
        )

    @staticmethod
    def _intersection_over_union(
        first: PrioritizedDetection,
        second: PrioritizedDetection,
    ) -> float:
        """İki sınır kutusunun IoU çakışma oranını hesaplar."""
        first_x1, first_y1, first_x2, first_y2 = first.detection.bbox
        second_x1, second_y1, second_x2, second_y2 = second.detection.bbox

        intersection_width = max(
            0,
            min(first_x2, second_x2) - max(first_x1, second_x1),
        )
        intersection_height = max(
            0,
            min(first_y2, second_y2) - max(first_y1, second_y1),
        )
        intersection_area = intersection_width * intersection_height

        first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
        second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
        union_area = first_area + second_area - intersection_area

        if union_area <= 0:
            return 0.0
        return intersection_area / union_area
