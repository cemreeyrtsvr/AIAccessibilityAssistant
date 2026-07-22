"""Erişilebilirlik uyarıları için algılama sonuçlarını azaltma ve sıralama katmanı."""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.settings import CONFIDENCE_THRESHOLD
from decision.priority_engine import PrioritizedDetection
from decision.rules import normalize_label
from models.detection import Detection


class AccessibilityFilterConfig(BaseModel):
    """Erişilebilirlik filtresinin değiştirilebilir davranışını tanımlar."""

    minimum_confidence: float = Field(
        default=CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    non_important_labels: set[str] = Field(default_factory=set)
    important_attribute_name: str = Field(default="important", min_length=1)
    max_alerts: int = Field(default=3, ge=1)


class AccessibilityFilter:
    """Kullanıcıya iletilmeye değer algılamaları seçer."""

    def __init__(self, config: AccessibilityFilterConfig | None = None) -> None:
        self.config = config or AccessibilityFilterConfig()
        self._non_important_labels = {
            normalize_label(label) for label in self.config.non_important_labels
        }

    def filter_detections(self, detections: list[Detection]) -> list[Detection]:
        """Güven eşiğinin altındaki ve önemsiz işaretli algılamaları çıkarır."""
        return [
            detection
            for detection in detections
            if self._is_eligible(detection)
        ]

    def sort_and_limit(
        self,
        prioritized_detections: list[PrioritizedDetection],
    ) -> list[PrioritizedDetection]:
        """Kalan algılamaları önceliğe göre sıralar ve uyarı sayısını sınırlar."""
        selected: list[PrioritizedDetection] = []

        for item in sorted(
            prioritized_detections,
            key=lambda prioritized: prioritized.priority,
            reverse=True,
        ):
            if not self._is_eligible(item.detection):
                continue

            selected.append(item)

            if len(selected) == self.config.max_alerts:
                break

        return selected

    def _is_eligible(self, detection: Detection) -> bool:
        """Bir algılamanın kullanıcı uyarısına dönüşüp dönüşmeyeceğini belirler."""
        if detection.confidence < self.config.minimum_confidence:
            return False
        if normalize_label(detection.label) in self._non_important_labels:
            return False

        important_value = detection.attributes.get(
            self.config.important_attribute_name,
            True,
        )
        return important_value is not False
