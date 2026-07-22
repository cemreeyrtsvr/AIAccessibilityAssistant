"""Güvenlik değerlendirmelerini kullanıcıya uygun öncelik sırasına koyar."""

from __future__ import annotations

from pydantic import BaseModel, Field

from decision.danger_detector import DangerAssessment, DangerDetector
from decision.rules import (
    DEFAULT_CATEGORY_FALLBACK_IMPORTANCE,
    DecisionSettings,
    normalize_label,
)
from models.detection import Detection


class PrioritizedDetection(BaseModel):
    """Bir algılamanın karar motorundaki öncelikli temsili."""

    detection: Detection
    danger: DangerAssessment
    priority: int = Field(..., ge=0, le=100)


class PriorityEngine:
    """Algılamaları risk, yakınlık, nesne kategorisi, yön ve güvene göre sıralar."""

    def __init__(
        self,
        danger_detector: DangerDetector | None = None,
        settings: DecisionSettings | None = None,
    ) -> None:
        self.danger_detector = danger_detector or DangerDetector()
        self.settings = settings or self.danger_detector.settings

    def prioritize(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int] | None = None,
    ) -> list[PrioritizedDetection]:
        """Algılamaları en yüksek öncelik önce gelecek şekilde döndürür."""
        prioritized = [
            self._prioritize_detection(detection, frame_size)
            for detection in detections
        ]
        return sorted(
            prioritized,
            key=lambda item: (
                item.priority,
                item.danger.score,
                item.detection.confidence,
            ),
            reverse=True,
        )

    def _prioritize_detection(
        self,
        detection: Detection,
        frame_size: tuple[int, int] | None,
    ) -> PrioritizedDetection:
        """Tek algılamanın uyarı önceliğini hesaplar."""
        danger = self.danger_detector.assess(detection, frame_size)

        if detection.distance is not None:
            proximity = max(
                0.0,
                min(
                    1.0,
                    1.0 - (detection.distance / self.settings.max_distance_meters),
                ),
            )
        else:
            proximity = danger.proximity

        label_normalized = normalize_label(detection.label)
        category_score = self.settings.category_importance.get(
            label_normalized,
            DEFAULT_CATEGORY_FALLBACK_IMPORTANCE,
        )

        direction_val = (
            detection.direction.value
            if hasattr(detection.direction, "value")
            else str(detection.direction)
        )
        direction_score = self.settings.direction_importance.get(
            direction_val.casefold(),
            0.7,
        )

        raw_priority = (
            danger.score * self.settings.priority_danger_weight
            + proximity * self.settings.priority_proximity_weight
            + detection.confidence * self.settings.priority_confidence_weight
            + category_score * self.settings.priority_category_weight
            + direction_score * self.settings.priority_direction_weight
        )

        priority = round(min(self.settings.max_score, raw_priority))

        return PrioritizedDetection(
            detection=detection,
            danger=danger,
            priority=priority,
        )
