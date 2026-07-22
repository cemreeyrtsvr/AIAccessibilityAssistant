"""Nesne algılamalarını güvenlik riski bakımından değerlendirme araçları."""

from __future__ import annotations

from pydantic import BaseModel, Field

from decision.rules import (
    DEFAULT_DECISION_SETTINGS,
    DEFAULT_DANGER_RULES,
    DEFAULT_FALLBACK_RULE,
    SEVERITY_THRESHOLDS,
    AlertSeverity,
    DecisionSettings,
    DangerRule,
    normalize_label,
)
from models.detection import Detection


class DangerAssessment(BaseModel):
    """Bir algılamanın hesaplanmış risk bilgisini taşır."""

    score: int = Field(..., ge=0, le=100)
    severity: AlertSeverity
    reason: str
    proximity: float = Field(..., ge=0.0, le=1.0)


class DangerDetector:
    """Nesne türü, güveni ve kapladığı alanla risk puanı üretir."""

    def __init__(
        self,
        rules: dict[str, DangerRule] | None = None,
        settings: DecisionSettings | None = None,
    ) -> None:
        self.rules = rules if rules is not None else DEFAULT_DANGER_RULES
        self.settings = settings or DEFAULT_DECISION_SETTINGS

    def assess(
        self,
        detection: Detection,
        frame_size: tuple[int, int] | None = None,
    ) -> DangerAssessment:
        """Tek bir algılamanın güvenlik riskini değerlendirir."""
        proximity = self._calculate_proximity(detection, frame_size)
        rule = self.rules.get(normalize_label(detection.label))

        active_rule = rule or DEFAULT_FALLBACK_RULE

        score = round(
            min(
                self.settings.max_score,
                active_rule.base_danger * self.settings.danger_base_weight
                + proximity * self.settings.danger_proximity_weight
                + detection.confidence * self.settings.danger_confidence_weight,
            )
        )
        severity = self._severity_for(score)

        return DangerAssessment(
            score=score,
            severity=severity,
            reason=active_rule.reason,
            proximity=proximity,
        )

    def _calculate_proximity(
        self,
        detection: Detection,
        frame_size: tuple[int, int] | None,
    ) -> float:
        """Sınır kutusunun kapladığı alandan yakınlık tahmini yapar."""
        if frame_size is None:
            return self.settings.default_proximity

        frame_width, frame_height = frame_size
        x1, y1, x2, y2 = detection.bbox
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        frame_area = frame_width * frame_height

        if frame_area <= 0:
            return self.settings.default_proximity

        return min(1.0, box_area / (frame_area * self.settings.near_object_area_ratio))

    @staticmethod
    def _severity_for(score: int) -> AlertSeverity:
        """Risk puanını standart önem seviyesine dönüştürür."""
        for threshold, severity in SEVERITY_THRESHOLDS:
            if score >= threshold:
                return severity

        return AlertSeverity.LOW
