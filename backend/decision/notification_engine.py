"""Önceliklendirilmiş algılamaları kullanıcı uyarılarına dönüştürür."""

from __future__ import annotations

from pydantic import BaseModel, Field

from decision.accessibility_filter import AccessibilityFilter, AccessibilityFilterConfig
from decision.context_suppressor import ContextAwareSuppressor
from decision.priority_engine import PrioritizedDetection, PriorityEngine
from decision.rules import AlertSeverity
from memory.short_memory import ShortTermMemory
from models.detection import Detection, Direction


class Alert(BaseModel):
    """İstemci katmanlarının işleyebileceği yapılandırılmış uyarı."""

    object: str = Field(..., min_length=1)
    severity: AlertSeverity
    direction: Direction
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    distance: float | None = Field(default=None, ge=0.0)
    priority: int = Field(..., ge=0, le=100)
    reason: str


class Alerts(BaseModel):
    """Eşzamanlı uyarılardan oluşan toplanmış uyarı kapsayıcısı."""

    alerts: list[Alert] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.alerts)

    def __getitem__(self, index: int) -> Alert:
        return self.alerts[index]

    def __iter__(self):
        return iter(self.alerts)


class NotificationEngine:
    """Bir karedeki en önemli algılamalar için kısa uyarılar oluşturur."""

    def __init__(
        self,
        priority_engine: PriorityEngine | None = None,
        accessibility_filter: AccessibilityFilter | None = None,
        context_suppressor: ContextAwareSuppressor | None = None,
        short_memory: ShortTermMemory | None = None,
        max_alerts: int = 3,
    ) -> None:
        if max_alerts < 1:
            raise ValueError("max_alerts en az 1 olmalıdır.")

        self.max_alerts = max_alerts
        self.priority_engine = priority_engine or PriorityEngine()
        self.accessibility_filter = accessibility_filter or AccessibilityFilter(
            AccessibilityFilterConfig(max_alerts=max_alerts)
        )
        self.context_suppressor = context_suppressor or ContextAwareSuppressor()
        self.short_memory = short_memory

    def aggregate_alerts(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int] | None = None,
    ) -> Alerts:
        """Uygun algılamaları önceliklendirip sınırlı tek bir Alerts nesnesinde toplar."""
        eligible_detections = self.accessibility_filter.filter_detections(detections)
        prioritized = self.priority_engine.prioritize(eligible_detections, frame_size)
        contextually_relevant = self.context_suppressor.suppress(prioritized)

        if self.short_memory is not None:
            contextually_relevant = [
                item
                for item in contextually_relevant
                if self.short_memory.should_speak(item.detection)
            ]

        selected = self.accessibility_filter.sort_and_limit(contextually_relevant)

        sorted_selected = sorted(
            selected,
            key=lambda item: (
                item.priority,
                item.danger.score,
                item.detection.confidence,
            ),
            reverse=True,
        )[: self.max_alerts]

        return Alerts(alerts=[self._to_alert(item) for item in sorted_selected])

    def create_alerts(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int] | None = None,
    ) -> list[Alert]:
        """Uygun algılamalardan önceliklendirilmiş yapılandırılmış uyarılar üretir."""
        return self.aggregate_alerts(detections, frame_size).alerts

    @staticmethod
    def _to_alert(item: PrioritizedDetection) -> Alert:
        """Öncelik sonucunu metin üretmeden yapılandırılmış uyarıya çevirir."""
        return Alert(
            object=item.detection.label,
            severity=item.danger.severity,
            direction=item.detection.direction,
            confidence=item.detection.confidence,
            distance=item.detection.distance,
            priority=item.priority,
            reason=item.danger.reason,
        )
