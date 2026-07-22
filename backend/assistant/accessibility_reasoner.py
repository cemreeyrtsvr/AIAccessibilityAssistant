"""Erişilebilirlik Akıl Yürütme (AccessibilityReasoner) bileşeni."""

from __future__ import annotations

from decision.rules import AlertSeverity, normalize_label
from models.detection import Direction
from models.scene import (
    NavigationHint,
    ReasonedObject,
    ReasonedScene,
    SemanticCategory,
    StructuredScene,
    WarningInfo,
)

CATEGORY_MAPPING: dict[str, SemanticCategory] = {
    "person": SemanticCategory.HUMAN,
    "car": SemanticCategory.VEHICLE,
    "bus": SemanticCategory.VEHICLE,
    "truck": SemanticCategory.VEHICLE,
    "motorcycle": SemanticCategory.VEHICLE,
    "bicycle": SemanticCategory.VEHICLE,
    "train": SemanticCategory.VEHICLE,
    "knife": SemanticCategory.DANGEROUS_OBJECT,
    "scissors": SemanticCategory.DANGEROUS_OBJECT,
    "door": SemanticCategory.NAVIGATION_LANDMARK,
    "traffic light": SemanticCategory.NAVIGATION_LANDMARK,
    "stairs": SemanticCategory.DANGEROUS_OBJECT,
    "pole": SemanticCategory.DANGEROUS_OBJECT,
    "chair": SemanticCategory.FURNITURE,
    "table": SemanticCategory.FURNITURE,
    "bench": SemanticCategory.FURNITURE,
    "bottle": SemanticCategory.BACKGROUND_OBJECT,
    "cup": SemanticCategory.BACKGROUND_OBJECT,
    "keyboard": SemanticCategory.BACKGROUND_OBJECT,
}


class AccessibilityReasoner:
    """Sahne nesnelerini görme engelli kullanıcı ihtiyaçlarına göre değerlendirir ve süzmektedir."""

    def __init__(
        self,
        max_announcements: int = 3,
        min_confidence: float = 0.50,
        max_informational_distance: float = 6.0,
    ) -> None:
        self.max_announcements = max_announcements
        self.min_confidence = min_confidence
        self.max_informational_distance = max_informational_distance

    def reason(self, scene: StructuredScene) -> ReasonedScene:
        """StructuredScene nesnesini değerlendirip önceliklendirilmiş ReasonedScene döndürür."""
        if scene is None or not scene.objects:
            return ReasonedScene(
                objects_to_announce=[],
                max_danger_level=AlertSeverity.LOW,
                warnings=[],
                navigation_hints=[],
            )

        reasoned_objects: list[ReasonedObject] = []
        warnings: list[WarningInfo] = []
        hints: list[NavigationHint] = []
        max_danger = AlertSeverity.LOW

        for obj in scene.objects:
            if obj.confidence < self.min_confidence:
                continue

            if (
                obj.distance is not None
                and obj.distance > self.max_informational_distance
            ):
                continue

            label_norm = normalize_label(obj.label)
            category = CATEGORY_MAPPING.get(
                label_norm, SemanticCategory.BACKGROUND_OBJECT
            )
            danger_level, is_warning = self._evaluate_danger(
                label_norm, category, obj.direction, obj.distance
            )

            is_landmark = category == SemanticCategory.NAVIGATION_LANDMARK

            reasoned_obj = ReasonedObject(
                label=obj.label,
                direction=obj.direction,
                distance=obj.distance,
                priority=obj.priority,
                confidence=obj.confidence,
                category=category,
                danger_level=danger_level,
                is_warning=is_warning,
                is_landmark=is_landmark,
            )
            reasoned_objects.append(reasoned_obj)

            if self._severity_rank(danger_level) > self._severity_rank(max_danger):
                max_danger = danger_level

            if is_warning:
                warnings.append(
                    WarningInfo(
                        object_label=obj.label,
                        direction=obj.direction,
                        danger_level=danger_level,
                        distance=obj.distance,
                        reason=f"{obj.label} engeli",
                    )
                )

            if is_landmark:
                hints.append(
                    NavigationHint(
                        object_label=obj.label,
                        direction=obj.direction,
                        distance=obj.distance,
                        landmark_type=label_norm,
                    )
                )

        sorted_objects = sorted(
            reasoned_objects,
            key=lambda item: (
                self._severity_rank(item.danger_level),
                item.priority if item.priority is not None else -1,
                item.confidence,
                -item.distance if item.distance is not None else -999,
            ),
            reverse=True,
        )

        selected_objects = sorted_objects[: self.max_announcements]

        return ReasonedScene(
            objects_to_announce=selected_objects,
            max_danger_level=max_danger,
            warnings=warnings,
            navigation_hints=hints,
        )

    def _evaluate_danger(
        self,
        label: str,
        category: SemanticCategory,
        direction: Direction,
        distance: float | None,
    ) -> tuple[AlertSeverity, bool]:
        """Nesne türü, yönü ve mesafesine göre tehlike seviyesini ve uyarı durumunu hesaplar."""
        if label in {"knife", "scissors"}:
            return AlertSeverity.CRITICAL, True

        if label in {"stairs", "pole"} and direction == Direction.CENTER:
            return AlertSeverity.HIGH, True

        if category == SemanticCategory.VEHICLE:
            if distance is not None and distance <= 2.0:
                return AlertSeverity.CRITICAL, True
            return AlertSeverity.HIGH, False

        if category == SemanticCategory.HUMAN:
            if distance is not None and distance <= 1.0:
                return AlertSeverity.MEDIUM, True
            return AlertSeverity.LOW, False

        return AlertSeverity.LOW, False

    @staticmethod
    def _severity_rank(severity: AlertSeverity) -> int:
        """Tehlike önem seviyesini sayısal sıralama değerine dönüştürür."""
        ranks = {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4,
        }
        return ranks.get(severity, 1)
