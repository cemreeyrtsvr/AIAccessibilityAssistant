"""Algılanan nesneleri ve uyarılardan yapılandırılmış sahne haritası çıkaran analiz servisi."""

from __future__ import annotations

from typing import Iterable

from config.settings import MAX_SCENE_OBJECTS, MIN_SCENE_PRIORITY
from decision.priority_engine import PrioritizedDetection
from models.detection import Detection, Direction
from models.scene import SceneObject, StructuredScene


class SceneAnalyzer:
    """Algılanan nesnelerden konum-etiket duyarlı yapılandırılmış sahne haritası süzmektedir."""

    def __init__(
        self,
        max_objects: int = MAX_SCENE_OBJECTS,
        min_priority: int = MIN_SCENE_PRIORITY,
    ) -> None:
        if max_objects < 1:
            raise ValueError("max_objects en az 1 olmalıdır.")
        self.max_objects = max_objects
        self.min_priority = min_priority

    def analyze_scene(
        self,
        items: Iterable[Detection | PrioritizedDetection | object] | None,
    ) -> StructuredScene:
        """Nesne ve öncelik listesinden tekilleştirilmiş ve filtrelenmiş StructuredScene üretir."""
        if items is None:
            return StructuredScene(objects=[], total_detected=0)

        item_list = list(items) if not isinstance(items, list) else items
        if not item_list:
            return StructuredScene(objects=[], total_detected=0)

        converted_objects: list[SceneObject] = []
        for item in item_list:
            scene_obj = self._to_scene_object(item)
            if scene_obj is not None:
                converted_objects.append(scene_obj)

        sorted_objects = sorted(
            converted_objects,
            key=lambda obj: (
                obj.priority if obj.priority is not None else -1,
                obj.confidence,
            ),
            reverse=True,
        )

        filtered_objects: list[SceneObject] = []
        seen_keys: set[tuple[str, Direction]] = set()

        for obj in sorted_objects:
            if (
                obj.priority is not None
                and obj.priority < self.min_priority
                and len(filtered_objects) > 0
            ):
                continue

            normalized_label = " ".join(obj.label.casefold().split())
            key = (normalized_label, obj.direction)

            if key not in seen_keys:
                seen_keys.add(key)
                filtered_objects.append(obj)

            if len(filtered_objects) == self.max_objects:
                break

        return StructuredScene(
            objects=filtered_objects,
            total_detected=len(converted_objects),
        )

    @staticmethod
    def _to_scene_object(
        item: Detection | PrioritizedDetection | object,
    ) -> SceneObject | None:
        """Farklı nesne türlerini standart SceneObject yapısına dönüştürür."""
        if isinstance(item, Detection):
            return SceneObject(
                label=item.label,
                direction=item.direction,
                distance=item.distance,
                priority=None,
                confidence=item.confidence,
            )
        if isinstance(item, PrioritizedDetection):
            return SceneObject(
                label=item.detection.label,
                direction=item.detection.direction,
                distance=item.detection.distance,
                priority=item.priority,
                confidence=item.detection.confidence,
            )
        if hasattr(item, "label") and hasattr(item, "direction"):
            return SceneObject(
                label=getattr(item, "label"),
                direction=getattr(item, "direction"),
                distance=getattr(item, "distance", None),
                priority=getattr(item, "priority", None),
                confidence=getattr(item, "confidence", 1.0),
            )
        if hasattr(item, "object") and hasattr(item, "direction"):
            return SceneObject(
                label=getattr(item, "object"),
                direction=getattr(item, "direction"),
                distance=getattr(item, "distance", None),
                priority=getattr(item, "priority", None),
                confidence=getattr(item, "confidence", 1.0),
            )
        return None
