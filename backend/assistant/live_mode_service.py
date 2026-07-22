"""Canlı Mod (Live Mode) için sürekli görüntü ve sesli uyarı orkestratörü."""

from __future__ import annotations

import numpy as np

from assistant.accessibility_reasoner import AccessibilityReasoner
from assistant.scene_analyzer import SceneAnalyzer
from assistant.scene_change_detector import SceneChangeDetector
from camera.frame_throttle import FrameThrottle
from decision.priority_engine import PriorityEngine
from memory.short_memory import ShortTermMemory
from models.response import LiveModeResult
from models.scene import StructuredScene
from speech.speech_manager import SpeechManager
from vision.detector import ObjectDetector


class LiveModeService:
    """Canlı Mod hattını orkestre eden, kare sınırlaması, sahne değişikliği ve erişilebilirlik akıl yürütmesi uygulayan servis."""

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        priority_engine: PriorityEngine | None = None,
        scene_analyzer: SceneAnalyzer | None = None,
        scene_change_detector: SceneChangeDetector | None = None,
        accessibility_reasoner: AccessibilityReasoner | None = None,
        short_memory: ShortTermMemory | None = None,
        speech_manager: SpeechManager | None = None,
        frame_throttle: FrameThrottle | None = None,
    ) -> None:
        self.detector = detector or ObjectDetector()
        self.priority_engine = priority_engine or PriorityEngine()
        self.scene_analyzer = scene_analyzer or SceneAnalyzer()
        self.scene_change_detector = scene_change_detector or SceneChangeDetector()
        self.accessibility_reasoner = accessibility_reasoner or AccessibilityReasoner()
        self.short_memory = short_memory or ShortTermMemory()
        self.speech_manager = speech_manager or SpeechManager()
        self.frame_throttle = frame_throttle or FrameThrottle()

    def process_frame(self, frame: np.ndarray) -> LiveModeResult:
        """Gelen kareden canlı mod işleme ve erişilebilirlik akıl yürütme hattını çalıştırır."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            empty_scene = StructuredScene(objects=[], total_detected=0)
            return LiveModeResult(spoken=False, scene=empty_scene, spoken_text=None)

        # 1. FrameThrottle
        if not self.frame_throttle.should_process():
            empty_scene = StructuredScene(objects=[], total_detected=0)
            return LiveModeResult(spoken=False, scene=empty_scene, spoken_text=None)

        frame_height, frame_width = frame.shape[:2]
        frame_size = (frame_width, frame_height)

        # 2. ObjectDetector
        detections = self.detector.detect(frame)

        # 3. PriorityEngine
        prioritized = self.priority_engine.prioritize(detections, frame_size)

        # 4. SceneAnalyzer
        structured_scene = self.scene_analyzer.analyze_scene(prioritized)

        # 5. SceneChangeDetector
        if not self.scene_change_detector.has_changed(structured_scene):
            return LiveModeResult(spoken=False, scene=structured_scene, spoken_text=None)

        # 6. AccessibilityReasoner
        reasoned_scene = self.accessibility_reasoner.reason(structured_scene)

        # 7. ShortMemory (süzgeç)
        new_objects = [
            obj
            for obj in reasoned_scene.objects_to_announce
            if self.short_memory.should_speak(obj)
        ]

        if not new_objects:
            return LiveModeResult(spoken=False, scene=structured_scene, spoken_text=None)

        # 8. SpeechManager (yüksek seviyeli konuşma kapsüllemesi)
        spoken_text = self.speech_manager.speak_scene(new_objects)

        return LiveModeResult(
            spoken=bool(spoken_text),
            scene=structured_scene,
            spoken_text=spoken_text,
        )
