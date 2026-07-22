"""Canlı erişilebilirlik arka plan işleme hattı servisi."""

from __future__ import annotations

import cv2
import numpy as np

from camera.camera import Camera
from decision.notification_engine import NotificationEngine
from memory.short_memory import ShortTermMemory
from models.response import Alerts
from vision.detector import ObjectDetector


class AssistantService:
    """Tüm modülleri tek bir canlı işleme hattında birleştiren ana servis."""

    def __init__(
        self,
        camera: Camera | None = None,
        detector: ObjectDetector | None = None,
        notification_engine: NotificationEngine | None = None,
        short_memory: ShortTermMemory | None = None,
    ) -> None:
        self.camera = camera
        self.detector = detector or ObjectDetector()
        self.short_memory = short_memory or ShortTermMemory()
        self.notification_engine = notification_engine or NotificationEngine(
            short_memory=self.short_memory
        )

    def process_frame(self, frame: np.ndarray) -> Alerts:
        """Görüntü karesini canlı işleme hattından geçirerek Alerts nesnesi döndürür.

        İşleme Hattı:
        Camera / Frame
        → ObjectDetector
        → Accessibility Filter
        → Priority Engine
        → Danger Detector
        → Short Memory
        → Alert Aggregation
        """
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return Alerts()

        frame_height, frame_width = frame.shape[:2]
        frame_size = (frame_width, frame_height)

        detections = self.detector.detect(frame)
        alerts = self.notification_engine.aggregate_alerts(detections, frame_size)
        return alerts

    def process_live_frame(self) -> Alerts:
        """Kameradan bir kare yakalayıp hattan geçirerek Alerts nesnesi döndürür."""
        if self.camera is None:
            raise RuntimeError("Kamera nesnesi ilklendirilmemiş.")

        frame = self.camera.capture_frame()
        if frame is None:
            return Alerts()

        return self.process_frame(frame)
