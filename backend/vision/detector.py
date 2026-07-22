"""YOLO11-based object detection for OpenCV image frames."""

from __future__ import annotations

from threading import Lock
from typing import ClassVar

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import CONFIDENCE_THRESHOLD
from models.detection import Detection, Direction


class ObjectDetector:
    """Detect objects in OpenCV frames with a cached Ultralytics YOLO11n model."""

    _models: ClassVar[dict[str, YOLO]] = {}
    _model_lock: ClassVar[Lock] = Lock()
    _inference_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = self._get_model(model_name)

    @classmethod
    def _get_model(cls, model_name: str) -> YOLO:
        """Return a process-wide cached model, loading it once when necessary."""
        with cls._model_lock:
            if model_name not in cls._models:
                cls._models[model_name] = YOLO(model_name)
            return cls._models[model_name]

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """Run YOLO11n inference on one OpenCV BGR frame.

        Args:
            frame: A non-empty OpenCV image in BGR format.

        Returns:
            Detected objects with coordinates relative to the input frame.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be an OpenCV numpy array.")
        if frame.size == 0:
            raise ValueError("frame must be a non-empty OpenCV image.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR OpenCV image.")

        prediction_kwargs: dict[str, object] = {
            "source": frame,
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if self.device is not None:
            prediction_kwargs["device"] = self.device

        # Aynı model örneğine eşzamanlı erişimi güvenli hâle getirir.
        with self._inference_lock:
            results = self.model.predict(**prediction_kwargs)
        detections: list[Detection] = []
        frame_height, frame_width = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = self._clamp_bbox(
                    box.xyxy[0].tolist(),
                    frame_width,
                    frame_height,
                )
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                detections.append(
                    Detection(
                        label=str(result.names[class_id]),
                        confidence=float(box.conf.item()),
                        bbox=(x1, y1, x2, y2),
                        center_x=center_x,
                        center_y=center_y,
                        direction=self._direction_for(center_x, frame_width),
                        distance=None,
                        class_id=class_id,
                    )
                )

        return detections

    @staticmethod
    def _direction_for(center_x: int, frame_width: int) -> Direction:
        """Nesne merkezini görüntünün yatay üçte birine göre sınıflandırır."""
        if center_x < frame_width / 3:
            return Direction.LEFT
        if center_x >= (frame_width * 2) / 3:
            return Direction.RIGHT
        return Direction.CENTER

    @staticmethod
    def _clamp_bbox(
        raw_bbox: list[float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        """YOLO koordinatlarını giriş karesinin geçerli sınırlarında tutar."""
        x1, y1, x2, y2 = (round(value) for value in raw_bbox)
        return (
            min(max(x1, 0), frame_width - 1),
            min(max(y1, 0), frame_height - 1),
            min(max(x2, 0), frame_width - 1),
            min(max(y2, 0), frame_height - 1),
        )
