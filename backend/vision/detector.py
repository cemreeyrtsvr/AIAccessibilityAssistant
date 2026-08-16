"""YOLO11-based object detection with global-optimal greedy tracking,
velocity prediction, label stability, EMA smoothing and direction hysteresis.

Tracking correctness design
───────────────────────────
Previous implementation used a per-track greedy loop: each track processed in
insertion order claimed the best available detection.  This caused a ghost track
(missed_frames > 0, stale CENTER position) to sometimes outbid the real track for
a new LEFT detection, which caused the real track to drop and a new one to spawn.

Current implementation:
  1. Score all (track, raw_detection) pairs globally.
  2. Sort candidates by score DESCENDING.
  3. Assign best pairs first — both track and raw are marked once assigned.
  4. Unassigned tracks increment missed_frames; beyond max they are logged and dropped.
  5. Unassigned raw detections become new tracks only at this point.

Velocity prediction
───────────────────
Each TrackedItem maintains a smoothed per-frame velocity (vx, vy).  When computing
the center-distance score for a track that has missed_frames > 0, the match uses the
predicted position  (center + vx * missed_frames, center + vy * missed_frames)
rather than the stale last-known position.  This keeps fast-moving objects matchable
even after 1–2 missed frames.

Debug logging (enabled per-match)
───────────────────────────────────
  [TRACK MATCH]        track_id matched a detection
  [TRACK NEW]          new track created
  [TRACK LOST]         track evicted after max_missed_frames
  [TRACK MISS]         track missed this frame (kept, missed_frames++)
  [TRACK MATCH FAILED] all candidates rejected (shows best distance and IoU found)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, ClassVar

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import CONFIDENCE_THRESHOLD
from models.detection import Detection, Direction

# Отладочные логи можно отключить в production, выставив False:
_DEBUG_TRACKING = True


def _dbg(msg: str) -> None:
    if _DEBUG_TRACKING:
        print(msg)


@dataclass
class TrackedItem:
    """Zaman boyunca izlenen tek bir nesne takibi durumu."""

    track_id: int
    label: str           # Onaylanmış (kararlı) etiket
    class_id: int | str
    bbox: tuple[int, int, int, int]
    center_x: int
    center_y: int
    direction: Direction
    confidence: float
    missed_frames: int = 0

    # Etiket kararlılığı
    candidate_label: str = field(default="")
    candidate_label_frames: int = field(default=0)

    # Yumuşatılmış hız (piksel / kare)
    vx: float = field(default=0.0)
    vy: float = field(default=0.0)

    @property
    def predicted_center(self) -> tuple[float, float]:
        """missed_frames kadar ilerletilmiş tahmini merkez konumu."""
        if self.missed_frames == 0:
            return float(self.center_x), float(self.center_y)
        return (
            self.center_x + self.vx * self.missed_frames,
            self.center_y + self.vy * self.missed_frames,
        )


class ObjectDetector:
    """YOLO11n ile küresel-optimal açgözlü takip, hız tahmini, etiket kararlılığı,
    yön histerezisi ve zamansal yumuşatma."""

    _models: ClassVar[dict[str, YOLO]] = {}
    _model_lock: ClassVar[Lock] = Lock()
    _inference_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: str | None = None,
        max_missed_frames: int = 12,
        hysteresis_margin_ratio: float = 0.03,
        iou_match_threshold: float = 0.10,
        center_distance_threshold_ratio: float = 0.85,
        label_stability_frames: int = 3,
    ) -> None:
        """
        Args:
            max_missed_frames:
                Bir iz kaç kare algılanmadan kalırsa kayıp sayılır.
                4 → ~400 ms @10 FPS; hızlı geçici ışık değişimlerinde kayıp olmaz.
            iou_match_threshold:
                IoU geçiş eşiği. 0.10 toleranslıdır; kısmen örtüşen kutuları yakalar.
            center_distance_threshold_ratio:
                Merkez mesafesi fallback eşiği (kare genişliği oranı).
                0.50 → kare genişliğinin %50'si; hızlı lateral hareketi kapsar.
            label_stability_frames:
                Yeni etiketin onay için kaç ardışık kare görülmesi gerektiği.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.max_missed_frames = max_missed_frames
        self.hysteresis_margin_ratio = hysteresis_margin_ratio
        self.iou_match_threshold = iou_match_threshold
        self.center_distance_threshold_ratio = center_distance_threshold_ratio
        self.label_stability_frames = label_stability_frames
        self.model = self._get_model(model_name)
        self._tracks: list[TrackedItem] = []
        self._next_track_id = 1

    # ─── Model önbelleği ───────────────────────────────────────────────────────

    @classmethod
    def _get_model(cls, model_name: str) -> YOLO:
        with cls._model_lock:
            if model_name not in cls._models:
                cls._models[model_name] = YOLO(model_name)
            return cls._models[model_name]

    # ─── Ana algılama + takip döngüsü ─────────────────────────────────────────

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """YOLO çıktısını küresel-optimal açgözlü takip ile Detection listesine dönüştürür."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be an OpenCV numpy array.")
        if frame.size == 0:
            raise ValueError("frame must be a non-empty OpenCV image.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR OpenCV image.")

        prediction_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if self.device is not None:
            prediction_kwargs["device"] = self.device

        with self._inference_lock:
            results = self.model.predict(**prediction_kwargs)

        frame_height, frame_width = frame.shape[:2]
        raw_detections: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = self._clamp_bbox(
                    box.xyxy[0].tolist(), frame_width, frame_height
                )
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                raw_detections.append(
                    {
                        "label": str(result.names[class_id]),
                        "class_id": class_id,
                        "confidence": float(box.conf.item()),
                        "bbox": (x1, y1, x2, y2),
                        "center_x": cx,
                        "center_y": cy,
                    }
                )

        # Class-agnostic duplicate suppression:
        # Sort raw detections by confidence descending so we keep the most confident one.
        raw_detections.sort(key=lambda d: d["confidence"], reverse=True)
        suppressed_detections: list[dict[str, Any]] = []
        dup_iou_threshold = 0.45
        dup_dist_threshold = frame_width * 0.05

        for raw in raw_detections:
            is_dup = False
            for kept in suppressed_detections:
                iou = self._calculate_iou(raw["bbox"], kept["bbox"])
                dist = math.hypot(raw["center_x"] - kept["center_x"], raw["center_y"] - kept["center_y"])
                if iou >= dup_iou_threshold or dist < dup_dist_threshold:
                    is_dup = True
                    _dbg(
                        f"[DUPLICATE SUPPRESSED] Suppressing raw '{raw['label']}' (conf={raw['confidence']:.2f}) "
                        f"due to overlap with kept '{kept['label']}' (conf={kept['confidence']:.2f}, iou={iou:.2f}, dist={dist:.1f}px)"
                    )
                    break
            if not is_dup:
                suppressed_detections.append(raw)
        raw_detections = suppressed_detections

        # ─── Aşama 1: Tüm (iz, algılama) çiftlerini puanla ───────────────────
        # score > 0  →  geçerli aday
        # Puanlar:
        # - Aynı etiket, IoU match: score = 3.0 + iou (3.1 to 4.0)
        # - Aynı etiket, Mesafe match: score = 2.0 + (1.0 - dist / max_dist) (2.0 to 3.0)
        # - Farklı etiket, IoU match: score = 1.0 + iou (1.1 to 2.0)
        # - Farklı etiket, Mesafe match: score = 0.0 + (1.0 - dist / (max_dist * 0.4)) (0.0 to 1.0)

        candidates: list[tuple[int, int, float]] = []  # (track_idx, raw_idx, score)
        max_dist = frame_width * self.center_distance_threshold_ratio

        for ti, track in enumerate(self._tracks):
            accepted = {track.label}
            if track.candidate_label:
                accepted.add(track.candidate_label)

            best_iou_found = 0.0
            best_dist_found = float("inf")

            for ri, raw in enumerate(raw_detections):
                iou = self._calculate_iou(track.bbox, raw["bbox"])
                px, py = track.predicted_center
                dist = math.hypot(raw["center_x"] - px, raw["center_y"] - py)

                best_iou_found = max(best_iou_found, iou)
                best_dist_found = min(best_dist_found, dist)

                is_same_label = raw["label"] in accepted

                if is_same_label:
                    if iou >= self.iou_match_threshold:
                        score = 3.0 + iou
                        candidates.append((ti, ri, score))
                    elif dist < max_dist:
                        score = 2.0 + (1.0 - dist / max_dist)
                        candidates.append((ti, ri, score))
                else:
                    # Relaxed matching for label changes
                    if iou >= 0.25:
                        score = 1.0 + iou
                        candidates.append((ti, ri, score))
                    elif dist < max_dist * 0.4:
                        score = 0.0 + (1.0 - dist / (max_dist * 0.4))
                        candidates.append((ti, ri, score))

            if not any(ti == c[0] for c in candidates):
                if raw_detections:
                    _dbg(
                        f"[TRACK MATCH FAILED] id={track.track_id} label={track.label} "
                        f"best_iou={best_iou_found:.3f} best_dist={best_dist_found:.1f}px "
                        f"(threshold={max_dist:.1f}px) missed={track.missed_frames}"
                    )

        # ─── Aşama 2: En iyi puandan başlayarak açgözlü atama ────────────────
        candidates.sort(key=lambda c: c[2], reverse=True)

        assigned_tracks: set[int] = set()
        assigned_raws: set[int] = set()
        assignments: dict[int, int] = {}  # track_idx → raw_idx

        for ti, ri, score in candidates:
            if ti in assigned_tracks or ri in assigned_raws:
                continue
            assignments[ti] = ri
            assigned_tracks.add(ti)
            assigned_raws.add(ri)

        # ─── Aşama 3: İzleri güncelle ─────────────────────────────────────────
        updated_tracks: list[TrackedItem] = []

        for ti, track in enumerate(self._tracks):
            if ti in assignments:
                ri = assignments[ti]
                raw = raw_detections[ri]

                # Etiket kararlılığı
                raw_label = raw["label"]
                if raw_label != track.label:
                    if raw_label == track.candidate_label:
                        track.candidate_label_frames += 1
                    else:
                        track.candidate_label = raw_label
                        track.candidate_label_frames = 1

                    if track.candidate_label_frames >= self.label_stability_frames:
                        _dbg(
                            f"[LABEL STABILIZED] id={track.track_id} "
                            f"{track.label} → {track.candidate_label}"
                        )
                        track.label = track.candidate_label
                        track.candidate_label = ""
                        track.candidate_label_frames = 0
                else:
                    track.candidate_label = ""
                    track.candidate_label_frames = 0

                smoothed_bbox = self._smooth_bbox(track.bbox, raw["bbox"])
                new_cx = (smoothed_bbox[0] + smoothed_bbox[2]) // 2
                new_cy = (smoothed_bbox[1] + smoothed_bbox[3]) // 2

                # Hız güncelleme (EMA, alpha=0.5)
                raw_vx = new_cx - track.center_x
                raw_vy = new_cy - track.center_y
                track.vx = 0.5 * raw_vx + 0.5 * track.vx
                track.vy = 0.5 * raw_vy + 0.5 * track.vy

                direction = self._direction_with_hysteresis(
                    new_cx, frame_width, track.direction
                )

                if direction != track.direction:
                    _dbg(
                        f"[TRACK UPDATE] id={track.track_id} {track.label} "
                        f"{track.direction.value} -> {direction.value}"
                    )
                else:
                    _dbg(f"[TRACK MATCH] id={track.track_id}")

                track.bbox = smoothed_bbox
                track.center_x = new_cx
                track.center_y = new_cy
                track.direction = direction
                track.confidence = raw["confidence"]
                track.missed_frames = 0
                updated_tracks.append(track)

            else:
                # Bu karede algılanamadı
                track.missed_frames += 1
                if track.missed_frames <= self.max_missed_frames:
                    track.confidence *= 0.9
                    _dbg(
                        f"[TRACK MISS] id={track.track_id} label={track.label} "
                        f"missed={track.missed_frames}/{self.max_missed_frames}"
                    )
                    updated_tracks.append(track)
                else:
                    _dbg(
                        f"[TRACK LOST] id={track.track_id} label={track.label} "
                        f"dir={track.direction.value} missed={track.missed_frames}"
                    )

        # ─── Aşama 4: Eşleşmeyen algılamalar → yeni izler ─────────────────────
        for ri, raw in enumerate(raw_detections):
            if ri not in assigned_raws:
                # Find why it didn't match any track
                reasons = []
                for ti, track in enumerate(self._tracks):
                    px, py = track.predicted_center
                    dist = math.hypot(raw["center_x"] - px, raw["center_y"] - py)
                    iou = self._calculate_iou(track.bbox, raw["bbox"])
                    reasons.append(
                        f"track_id={track.track_id} (label={track.label}, dist={dist:.1f}px, iou={iou:.2f}, pred_center=({px:.1f},{py:.1f}))"
                    )
                reasons_str = "; ".join(reasons) if reasons else "no existing tracks"

                direction = self._direction_with_hysteresis(
                    raw["center_x"], frame_width, None
                )
                new_track = TrackedItem(
                    track_id=self._next_track_id,
                    label=raw["label"],
                    class_id=raw["class_id"],
                    bbox=raw["bbox"],
                    center_x=raw["center_x"],
                    center_y=raw["center_y"],
                    direction=direction,
                    confidence=raw["confidence"],
                )
                _dbg(
                    f"[TRACK NEW] id={self._next_track_id} label={raw['label']} "
                    f"direction={direction.value}. Why: {reasons_str}"
                )
                self._next_track_id += 1
                updated_tracks.append(new_track)

        self._tracks = updated_tracks

        return [
            Detection(
                label=t.label,
                confidence=t.confidence,
                bbox=t.bbox,
                center_x=t.center_x,
                center_y=t.center_y,
                direction=t.direction,
                distance=None,
                class_id=t.class_id,
                tracker_id=t.track_id,
            )
            for t in self._tracks
            if t.missed_frames == 0
        ]

    # ─── Yön histerezisi ──────────────────────────────────────────────────────

    def _direction_with_hysteresis(
        self,
        center_x: int,
        frame_width: int,
        previous_direction: Direction | None,
    ) -> Direction:
        """Yatay sınırların etrafında titreşimi önleyen histerezisli yön hesaplama."""
        margin = int(frame_width * self.hysteresis_margin_ratio)
        left_thresh  = frame_width // 3
        right_thresh = (frame_width * 2) // 3

        if previous_direction == Direction.LEFT:
            if center_x >= left_thresh + margin:
                return Direction.RIGHT if center_x >= right_thresh else Direction.CENTER
            return Direction.LEFT

        if previous_direction == Direction.RIGHT:
            if center_x <= right_thresh - margin:
                return Direction.LEFT if center_x < left_thresh else Direction.CENTER
            return Direction.RIGHT

        if previous_direction == Direction.CENTER:
            if center_x < left_thresh - margin:
                return Direction.LEFT
            if center_x >= right_thresh + margin:
                return Direction.RIGHT
            return Direction.CENTER

        # İlk kez (previous_direction is None)
        if center_x < left_thresh:
            return Direction.LEFT
        if center_x >= right_thresh:
            return Direction.RIGHT
        return Direction.CENTER

    # ─── Statik yardımcılar ───────────────────────────────────────────────────

    @staticmethod
    def _smooth_bbox(
        old_box: tuple[int, int, int, int],
        new_box: tuple[int, int, int, int],
        alpha: float = 0.7,
    ) -> tuple[int, int, int, int]:
        """EMA ile sınır kutusu koordinatlarını yumuşatır (alpha=0.7 → hızlı tepki)."""
        return (
            round(alpha * new_box[0] + (1 - alpha) * old_box[0]),
            round(alpha * new_box[1] + (1 - alpha) * old_box[1]),
            round(alpha * new_box[2] + (1 - alpha) * old_box[2]),
            round(alpha * new_box[3] + (1 - alpha) * old_box[3]),
        )

    @staticmethod
    def _calculate_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """İki sınır kutusu arasındaki IoU değerini hesaplar."""
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        union_area = float(box_a_area + box_b_area - inter_area)
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _clamp_bbox(
        raw_bbox: list[float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        """YOLO koordinatlarını giriş karesinin geçerli sınırlarında tutar."""
        x1, y1, x2, y2 = (round(v) for v in raw_bbox)
        return (
            min(max(x1, 0), frame_width - 1),
            min(max(y1, 0), frame_height - 1),
            min(max(x2, 0), frame_width - 1),
            min(max(y2, 0), frame_height - 1),
        )
