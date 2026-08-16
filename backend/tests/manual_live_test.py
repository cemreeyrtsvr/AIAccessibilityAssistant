"""Manuel End-to-End Canlı Test Scripti.

AI Accessibility Assistant arka planının canlı webcam ortamında tüm modüllerini
gerçek nesneler ve bağımlılık enjeksiyonu ile test eder.

Varsayılan mod (kompakt):
  Tek satır durum + yalnızca event olduğunda log.

Ayrıntılı mod (P tuşu):
  Her pipeline aşamasının çıktısı yazdırılır.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Sequence

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import colorama
    colorama.init(autoreset=True)
except ImportError:
    pass

from assistant.accessibility_reasoner import AccessibilityReasoner
from assistant.live_mode_service import LiveModeService
from assistant.scene_analyzer import SceneAnalyzer
from assistant.scene_change_detector import SceneChangeDetector
from camera.camera import Camera
from camera.frame_throttle import FrameThrottle
from decision.priority_engine import PriorityEngine
from memory.short_memory import ShortTermMemory
from models.detection import Detection
from models.response import LiveModeResult
from models.scene import ReasonedScene, StructuredScene
from speech.speech_manager import SpeechManager
from vision.detector import ObjectDetector


# ─── Renkler ───────────────────────────────────────────────────────────────────

class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def _g(msg: str) -> None: print(f"{C.GREEN}{msg}{C.RESET}")
def _y(msg: str) -> None: print(f"{C.YELLOW}{msg}{C.RESET}")
def _r(msg: str) -> None: print(f"{C.RED}{msg}{C.RESET}")
def _b(msg: str) -> None: print(f"{C.BLUE}{msg}{C.RESET}")
def _c(msg: str) -> None: print(f"{C.CYAN}{msg}{C.RESET}")


# ─── Durum takipçisi ───────────────────────────────────────────────────────────

class FrameState:
    """Tek bir işlenmiş kareye ait ara durum bilgileri."""

    def __init__(self) -> None:
        self.verbose: bool = False  # pipeline döngüsünde korunur
        self.reset()

    def reset(self) -> None:
        """verbose değerini koruyarak frame durumunu sıfırla."""
        self.throttled: bool = False
        self.detections: list[Detection] = []
        self.prioritized: list[Any] = []
        self.structured_scene: StructuredScene | None = None
        self.scene_changed: bool = False
        self.reasoned_scene: ReasonedScene | None = None
        self.filtered_objects: list[Any] = []
        self.spoken_text: str | None = None
        self.detector_ms: float = 0.0
        self.reasoning_ms: float = 0.0
        self.pipeline_ms: float = 0.0


# ─── Araçlanmış sarmalayıcılar ─────────────────────────────────────────────────

class InstrumentedDetector:
    def __init__(self, real: ObjectDetector, state: FrameState) -> None:
        self.real = real
        self.state = state

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.state.verbose:
            _b("  [STAGE 1/7] Object Detection…")
        t0 = perf_counter()
        try:
            dets = self.real.detect(frame)
            self.state.detector_ms = (perf_counter() - t0) * 1000
            self.state.detections = dets
            if self.state.verbose:
                _b(f"  [STAGE 1/7] Done — {len(dets)} detected ({self.state.detector_ms:.1f} ms)")
            return dets
        except Exception as e:
            _r(f"[ERROR] ObjectDetector: {e}")
            return []


class InstrumentedPriorityEngine:
    def __init__(self, real: PriorityEngine, state: FrameState) -> None:
        self.real = real
        self.state = state

    def prioritize(
        self, detections: list[Detection], frame_size: tuple[int, int] | None = None
    ) -> list[Any]:
        if self.state.verbose:
            _b("  [STAGE 2/7] Priority Engine…")
        try:
            result = self.real.prioritize(detections, frame_size)
            self.state.prioritized = result
            if self.state.verbose:
                _b(f"  [STAGE 2/7] Done — {len(result)} prioritized")
            return result
        except Exception as e:
            _r(f"[ERROR] PriorityEngine: {e}")
            return []


class InstrumentedSceneAnalyzer:
    def __init__(self, real: SceneAnalyzer, state: FrameState) -> None:
        self.real = real
        self.state = state

    def analyze_scene(self, prioritized: list[Any]) -> StructuredScene:
        if self.state.verbose:
            _b("  [STAGE 3/7] Scene Analysis…")
        try:
            scene = self.real.analyze_scene(prioritized)
            self.state.structured_scene = scene
            if self.state.verbose:
                _b(f"  [STAGE 3/7] Done — {len(scene.objects)} objects")
            return scene
        except Exception as e:
            _r(f"[ERROR] SceneAnalyzer: {e}")
            empty = StructuredScene(objects=[], total_detected=0)
            self.state.structured_scene = empty
            return empty


class InstrumentedSceneChangeDetector:
    def __init__(self, real: SceneChangeDetector, state: FrameState) -> None:
        self.real = real
        self.state = state

    def has_changed(self, scene: StructuredScene) -> bool:
        if self.state.verbose:
            _b("  [STAGE 4/7] Scene Change Detection…")
        try:
            changed = self.real.has_changed(scene)
            self.state.scene_changed = changed
            if self.state.verbose:
                _b(f"  [STAGE 4/7] Changed: {changed}")
            return changed
        except Exception as e:
            _r(f"[ERROR] SceneChangeDetector: {e}")
            return True


class InstrumentedAccessibilityReasoner:
    def __init__(self, real: AccessibilityReasoner, state: FrameState) -> None:
        self.real = real
        self.state = state

    def reason(self, scene: StructuredScene) -> ReasonedScene:
        if self.state.verbose:
            _b("  [STAGE 5/7] Accessibility Reasoning…")
        t0 = perf_counter()
        try:
            reasoned = self.real.reason(scene)
            self.state.reasoning_ms = (perf_counter() - t0) * 1000
            self.state.reasoned_scene = reasoned
            if self.state.verbose:
                _b(
                    f"  [STAGE 5/7] Done — {len(reasoned.objects_to_announce)} candidates "
                    f"({self.state.reasoning_ms:.1f} ms)"
                )
            return reasoned
        except Exception as e:
            _r(f"[ERROR] AccessibilityReasoner: {e}")
            return ReasonedScene()


class InstrumentedShortMemory:
    def __init__(self, real: ShortTermMemory, state: FrameState) -> None:
        self.real = real
        self.state = state

    def should_speak(self, item: Any) -> bool:
        try:
            res = self.real.should_speak(item)
            if res:
                self.state.filtered_objects.append(item)
            if self.state.verbose:
                _b(f"  [STAGE 6/7] Memory: '{getattr(item, 'label', item)}' → {res}")
            return res
        except Exception as e:
            _r(f"[ERROR] ShortTermMemory should_speak: {e}")
            return True

    def update_last_seen(self, items: list[Any]) -> None:
        try:
            self.real.update_last_seen(items)
        except Exception as e:
            _r(f"[ERROR] ShortTermMemory update_last_seen: {e}")

    def remember_spoken(self, item: Any) -> None:
        self.real.remember_spoken(item)

    def has_recent_alert(self, item: Any) -> bool:
        return self.real.has_recent_alert(item)

    def clear(self) -> None:
        self.real.clear()

    @property
    def size(self) -> int:
        return self.real.size

    @property
    def _entries(self) -> list[Any]:
        return self.real._entries

    @property
    def _lock(self) -> Any:
        return self.real._lock


class InstrumentedSpeechManager:
    def __init__(self, real: SpeechManager, state: FrameState) -> None:
        self.real = real
        self.state = state

    def speak_scene(self, items: Sequence[Any]) -> str | None:
        if self.state.verbose:
            _b("  [STAGE 7/7] SpeechManager…")
        try:
            spoken = self.real.speak_scene(items)
            self.state.spoken_text = spoken
            if self.state.verbose:
                _b(f"  [STAGE 7/7] Done — '{spoken}'")
            return spoken
        except Exception as e:
            _r(f"[ERROR] SpeechManager: {e}")
            return None


class InstrumentedFrameThrottle:
    def __init__(self, real: FrameThrottle, state: FrameState) -> None:
        self.real = real
        self.state = state

    def should_process(self) -> bool:
        try:
            result = self.real.should_process()
            if not result:
                self.state.throttled = True
            return result
        except Exception as e:
            _r(f"[ERROR] FrameThrottle: {e}")
            return True


# ─── Overlay çizimi ────────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    state: FrameState,
    fps: float,
    mem_size: int,
) -> np.ndarray:
    annotated = frame.copy()

    for det in state.detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{det.label} {det.confidence:.0%} ({det.direction.value})",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

    status = (
        f"FPS: {fps:.1f} | Pipeline: {state.pipeline_ms:.0f}ms | "
        f"Mem: {mem_size}{' | [VERBOSE]' if state.verbose else ''}"
    )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), (0, 0, 0), cv2.FILLED)
    cv2.putText(annotated, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    help_bar = "[Q] Quit  [S] Save  [T] Memory  [C] Clear  [P] Verbose"
    cv2.rectangle(
        annotated, (0, annotated.shape[0] - 30),
        (annotated.shape[1], annotated.shape[0]), (0, 0, 0), cv2.FILLED,
    )
    cv2.putText(
        annotated, help_bar, (10, annotated.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )
    return annotated


# ─── Kompakt durum satırı ──────────────────────────────────────────────────────

def _compact_status(state: FrameState, fps: float) -> None:
    """İşlenmiş her kare için tek satır durum (satır sonu yok, üzerine yazar)."""
    if state.structured_scene and state.structured_scene.objects:
        labels = ", ".join(o.label for o in state.structured_scene.objects[:3])
        dirs   = ", ".join(o.direction.value for o in state.structured_scene.objects[:3])
    else:
        labels = "—"
        dirs   = "—"

    speaking_tag = "Speaking" if state.spoken_text else "Monitoring"
    line = (
        f"\r{C.CYAN}FPS: {fps:5.1f}{C.RESET} | "
        f"Pipeline: {state.pipeline_ms:3.0f}ms | "
        f"Objects: {labels} | Dir: {dirs} | {speaking_tag}          "
    )
    print(line, end="", flush=True)


# ─── Ana döngü ─────────────────────────────────────────────────────────────────

def main() -> None:
    _g("=" * 58)
    _g("  AI ACCESSIBILITY ASSISTANT — MANUAL LIVE TEST MODE  ")
    _g("=" * 58)

    state = FrameState()

    _b("Initializing production runtime components…")
    camera                = Camera()
    real_detector         = ObjectDetector()
    real_priority         = PriorityEngine()
    real_analyzer         = SceneAnalyzer()
    real_change_det       = SceneChangeDetector()
    real_reasoner         = AccessibilityReasoner()
    real_short_memory     = ShortTermMemory()
    real_speech_manager   = SpeechManager()
    real_frame_throttle   = FrameThrottle(target_fps=10.0)

    instr_detector   = InstrumentedDetector(real_detector, state)
    instr_priority   = InstrumentedPriorityEngine(real_priority, state)
    instr_analyzer   = InstrumentedSceneAnalyzer(real_analyzer, state)
    instr_change_det = InstrumentedSceneChangeDetector(real_change_det, state)
    instr_reasoner   = InstrumentedAccessibilityReasoner(real_reasoner, state)
    instr_memory     = InstrumentedShortMemory(real_short_memory, state)
    instr_speech     = InstrumentedSpeechManager(real_speech_manager, state)
    instr_throttle   = InstrumentedFrameThrottle(real_frame_throttle, state)

    live_service = LiveModeService(
        detector=instr_detector,           # type: ignore[arg-type]
        priority_engine=instr_priority,    # type: ignore[arg-type]
        scene_analyzer=instr_analyzer,     # type: ignore[arg-type]
        scene_change_detector=instr_change_det,  # type: ignore[arg-type]
        accessibility_reasoner=instr_reasoner,   # type: ignore[arg-type]
        short_memory=instr_memory,         # type: ignore[arg-type]
        speech_manager=instr_speech,       # type: ignore[arg-type]
        frame_throttle=instr_throttle,     # type: ignore[arg-type]
    )

    _g("All components initialized successfully.")
    _c("Q=Quit  S=Save frame  T=Memory  C=Clear memory  P=Verbose\n")

    # Yuvarlan FPS: son 30 işlenmiş karedeki gerçek saniye başına oran
    processed_times: deque[float] = deque(maxlen=30)
    pipeline_times: deque[float]  = deque(maxlen=100)
    current_fps  = 0.0
    start_time   = monotonic()
    frame_count  = 0
    processed_count = 0
    prev_dirs: dict[str, str] = {}  # son onaylanan yönler — [SCENE CHANGE] log için

    try:
        while True:
            # 1. Kare yakala
            try:
                frame = camera.capture_frame()
            except Exception as cam_err:
                _r(f"[FATAL] Camera: {cam_err}")
                break

            if frame is None:
                _r("[FATAL] Camera returned None. Exiting.")
                break

            frame_count += 1
            state.reset()  # verbose korunur

            # 2. Pipeline
            t_start = perf_counter()
            try:
                result: LiveModeResult = live_service.process_frame(frame)
            except Exception as pipe_err:
                _r(f"[ERROR] Pipeline: {pipe_err}")
                result = LiveModeResult(
                    spoken=False,
                    scene=StructuredScene(objects=[], total_detected=0),
                )
            pipeline_ms = (perf_counter() - t_start) * 1000
            state.pipeline_ms = pipeline_ms

            # 3. İşlenmiş kare metriklerini güncelle
            if not state.throttled:
                processed_count += 1
                now = monotonic()
                processed_times.append(now)
                pipeline_times.append(pipeline_ms)

                if len(processed_times) >= 2:
                    elapsed = processed_times[-1] - processed_times[0]
                    current_fps = (len(processed_times) - 1) / max(elapsed, 0.001)

                # Kompakt durum satırı (verbose değilse)
                if not state.verbose:
                    _compact_status(state, current_fps)

                # ── Olay bazlı loglar ──────────────────────────────────────────
                # [SCENE CHANGE]
                if state.scene_changed and state.structured_scene:
                    cur_dirs = {
                        o.label: o.direction.value
                        for o in state.structured_scene.objects
                    }
                    for label, direction in cur_dirs.items():
                        prev = prev_dirs.get(label)
                        if prev is None:
                            print()
                            _g(f"[SCENE CHANGE] {label}: (new) → {direction}")
                        elif prev != direction:
                            print()
                            _g(f"[SCENE CHANGE] {label}: {prev} → {direction}")
                    prev_dirs = cur_dirs

                # [TTS]
                if state.spoken_text:
                    print()
                    _g(f"[TTS] {state.spoken_text}")

                # Ayrıntılı mod ek çıktı
                if state.verbose and state.structured_scene:
                    print()
                    _b(f"  Objects: {[(o.label, o.direction.value) for o in state.structured_scene.objects]}")
                    _b(f"  Pipeline: {pipeline_ms:.1f} ms  Detector: {state.detector_ms:.1f} ms")

            # 4. Görüntü önizlemesi
            preview = draw_overlay(frame, state, current_fps, real_short_memory.size)
            cv2.imshow("AI Accessibility Assistant - Live Test", preview)

            # 5. Klavye kısayolları
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                print()
                _y("[USER] Quit.")
                break

            elif key in (ord("s"), ord("S")):
                save_dir = BACKEND_DIR / "temp" / "debug_frames"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"debug_frame_{frame_count}_{int(time.time())}.jpg"
                cv2.imwrite(str(save_path), frame)
                print()
                _g(f"[SAVED] {save_path}")

            elif key in (ord("t"), ord("T")):
                print()
                _c("── ShortMemory ───────────────────────────────────────")
                _c(f"Active entries: {real_short_memory.size}")
                with real_short_memory._lock:
                    cur = monotonic()
                    for entry in real_short_memory._entries:
                        absent = cur - entry.last_seen
                        if entry.last_announced is not None:
                            ann_str = f"announced {cur - entry.last_announced:.1f}s ago"
                        else:
                            ann_str = "not announced yet"
                        _c(
                            f"  {entry.label:12s} | {entry.direction.value:7s} | "
                            f"absent {absent:.2f}s | {ann_str}"
                        )
                _c("──────────────────────────────────────────────────────")

            elif key in (ord("c"), ord("C")):
                real_short_memory.clear()
                prev_dirs.clear()
                print()
                _y("[MEMORY] Cleared.")

            elif key in (ord("p"), ord("P")):
                state.verbose = not state.verbose
                print()
                _b(f"[VERBOSE] {'ENABLED' if state.verbose else 'DISABLED'}")

    finally:
        total = monotonic() - start_time
        camera.release()
        cv2.destroyAllWindows()
        real_speech_manager.stop()

        avg_pipe = sum(pipeline_times) / len(pipeline_times) if pipeline_times else 0.0
        print("\n" + "=" * 58)
        print(f"{C.BOLD}{C.GREEN}  LIVE TEST SUMMARY{C.RESET}")
        print("=" * 58)
        print(f"Captured frames    : {frame_count}")
        print(f"Processed frames   : {processed_count}")
        print(f"Total time         : {total:.1f} s")
        print(f"Avg processed FPS  : {processed_count / max(total, 0.001):.1f}")
        print(f"Avg pipeline       : {avg_pipe:.1f} ms")
        print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
