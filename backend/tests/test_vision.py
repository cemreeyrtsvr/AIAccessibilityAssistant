"""Vision modülünü webcam üzerinden elle doğrulamak için bağımsız test."""

import argparse
from time import perf_counter

import cv2

from camera.camera import Camera
from config.settings import WEBCAM_MIRROR_CORRECTION
from vision.detector import ObjectDetector


BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND_COLOR = (0, 0, 0)
GUIDE_COLOR = (0, 200, 255)


def draw_detection(frame, detection) -> None:
    """Bir Detection sonucunun kutu ve özet bilgisini görüntüye çizer."""
    x1, y1, x2, y2 = detection.bbox
    label = (
        f"{detection.label} "
        f"{detection.confidence:.0%} "
        f"{detection.direction.value.upper()}"
    )

    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )
    text_y = max(text_height + baseline + 4, y1)
    cv2.rectangle(
        frame,
        (x1, text_y - text_height - baseline - 4),
        (x1 + text_width + 6, text_y + 2),
        TEXT_BACKGROUND_COLOR,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, text_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_direction_guides(frame) -> None:
    """Düzeltilmiş görüntü koordinat sisteminde yön bölgelerini çizer."""
    frame_height, frame_width = frame.shape[:2]
    first_divider = frame_width // 3
    second_divider = (frame_width * 2) // 3

    cv2.line(frame, (first_divider, 0), (first_divider, frame_height), GUIDE_COLOR, 1)
    cv2.line(frame, (second_divider, 0), (second_divider, frame_height), GUIDE_COLOR, 1)
    cv2.putText(
        frame,
        "LEFT",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        GUIDE_COLOR,
        2,
    )
    cv2.putText(
        frame,
        "CENTER",
        (first_divider + 20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        GUIDE_COLOR,
        2,
    )
    cv2.putText(
        frame,
        "RIGHT",
        (second_divider + 20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        GUIDE_COLOR,
        2,
    )


def main(mirror_correction_required: bool = WEBCAM_MIRROR_CORRECTION) -> None:
    """Webcam karelerini algılar ve Q tuşuna kadar sonuçları gösterir."""
    camera: Camera | None = None

    try:
        camera = Camera()
        detector = ObjectDetector()

        while True:
            frame = camera.capture_frame()
            if frame is None:
                break

            # Aynalı kaynak düzeltmesi, çıkarımdan ve tüm çizimlerden önce uygulanır.
            corrected_frame = cv2.flip(frame, 1) if mirror_correction_required else frame
            start_time = perf_counter()
            detections = detector.detect(corrected_frame)
            frames_per_second = 1 / max(perf_counter() - start_time, 0.0001)

            # Tüm katmanlar yalnızca düzeltilmiş frame üzerinde çizilir.
            draw_direction_guides(corrected_frame)
            for detection in detections:
                draw_detection(corrected_frame, detection)

            cv2.putText(
                corrected_frame,
                f"FPS: {frames_per_second:.1f} | Q: Exit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Vision Module Test", corrected_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision webcam doğrulama testi")
    parser.add_argument(
        "--mirror-correction",
        dest="mirror_correction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Yerel webcam aynalama düzeltmesini geçici olarak değiştirir.",
    )
    arguments = parser.parse_args()
    main(
        mirror_correction_required=(
            WEBCAM_MIRROR_CORRECTION
            if arguments.mirror_correction is None
            else arguments.mirror_correction
        )
    )
