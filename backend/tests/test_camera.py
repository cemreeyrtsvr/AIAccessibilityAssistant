"""
Camera module integration tests.

Tests:
- Camera initialization
- Camera availability
- Frame capture
- Resource cleanup
"""

from camera.camera import Camera


def test_camera_capture() -> None:
    """Camera should open and capture at least one frame."""

    camera = Camera()

    try:
        assert camera.is_opened(), "Camera failed to open."

        frame = camera.capture_frame()

        assert frame is not None, "No frame was captured."
        assert frame.size > 0, "Captured frame is empty."

    finally:
        camera.release()