"""End-to-end integration tests for the live backend pipeline."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from assistant.assistant_service import AssistantService
from decision.accessibility_filter import AccessibilityFilter
from decision.notification_engine import NotificationEngine
from decision.priority_engine import PriorityEngine
from memory.short_memory import ShortTermMemory
from models.detection import Detection, Direction
from models.response import Alert, Alerts
from vision.detector import ObjectDetector


class TestLiveBackendPipelineIntegration(unittest.TestCase):
    """İntegrasyon testi: Canlı arka plan işleme hattını uçtan uca doğrular."""

    def setUp(self) -> None:
        """Test ortamını ve bağımlılıkları hazırlar."""
        self.mock_detector = MagicMock(spec=ObjectDetector)
        self.short_memory = ShortTermMemory(expiration_seconds=5.0)
        self.priority_engine = PriorityEngine()
        self.accessibility_filter = AccessibilityFilter()
        self.notification_engine = NotificationEngine(
            priority_engine=self.priority_engine,
            accessibility_filter=self.accessibility_filter,
            short_memory=self.short_memory,
            max_alerts=3,
        )
        self.service = AssistantService(
            detector=self.mock_detector,
            notification_engine=self.notification_engine,
            short_memory=self.short_memory,
        )
        self.dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_pipeline_end_to_end(self) -> None:
        """Tüm hattın mock algılamalarla eksiksiz çalıştığını doğrular."""
        detection_car = Detection(
            label="car",
            confidence=0.92,
            bbox=(300, 200, 900, 650),
            center_x=600,
            center_y=425,
            direction=Direction.CENTER,
            distance=1.2,
            class_id=2,
        )
        detection_person = Detection(
            label="person",
            confidence=0.85,
            bbox=(50, 100, 250, 500),
            center_x=150,
            center_y=300,
            direction=Direction.LEFT,
            distance=3.0,
            class_id=0,
        )
        detection_low_conf = Detection(
            label="bottle",
            confidence=0.15,
            bbox=(500, 500, 550, 600),
            center_x=525,
            center_y=550,
            direction=Direction.CENTER,
            distance=4.0,
            class_id=39,
        )

        self.mock_detector.detect.return_value = [
            detection_car,
            detection_person,
            detection_low_conf,
        ]

        alerts = self.service.process_frame(self.dummy_frame)

        self.assertIsInstance(alerts, Alerts)
        self.assertGreater(len(alerts), 0)

        object_labels = [alert.object for alert in alerts]
        self.assertIn("car", object_labels)
        self.assertIn("person", object_labels)
        self.assertNotIn("bottle", object_labels)

        self.assertGreaterEqual(alerts[0].priority, alerts[1].priority)
        self.assertEqual(alerts[0].object, "car")
        self.assertIsInstance(alerts[0], Alert)

        second_alerts = self.service.process_frame(self.dummy_frame)
        self.assertIsInstance(second_alerts, Alerts)
        self.assertEqual(len(second_alerts), 0)

    def test_max_alerts_respected(self) -> None:
        """Üretilen uyarı sayısının max_alerts sınırını aşmadığını doğrular."""
        detections = [
            Detection(
                label=label,
                confidence=0.85,
                bbox=(10 * i, 10 * i, 100 + 10 * i, 100 + 10 * i),
                center_x=50 + 10 * i,
                center_y=50 + 10 * i,
                direction=Direction.CENTER,
                distance=2.0,
                class_id=i,
            )
            for i, label in enumerate(["car", "person", "bus", "motorcycle", "bicycle"])
        ]
        self.mock_detector.detect.return_value = detections

        alerts = self.service.process_frame(self.dummy_frame)

        self.assertIsInstance(alerts, Alerts)
        self.assertEqual(len(alerts), self.notification_engine.max_alerts)

    def test_dangerous_objects_receive_higher_priority(self) -> None:
        """Tehlikeli nesnelerin tehlikesiz nesnelere göre daha yüksek öncelik aldığını doğrular."""
        dangerous_item = Detection(
            label="knife",
            confidence=0.80,
            bbox=(100, 100, 300, 300),
            center_x=200,
            center_y=200,
            direction=Direction.CENTER,
            distance=2.0,
            class_id=43,
        )
        non_dangerous_item = Detection(
            label="chair",
            confidence=0.80,
            bbox=(100, 100, 300, 300),
            center_x=200,
            center_y=200,
            direction=Direction.CENTER,
            distance=2.0,
            class_id=56,
        )
        self.mock_detector.detect.return_value = [dangerous_item, non_dangerous_item]

        alerts = self.service.process_frame(self.dummy_frame)

        self.assertIsInstance(alerts, Alerts)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].object, "knife")


if __name__ == "__main__":
    unittest.main()
