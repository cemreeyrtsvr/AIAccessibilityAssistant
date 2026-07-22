"""Birim ve entegrasyon testleri: Assistant, Decision, SceneAnalyzer, LLMService, IntentClassifier ve AskModeService."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from assistant.ask_mode_service import AskModeService
from assistant.assistant_service import AssistantService
from assistant.intent_classifier import IntentClassifier, UserIntent
from assistant.scene_analyzer import SceneAnalyzer
from decision.accessibility_filter import AccessibilityFilter
from decision.notification_engine import NotificationEngine
from decision.priority_engine import PriorityEngine, PrioritizedDetection
from decision.danger_detector import DangerAssessment
from decision.rules import AlertSeverity
from llm.client import LLMClient
from llm.llm_service import LLMService
from llm.prompt_builder import PromptBuilder
from memory.short_memory import ShortTermMemory
from models.detection import Detection, Direction
from models.response import Alert, Alerts, AskModeResult
from models.scene import SceneObject, StructuredScene
from ocr.ocr_reader import OCRReader
from vision.detector import ObjectDetector


class TestIntentClassifier(unittest.TestCase):
    """IntentClassifier servisi için birim testleri."""

    def setUp(self) -> None:
        self.classifier = IntentClassifier()

    def test_read_text_detection(self) -> None:
        self.assertEqual(self.classifier.classify("yazıyı oku"), UserIntent.ReadText)
        self.assertEqual(self.classifier.classify("tabelayı okur musun"), UserIntent.ReadText)

    def test_describe_scene_detection(self) -> None:
        self.assertEqual(self.classifier.classify("önümde ne var"), UserIntent.DescribeScene)
        self.assertEqual(self.classifier.classify("etrafımı anlat"), UserIntent.DescribeScene)

    def test_general_question_detection(self) -> None:
        self.assertEqual(self.classifier.classify("saat kaç"), UserIntent.GeneralQuestion)
        self.assertEqual(self.classifier.classify("hava nasıl"), UserIntent.GeneralQuestion)

    def test_unknown_detection(self) -> None:
        self.assertEqual(self.classifier.classify("xyz123"), UserIntent.Unknown)
        self.assertEqual(self.classifier.classify(""), UserIntent.Unknown)

    def test_turkish_normalization(self) -> None:
        self.assertEqual(self.classifier.classify("ÖNÜMDE NE VAR"), UserIntent.DescribeScene)
        self.assertEqual(self.classifier.classify("YAZIYI OKU"), UserIntent.ReadText)


class TestSceneAnalyzer(unittest.TestCase):
    """SceneAnalyzer servisi için birim testleri."""

    def setUp(self) -> None:
        self.analyzer = SceneAnalyzer(max_objects=3, min_priority=20)

    def test_empty_input(self) -> None:
        result = self.analyzer.analyze_scene(None)
        self.assertIsInstance(result, StructuredScene)
        self.assertEqual(len(result.objects), 0)
        self.assertEqual(result.total_detected, 0)

    def test_direction_aware_duplicates(self) -> None:
        person_left = Detection(
            label="person",
            confidence=0.90,
            bbox=(0, 0, 100, 100),
            center_x=50,
            center_y=50,
            direction=Direction.LEFT,
            class_id=0,
        )
        person_right = Detection(
            label="person",
            confidence=0.88,
            bbox=(400, 0, 500, 100),
            center_x=450,
            center_y=50,
            direction=Direction.RIGHT,
            class_id=0,
        )
        scene = self.analyzer.analyze_scene([person_left, person_right])
        self.assertEqual(len(scene.objects), 2)
        directions = {obj.direction for obj in scene.objects}
        self.assertIn(Direction.LEFT, directions)
        self.assertIn(Direction.RIGHT, directions)

    def test_priority_and_max_object_filtering(self) -> None:
        detections = [
            Detection(
                label=f"obj_{i}",
                confidence=0.90,
                bbox=(i * 10, i * 10, 100 + i * 10, 100 + i * 10),
                center_x=50 + i * 10,
                center_y=50 + i * 10,
                direction=Direction.CENTER,
                class_id=i,
            )
            for i in range(5)
        ]
        scene = self.analyzer.analyze_scene(detections)
        self.assertLessEqual(len(scene.objects), 3)


class TestLLMService(unittest.TestCase):
    """LLMService için birim testleri."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=LLMClient)
        self.mock_prompt_builder = MagicMock(spec=PromptBuilder)
        self.llm_service = LLMService(
            llm_client=self.mock_client,
            prompt_builder=self.mock_prompt_builder,
        )

    def test_delegates_to_prompt_builder_and_client(self) -> None:
        scene = StructuredScene(
            objects=[SceneObject(label="car", direction=Direction.CENTER, priority=80)],
            total_detected=1,
        )
        self.mock_prompt_builder.build.return_value = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]
        self.mock_client.generate.return_value = "Test LLM Answer"

        answer = self.llm_service.generate_answer("What is this?", scene=scene, ocr_text="STOP")

        self.assertEqual(answer, "Test LLM Answer")
        self.mock_prompt_builder.build.assert_called_once_with(
            question="What is this?",
            scene=scene,
            ocr_text="STOP",
        )
        self.mock_client.generate.assert_called_once()

    def test_graceful_error_handling(self) -> None:
        self.mock_client.generate.side_effect = Exception("LLM connection error")
        self.mock_prompt_builder.build.return_value = []

        answer = self.llm_service.generate_answer("Hello?")
        self.assertEqual(answer, "")


class TestAskModeServiceIntegration(unittest.TestCase):
    """AskModeService orkestratörü için entegrasyon akış testleri."""

    def setUp(self) -> None:
        self.mock_classifier = MagicMock(spec=IntentClassifier)
        self.mock_scene_analyzer = MagicMock(spec=SceneAnalyzer)
        self.mock_ocr_reader = MagicMock(spec=OCRReader)
        self.mock_llm_service = MagicMock(spec=LLMService)

        self.ask_service = AskModeService(
            intent_classifier=self.mock_classifier,
            scene_analyzer=self.mock_scene_analyzer,
            ocr_reader=self.mock_ocr_reader,
            llm_service=self.mock_llm_service,
        )
        self.dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_describe_scene_flow(self) -> None:
        self.mock_classifier.classify.return_value = UserIntent.DescribeScene
        expected_scene = StructuredScene(
            objects=[SceneObject(label="door", direction=Direction.CENTER)],
            total_detected=1,
        )
        self.mock_scene_analyzer.analyze_scene.return_value = expected_scene

        result = self.ask_service.process_query("önümde ne var", frame=self.dummy_frame)

        self.assertIsInstance(result, AskModeResult)
        self.assertTrue(result.success)
        self.assertEqual(result.intent, UserIntent.DescribeScene)
        self.assertEqual(result.scene, expected_scene)
        self.assertIsNone(result.answer)
        self.mock_ocr_reader.read_text.assert_not_called()
        self.mock_llm_service.generate_answer.assert_not_called()

    def test_read_text_flow(self) -> None:
        self.mock_classifier.classify.return_value = UserIntent.ReadText
        self.mock_ocr_reader.read_text.return_value = "DUR İŞARETİ"

        result = self.ask_service.process_query("yazıyı oku", frame=self.dummy_frame)

        self.assertIsInstance(result, AskModeResult)
        self.assertTrue(result.success)
        self.assertEqual(result.intent, UserIntent.ReadText)
        self.assertEqual(result.ocr_text, "DUR İŞARETİ")
        self.mock_llm_service.generate_answer.assert_not_called()

    def test_general_question_flow(self) -> None:
        self.mock_classifier.classify.return_value = UserIntent.GeneralQuestion
        expected_scene = StructuredScene(
            objects=[SceneObject(label="bus", direction=Direction.RIGHT)],
            total_detected=1,
        )
        self.mock_scene_analyzer.analyze_scene.return_value = expected_scene
        self.mock_ocr_reader.read_text.return_value = "Otobüs Durağı"
        self.mock_llm_service.generate_answer.return_value = "Sağınızda otobüs durağı var."

        result = self.ask_service.process_query("neredeyim", frame=self.dummy_frame)

        self.assertIsInstance(result, AskModeResult)
        self.assertTrue(result.success)
        self.assertEqual(result.intent, UserIntent.GeneralQuestion)
        self.assertEqual(result.answer, "Sağınızda otobüs durağı var.")
        self.mock_llm_service.generate_answer.assert_called_once()


class TestLiveBackendPipelineIntegration(unittest.TestCase):
    """Canlı arka plan işleme hattı entegrasyon testleri."""

    def setUp(self) -> None:
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
