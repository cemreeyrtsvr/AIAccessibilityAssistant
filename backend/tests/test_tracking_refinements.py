"""Unit tests for tracking, duplicate suppression, and memory refinements."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from vision.detector import ObjectDetector
from models.detection import Direction
from memory.short_memory import ShortTermMemory
from models.scene import ReasonedObject
from speech.sentence_builder import SentenceBuilder


def create_mock_box(cls_val, xyxy_list, conf_val):
    box = MagicMock()
    box.cls.item.return_value = cls_val
    mock_tensor = MagicMock()
    mock_tensor.tolist.return_value = xyxy_list
    box.xyxy = [mock_tensor]
    box.conf.item.return_value = conf_val
    return box


def create_mock_result(boxes, names_dict):
    result = MagicMock()
    result.boxes = boxes
    result.names = names_dict
    return result


class TestTrackingRefinements(unittest.TestCase):
    def setUp(self):
        # Prevent actually loading the YOLO model by patching YOLO initialization
        with patch('vision.detector.YOLO') as mock_yolo_class:
            self.mock_yolo_instance = MagicMock()
            mock_yolo_class.return_value = self.mock_yolo_instance
            self.detector = ObjectDetector(model_name="mock_yolo.pt")
            self.detector.model = self.mock_yolo_instance

        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.names = {0: "person", 41: "cup", 42: "wine glass"}
        self.sentence_builder = SentenceBuilder()

    def test_same_person_moves_left_to_right(self):
        """1. Same person moves LEFT -> RIGHT -> Expected: same track ID."""
        box1 = create_mock_box(0, [50.0, 100.0, 130.0, 300.0], 0.9)
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box1], self.names)]
        dets1 = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets1), 1)
        track_id = dets1[0].tracker_id

        box2 = create_mock_box(0, [500.0, 100.0, 580.0, 300.0], 0.9)
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box2], self.names)]
        dets2 = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets2), 1)
        self.assertEqual(dets2[0].tracker_id, track_id)

    def test_same_person_moves_right_to_left(self):
        """2. Same person moves RIGHT -> LEFT -> Expected: same track ID."""
        box1 = create_mock_box(0, [500.0, 100.0, 580.0, 300.0], 0.9)
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box1], self.names)]
        dets1 = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets1), 1)
        track_id = dets1[0].tracker_id

        box2 = create_mock_box(0, [50.0, 100.0, 130.0, 300.0], 0.9)
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box2], self.names)]
        dets2 = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets2), 1)
        self.assertEqual(dets2[0].tracker_id, track_id)

    def test_person_temporarily_disappears(self):
        """3. Person temporarily disappears for a few frames and returns -> Expected: same track ID."""
        box1 = create_mock_box(0, [300.0, 100.0, 380.0, 300.0], 0.9)
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box1], self.names)]
        dets1 = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets1), 1)
        track_id = dets1[0].tracker_id

        self.mock_yolo_instance.predict.return_value = [create_mock_result([], self.names)]
        dets2 = self.detector.detect(self.dummy_frame)
        dets3 = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets2), 0)
        self.assertEqual(len(dets3), 0)

        box4 = create_mock_box(0, [310.0, 100.0, 390.0, 300.0], 0.9)
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box4], self.names)]
        dets4 = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets4), 1)
        self.assertEqual(dets4[0].tracker_id, track_id)

    def test_cup_wine_glass_nearly_identical_bbox(self):
        """4. cup + wine glass with nearly identical bbox -> Expected: ONE physical object."""
        box_cup = create_mock_box(41, [100.0, 100.0, 150.0, 200.0], 0.80)
        box_glass = create_mock_box(42, [101.0, 100.0, 151.0, 201.0], 0.85)
        
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box_cup, box_glass], self.names)]
        dets = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].label, "wine glass")

    def test_two_different_cups(self):
        """5. Two genuinely different cups at different positions -> Expected: TWO objects."""
        box_cup1 = create_mock_box(41, [100.0, 100.0, 150.0, 200.0], 0.8)
        box_cup2 = create_mock_box(41, [400.0, 100.0, 450.0, 200.0], 0.8)
        
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box_cup1, box_cup2], self.names)]
        dets = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets), 2)

    def test_label_flips_between_classes(self):
        """6. Same object changes detector label from cup -> wine glass -> Expected: ONE track."""
        box1 = create_mock_box(41, [100.0, 100.0, 150.0, 200.0], 0.8)
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box1], self.names)]
        dets1 = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets1), 1)
        track_id = dets1[0].tracker_id

        box2 = create_mock_box(42, [101.0, 101.0, 151.0, 201.0], 0.8)
        self.mock_yolo_instance.predict.return_value = [create_mock_result([box2], self.names)]
        dets2 = self.detector.detect(self.dummy_frame)
        self.assertEqual(len(dets2), 1)
        self.assertEqual(dets2[0].tracker_id, track_id)

    def test_short_memory_direction_change_no_new_announcement(self):
        """7. Direction change updates memory entry and marks is_direction_change instead of new entry."""
        memory = ShortTermMemory(max_missing_seconds=2.0)
        
        obj = ReasonedObject(
            label="person",
            direction=Direction.CENTER,
            tracker_id=1,
            confidence=0.9
        )
        
        should1 = memory.should_speak(obj)
        self.assertTrue(should1)
        self.assertFalse(obj.is_direction_change)
        
        obj_same = ReasonedObject(
            label="person",
            direction=Direction.CENTER,
            tracker_id=1,
            confidence=0.9
        )
        should2 = memory.should_speak(obj_same)
        self.assertFalse(should2)
        
        obj_changed = ReasonedObject(
            label="person",
            direction=Direction.LEFT,
            tracker_id=1,
            confidence=0.9
        )
        should3 = memory.should_speak(obj_changed)
        self.assertTrue(should3)
        self.assertTrue(obj_changed.is_direction_change)
        self.assertEqual(obj_changed.prev_direction, Direction.CENTER)

    def test_new_person_announcement_vs_movement_announcement(self):
        """8. New person generates 'var' sentence, existing person movement generates 'geçti' sentence."""
        obj_new = ReasonedObject(
            label="person",
            direction=Direction.LEFT,
            tracker_id=2,
            confidence=0.9,
            is_direction_change=False
        )
        sent_new = self.sentence_builder.build_sentence(obj_new)
        self.assertIn("var", sent_new)
        self.assertNotIn("geçti", sent_new)

        obj_move = ReasonedObject(
            label="person",
            direction=Direction.LEFT,
            tracker_id=1,
            confidence=0.9,
            is_direction_change=True,
            prev_direction=Direction.CENTER
        )
        sent_move = self.sentence_builder.build_sentence(obj_move)
        self.assertIn("sola geçti", sent_move)
        self.assertNotIn("var", sent_move)

    def test_stationary_person_no_repeated_speech_and_no_eviction(self):
        """9. Same continuously visible person doesn't trigger repeated speech and stays in memory."""
        memory = ShortTermMemory(max_missing_seconds=1.5)
        
        obj = ReasonedObject(
            label="person",
            direction=Direction.CENTER,
            tracker_id=1,
            confidence=0.9
        )
        
        # Initial announcement
        self.assertTrue(memory.should_speak(obj))
        
        # Repeated checks with simulated clock increments (using update_last_seen)
        # 10 iterations simulating 0.2s elapsed per iteration (total 2.0s > 1.5s limit)
        for _ in range(10):
            # simulate frame update where object is still seen
            memory.update_last_seen([obj])
            # check that it is NOT announced again
            self.assertFalse(memory.should_speak(obj))
            
        # Verify it is still retained in memory
        self.assertEqual(memory.size, 1)

    def test_new_person_enters_alongside_existing_person(self):
        """10. Multiple independent tracks are kept distinct and do not bleed announcements."""
        memory = ShortTermMemory(max_missing_seconds=2.0)
        
        # Person 1 is on LEFT
        p1 = ReasonedObject(
            label="person",
            direction=Direction.LEFT,
            tracker_id=1,
            confidence=0.9
        )
        self.assertTrue(memory.should_speak(p1))
        
        # Person 2 enters at CENTER
        p2 = ReasonedObject(
            label="person",
            direction=Direction.CENTER,
            tracker_id=2,
            confidence=0.9
        )
        # Should speak (new entry)
        self.assertTrue(memory.should_speak(p2))
        # Ensure it is NOT flagged as direction change of Person 1
        self.assertFalse(p2.is_direction_change)
        
        # Verify size
        self.assertEqual(memory.size, 2)


    def test_instrumented_short_memory_delegates_update_last_seen(self):
        """11. InstrumentedShortMemory wrapper delegates update_last_seen without AttributeError."""
        from tests.manual_live_test import InstrumentedShortMemory, FrameState
        
        real_memory = ShortTermMemory()
        state = FrameState()
        instrumented = InstrumentedShortMemory(real_memory, state)
        
        obj = ReasonedObject(
            label="person",
            direction=Direction.CENTER,
            tracker_id=1,
            confidence=0.9
        )
        
        # Initial speak to create entry
        instrumented.should_speak(obj)
        
        # Test delegation (should run without AttributeError)
        try:
            instrumented.update_last_seen([obj])
        except AttributeError as e:
            self.fail(f"InstrumentedShortMemory failed to delegate update_last_seen: {e}")


if __name__ == '__main__':
    unittest.main()
