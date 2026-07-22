"""Kullanıcı sesli/metinsel soruları için Soru-Cevap (Ask Mode) orkestrasyon servisi."""

from __future__ import annotations

from typing import Iterable
import numpy as np

from assistant.intent_classifier import IntentClassifier, UserIntent
from assistant.scene_analyzer import SceneAnalyzer
from decision.priority_engine import PrioritizedDetection
from llm.llm_service import LLMService
from models.detection import Detection
from models.response import AskModeResult
from models.scene import StructuredScene
from ocr.ocr_reader import OCRReader


class AskModeService:
    """Kullanıcı sorularını niyet sınıflandırmasına göre servisler arasında orkestre eden sınıf."""

    def __init__(
        self,
        intent_classifier: IntentClassifier | None = None,
        scene_analyzer: SceneAnalyzer | None = None,
        ocr_reader: OCRReader | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.scene_analyzer = scene_analyzer or SceneAnalyzer()
        self.ocr_reader = ocr_reader or OCRReader()
        self.llm_service = llm_service or LLMService()

    def process_query(
        self,
        query: str,
        frame: np.ndarray | None = None,
        items: Iterable[Detection | PrioritizedDetection | object] | None = None,
    ) -> AskModeResult:
        """Kullanıcı sorgusunu analiz eder ve uygun niyet rotasından geçerek AskModeResult döndürür."""
        if not query or not isinstance(query, str):
            return AskModeResult(intent=UserIntent.Unknown, success=False)

        try:
            intent = self.intent_classifier.classify(query)
        except Exception:
            return AskModeResult(intent=UserIntent.Unknown, success=False)

        if intent == UserIntent.DescribeScene:
            try:
                structured_scene = self.scene_analyzer.analyze_scene(items)
                return AskModeResult(
                    intent=intent,
                    success=True,
                    scene=structured_scene,
                )
            except Exception:
                return AskModeResult(
                    intent=intent,
                    success=False,
                    scene=StructuredScene(objects=[], total_detected=0),
                )

        if intent == UserIntent.ReadText:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return AskModeResult(
                    intent=intent,
                    success=False,
                    ocr_text="",
                )
            try:
                extracted_text = self.ocr_reader.read_text(frame)
                return AskModeResult(
                    intent=intent,
                    success=bool(extracted_text),
                    ocr_text=extracted_text,
                )
            except Exception:
                return AskModeResult(
                    intent=intent,
                    success=False,
                    ocr_text="",
                )

        if intent == UserIntent.GeneralQuestion:
            structured_scene: StructuredScene | None = None
            try:
                structured_scene = self.scene_analyzer.analyze_scene(items)
            except Exception:
                structured_scene = StructuredScene(objects=[], total_detected=0)

            extracted_text = ""
            if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                try:
                    extracted_text = self.ocr_reader.read_text(frame)
                except Exception:
                    extracted_text = ""

            try:
                answer = self.llm_service.generate_answer(
                    question=query,
                    scene=structured_scene,
                    ocr_text=extracted_text,
                )
                return AskModeResult(
                    intent=intent,
                    success=bool(answer),
                    scene=structured_scene,
                    ocr_text=extracted_text,
                    answer=answer,
                )
            except Exception:
                return AskModeResult(
                    intent=intent,
                    success=False,
                    scene=structured_scene,
                    ocr_text=extracted_text,
                    answer=None,
                )

        return AskModeResult(intent=UserIntent.Unknown, success=False)
