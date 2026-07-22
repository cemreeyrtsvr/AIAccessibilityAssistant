"""Yapılandırılmış erişilebilirlik sahne modelleri."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from decision.rules import AlertSeverity
from models.detection import Direction


class SemanticCategory(StrEnum):
    HUMAN = "human"
    VEHICLE = "vehicle"
    DANGEROUS_OBJECT = "dangerous_object"
    NAVIGATION_LANDMARK = "navigation_landmark"
    FURNITURE = "furniture"
    BACKGROUND_OBJECT = "background_object"


class SceneObject(BaseModel):
    """Yapılandırılmış sahne analizindeki tek nesne temsili."""

    label: str = Field(..., min_length=1)
    direction: Direction
    distance: float | None = Field(default=None, ge=0.0)
    priority: int | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class StructuredScene(BaseModel):
    """Filtrelenmiş ve yapısı düzenlenmiş sahne temsili kapsayıcısı."""

    objects: list[SceneObject] = Field(default_factory=list)
    total_detected: int = Field(default=0, ge=0)


class ReasonedObject(BaseModel):
    """Erişilebilirlik akıl yürütmesinden geçmiş nesne modeli."""

    label: str = Field(..., min_length=1)
    direction: Direction
    distance: float | None = Field(default=None, ge=0.0)
    priority: int | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    category: SemanticCategory = SemanticCategory.BACKGROUND_OBJECT
    danger_level: AlertSeverity = AlertSeverity.LOW
    is_warning: bool = False
    is_landmark: bool = False


class WarningInfo(BaseModel):
    """Gelişmiş erişilebilirlik uyarısı detay modeli."""

    object_label: str = Field(..., min_length=1)
    direction: Direction
    danger_level: AlertSeverity = AlertSeverity.HIGH
    distance: float | None = Field(default=None, ge=0.0)
    reason: str | None = None


class NavigationHint(BaseModel):
    """Gelişmiş yönlendirme ve ipucu detay modeli."""

    object_label: str = Field(..., min_length=1)
    direction: Direction
    distance: float | None = Field(default=None, ge=0.0)
    landmark_type: str | None = None


class ReasonedScene(BaseModel):
    """AccessibilityReasoner tarafından kararlaştırılan erişilebilirlik sahnesi."""

    objects_to_announce: list[ReasonedObject] = Field(default_factory=list)
    max_danger_level: AlertSeverity = AlertSeverity.LOW
    warnings: list[WarningInfo] = Field(default_factory=list)
    navigation_hints: list[NavigationHint] = Field(default_factory=list)
