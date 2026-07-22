"""Karar motorunun algılayıcıdan bağımsız güvenlik kuralları."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AlertSeverity(StrEnum):
    """Uyarının kullanıcıya iletilme aciliyetini belirtir."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DangerRule(BaseModel):
    """Bir nesne sınıfı için temel risk bilgisini tanımlar."""

    base_danger: int = Field(..., ge=0, le=100)
    reason: str = Field(..., min_length=1)


DEFAULT_CATEGORY_IMPORTANCE: dict[str, float] = {
    "person": 1.0,
    "car": 1.0,
    "bus": 1.0,
    "motorcycle": 1.0,
    "bicycle": 1.0,
    "stairs": 1.0,
    "door": 1.0,
    "traffic light": 1.0,
    "pole": 1.0,
    "chair": 0.3,
    "bottle": 0.2,
    "cup": 0.2,
    "keyboard": 0.2,
}

DEFAULT_CATEGORY_FALLBACK_IMPORTANCE: float = 0.5

DEFAULT_DIRECTION_IMPORTANCE: dict[str, float] = {
    "center": 1.0,
    "left": 0.7,
    "right": 0.7,
}


class DecisionSettings(BaseModel):
    """Risk ve öncelik hesaplamalarının merkezi, değiştirilebilir ayarları."""

    max_score: int = Field(default=100, ge=1)
    default_proximity: float = Field(default=0.5, ge=0.0, le=1.0)
    near_object_area_ratio: float = Field(default=0.20, gt=0.0, le=1.0)
    danger_base_weight: float = Field(default=0.65, ge=0.0)
    danger_proximity_weight: float = Field(default=25.0, ge=0.0)
    danger_confidence_weight: float = Field(default=10.0, ge=0.0)
    priority_danger_weight: float = Field(default=0.50, ge=0.0)
    priority_proximity_weight: float = Field(default=15.0, ge=0.0)
    priority_confidence_weight: float = Field(default=10.0, ge=0.0)
    priority_category_weight: float = Field(default=15.0, ge=0.0)
    priority_direction_weight: float = Field(default=10.0, ge=0.0)
    max_distance_meters: float = Field(default=5.0, gt=0.0)
    category_importance: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_IMPORTANCE)
    )
    direction_importance: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_DIRECTION_IMPORTANCE)
    )


# Tanımlı bir güvenlik kuralı olmayan sınıflar bu varsayılanla değerlendirilir.
DEFAULT_FALLBACK_RULE = DangerRule(
    base_danger=10,
    reason="tespit edilen nesne",
)

DEFAULT_DECISION_SETTINGS = DecisionSettings()


# Değerler, sınıfın tek başına değil kullanıcının yakınındaki varlığının
# oluşturabileceği riski temsil eder. Yakınlık ve güven skoru daha sonra eklenir.
DEFAULT_DANGER_RULES: dict[str, DangerRule] = {
    "train": DangerRule(base_danger=95, reason="hareketli ulaşım aracı"),
    "bus": DangerRule(base_danger=85, reason="hareketli ulaşım aracı"),
    "truck": DangerRule(base_danger=85, reason="hareketli ulaşım aracı"),
    "car": DangerRule(base_danger=80, reason="hareketli ulaşım aracı"),
    "motorcycle": DangerRule(base_danger=80, reason="hareketli ulaşım aracı"),
    "bicycle": DangerRule(base_danger=70, reason="hareketli ulaşım aracı"),
    "knife": DangerRule(base_danger=90, reason="kesici nesne"),
    "scissors": DangerRule(base_danger=65, reason="kesici nesne"),
    "chair": DangerRule(base_danger=35, reason="yol üzerindeki engel"),
    "table": DangerRule(base_danger=35, reason="yol üzerindeki engel"),
    "bench": DangerRule(base_danger=30, reason="yol üzerindeki engel"),
    "suitcase": DangerRule(base_danger=35, reason="yol üzerindeki engel"),
    "potted plant": DangerRule(base_danger=25, reason="yol üzerindeki engel"),
}


SEVERITY_THRESHOLDS: tuple[tuple[int, AlertSeverity], ...] = (
    (80, AlertSeverity.CRITICAL),
    (60, AlertSeverity.HIGH),
    (35, AlertSeverity.MEDIUM),
    (0, AlertSeverity.LOW),
)


def normalize_label(label: str) -> str:
    """Farklı algılayıcıların sınıf adlarını ortak kurallarla eşleştirir."""
    return " ".join(label.casefold().split())
