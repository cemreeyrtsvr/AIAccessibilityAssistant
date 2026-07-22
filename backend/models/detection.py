"""Görüntü algılama sonuçları için algılayıcıdan bağımsız veri modelleri."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Direction(StrEnum):
    """Nesnenin görüntü karesindeki yatay konumunu belirtir."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class Detection(BaseModel):
    """Bir görüntü karesinde tespit edilen nesnenin standart temsili.

    Bu model, kullanılan algılama kütüphanesinin yerel sonuç nesnelerini
    uygulamanın geri kalanına geçirmez. Böylece YOLO yerine farklı bir model
    kullanıldığında tüketen katmanların değişmesine gerek kalmaz.
    """

    label: str = Field(..., min_length=1, description="İnsan tarafından okunabilir sınıf adı.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Algılama güven skoru."
    )
    bbox: tuple[int, int, int, int] = Field(
        ..., description="Giriş karesine göre (x1, y1, x2, y2) sınır kutusu."
    )
    center_x: int = Field(..., ge=0, description="Nesne merkezinin yatay piksel konumu.")
    center_y: int = Field(..., ge=0, description="Nesne merkezinin dikey piksel konumu.")
    direction: Direction = Field(..., description="Nesnenin yatay yön bölgesi.")
    distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Varsa nesneye tahminî uzaklık (metre).",
    )
    class_id: int | str = Field(
        ..., description="Algılayıcının sınıf kimliği; sayısal veya metin olabilir."
    )
    tracker_id: int | str | None = Field(
        default=None, description="Varsa nesne takip kimliği."
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Algılayıcıya veya uygulamaya özgü ek, serileştirilebilir bilgiler.",
    )
