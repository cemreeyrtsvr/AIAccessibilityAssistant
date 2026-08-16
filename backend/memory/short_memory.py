"""Nesne Kimliği tabanlı kısa süreli bellek (ShortTermMemory) - Kararlı durum yönetimi."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any

from decision.rules import AlertSeverity
from models.detection import Direction


@dataclass
class MemoryEntry:
    """Bellekte takip edilen tek bir nesnenin durum bilgileri."""

    tracker_id: int | str | None     # Birincil kimlik (ObjectDetector'dan)
    label: str
    direction: Direction
    bbox: tuple[int, int, int, int]
    first_seen: float
    last_seen: float
    last_announced: float | None = None
    last_announced_direction: Direction | None = None
    last_announced_danger: AlertSeverity | None = None
    consecutive_missed_frames: int = 0
    danger_level: AlertSeverity = AlertSeverity.LOW


class ShortTermMemory:
    """tracker_id birincil anahtar olarak kullanan nesne ömrü ve tekrar-duyuru engelleme servisi."""

    def __init__(
        self,
        max_missing_seconds: float = 1.5,
        max_missing_frames: int = 15,
        expiration_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        if expiration_seconds is not None:
            max_missing_seconds = expiration_seconds

        if max_missing_seconds <= 0:
            raise ValueError("max_missing_seconds sıfırdan büyük olmalıdır.")
        if max_missing_frames < 1:
            raise ValueError("max_missing_frames en az 1 olmalıdır.")

        self.max_missing_seconds = max_missing_seconds
        self.max_missing_frames = max_missing_frames
        self.expiration_seconds = max_missing_seconds
        self._entries: list[MemoryEntry] = []
        self._lock = Lock()

    # ─── Genel API ─────────────────────────────────────────────────────────────

    def should_speak(self, item: Any) -> bool:
        """Nesnenin bellek durumunu günceller, kayıpları temizler ve seslendirilip
        seslendirilmeyeceğini belirler."""
        if item is None:
            return False

        current_time = monotonic()

        with self._lock:
            self._prune_absent_entries(current_time)
            entry = self._match_or_add_entry(item, current_time)
            should_announce = self._evaluate_announcement(entry)

            if should_announce:
                if entry.last_announced_direction is not None and entry.last_announced_direction != entry.direction:
                    # Yön değişimi gerçekleşti!
                    item.is_direction_change = True
                    item.prev_direction = entry.last_announced_direction
                    print(f"[MOVEMENT ANNOUNCE] id={entry.tracker_id} {entry.last_announced_direction.value} -> {entry.direction.value}")
                else:
                    print(f"[MEMORY ANNOUNCE] {entry.label} (id={entry.tracker_id}, {entry.direction.value})")

                entry.last_announced = current_time
                entry.last_announced_direction = entry.direction
                entry.last_announced_danger = entry.danger_level
                return True

        return False

    def update_last_seen(self, items: list[Any]) -> None:
        """Güncel karede tespit edilen nesnelerin last_seen zamanlarını günceller (erken dönme durumunda)."""
        if not items:
            return
        current_time = monotonic()
        with self._lock:
            for item in items:
                entry = self._find_entry(item)
                if entry is not None:
                    entry.last_seen = current_time
                    entry.consecutive_missed_frames = 0
                    print(f"[MEMORY TOUCH] id={entry.tracker_id}")

    def remember_spoken(self, item: Any) -> None:
        """Nesnenin son seslendirilme zamanını ve durumunu yeniler."""
        if item is None:
            return
        current_time = monotonic()
        with self._lock:
            entry = self._match_or_add_entry(item, current_time)
            entry.last_announced = current_time
            entry.last_announced_direction = entry.direction
            entry.last_announced_danger = entry.danger_level

    def has_recent_alert(self, item: Any) -> bool:
        """Nesnenin yakın zamanda duyurulup duyurulmadığını kontrol eder."""
        if item is None:
            return False
        with self._lock:
            entry = self._find_entry(item)
            return entry is not None and entry.last_announced is not None

    def clear(self) -> None:
        """Bellekteki tüm takip edilen nesneleri temizler."""
        with self._lock:
            self._entries.clear()

    @property
    def size(self) -> int:
        """Aktif takip edilen nesne sayısını döndürür."""
        current_time = monotonic()
        with self._lock:
            self._prune_absent_entries(current_time)
            return len(self._entries)

    # ─── İç yardımcılar ────────────────────────────────────────────────────────

    def _evaluate_announcement(self, entry: MemoryEntry) -> bool:
        """Bir duyuru gerekip gerekmediğini değerlendirir."""
        if entry.last_announced is None:
            # a) İlk kez — yeni nesne
            return True

        if entry.last_announced_direction != entry.direction:
            # b) Yön değişmiş (CENTER → LEFT)
            return True

        if (
            entry.last_announced_danger is not None
            and self._severity_rank(entry.danger_level)
            > self._severity_rank(entry.last_announced_danger)
        ):
            # c) Tehlike seviyesi yükselmiş
            return True

        return False

    def _match_or_add_entry(self, item: Any, current_time: float) -> MemoryEntry:
        """Gelen nesneyi tracker_id ile eşleştirir; bulunamazsa IoU fallback kullanır.

        Bulunan kayıt varsa direction ve danger_level güncellenir ama
        yeni kayıt OLUŞTURULMAZ.
        """
        tracker_id = getattr(item, "tracker_id", None)
        label_norm  = self._extract_label(item)
        direction   = getattr(item, "direction", Direction.CENTER)
        bbox        = getattr(item, "bbox", (0, 0, 100, 100))
        danger_level = getattr(
            item, "danger_level", getattr(item, "severity", AlertSeverity.LOW)
        )

        # ─── Geçiş 1: tracker_id birincil eşleştirme ──────────────────────────
        if tracker_id is not None:
            for entry in self._entries:
                if entry.tracker_id == tracker_id:
                    prev_dir = entry.direction
                    entry.last_seen = current_time
                    entry.consecutive_missed_frames = 0
                    entry.bbox = bbox
                    entry.danger_level = danger_level

                    if entry.direction != direction:
                        print(
                            f"[MEMORY UPDATE] id={tracker_id} direction={prev_dir.value} -> {direction.value}"
                        )
                        entry.direction = direction
                    else:
                        print(
                            f"[MEMORY UPDATE] {entry.label} (id={tracker_id}, {direction.value})"
                        )
                    return entry

        # ─── Geçiş 2: IoU tabanlı fallback (tracker_id yok veya yeni iz) ──────
        best_entry: MemoryEntry | None = None
        best_iou = 0.0

        for entry in self._entries:
            # Farklı tracker_id eşleşmelerini engelle
            if entry.tracker_id is not None and tracker_id is not None and entry.tracker_id != tracker_id:
                continue
            if entry.label == label_norm:
                iou = self._calculate_iou(entry.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_entry = entry

        if best_entry is not None and best_iou >= 0.15:
            prev_dir = best_entry.direction
            best_entry.last_seen = current_time
            best_entry.consecutive_missed_frames = 0
            best_entry.bbox = bbox
            best_entry.danger_level = danger_level

            if best_entry.direction != direction:
                print(
                    f"[MEMORY UPDATE] id={best_entry.tracker_id} direction={prev_dir.value} -> {direction.value} (IoU fallback)"
                )
                best_entry.direction = direction
            else:
                print(
                    f"[MEMORY UPDATE] {best_entry.label} "
                    f"(id={best_entry.tracker_id}, {direction.value}, IoU fallback)"
                )

            # tracker_id geldi ama henüz yok — ata
            if tracker_id is not None and best_entry.tracker_id is None:
                best_entry.tracker_id = tracker_id
            return best_entry

        # ─── Geçiş 3: Aynı etiket + yön ile eşleşme (tracker_id yok durumu) ───
        for entry in self._entries:
            # Farklı tracker_id eşleşmelerini engelle
            if entry.tracker_id is not None and tracker_id is not None and entry.tracker_id != tracker_id:
                continue
            if entry.label == label_norm and entry.direction == direction:
                entry.last_seen = current_time
                entry.consecutive_missed_frames = 0
                entry.bbox = bbox
                entry.danger_level = danger_level
                if tracker_id is not None and entry.tracker_id is None:
                    entry.tracker_id = tracker_id
                print(
                    f"[MEMORY UPDATE] {entry.label} (id={entry.tracker_id}, {direction.value})"
                )
                return entry

        # ─── Geçiş 4: Gerçekten yeni nesne ───────────────────────────────────
        new_entry = MemoryEntry(
            tracker_id=tracker_id,
            label=label_norm,
            direction=direction,
            bbox=bbox,
            first_seen=current_time,
            last_seen=current_time,
            danger_level=danger_level,
        )
        self._entries.append(new_entry)
        print(f"[MEMORY ADD] {new_entry.label} (id={tracker_id}, {new_entry.direction.value})")
        return new_entry

    def _find_entry(self, item: Any) -> MemoryEntry | None:
        """Nesneye karşılık gelen bellek kaydını döndürür."""
        tracker_id = getattr(item, "tracker_id", None)
        label_norm = self._extract_label(item)
        direction  = getattr(item, "direction", Direction.CENTER)
        bbox       = getattr(item, "bbox", (0, 0, 100, 100))

        # Geçiş 1: tracker_id eşleşmesi
        if tracker_id is not None:
            for entry in self._entries:
                if entry.tracker_id == tracker_id:
                    return entry

        # Geçiş 2: IoU fallback (tracker_id eşleşmeyenleri atla)
        best_entry = None
        best_iou = 0.0
        for entry in self._entries:
            if entry.tracker_id is not None and tracker_id is not None and entry.tracker_id != tracker_id:
                continue
            if entry.label == label_norm:
                iou = self._calculate_iou(entry.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_entry = entry
        if best_entry is not None and best_iou >= 0.15:
            return best_entry

        # Geçiş 3: Aynı etiket + yön fallback
        for entry in self._entries:
            if entry.tracker_id is not None and tracker_id is not None and entry.tracker_id != tracker_id:
                continue
            if entry.label == label_norm and entry.direction == direction:
                return entry

        return None

    def _prune_absent_entries(self, current_time: float) -> None:
        """Süresi dolmuş veya çok fazla kare kaçırılmış nesneleri temizler."""
        retained: list[MemoryEntry] = []
        for entry in self._entries:
            time_absent = current_time - entry.last_seen
            if (
                time_absent > self.max_missing_seconds
                or entry.consecutive_missed_frames >= self.max_missing_frames
            ):
                print(
                    f"[MEMORY REMOVE] {entry.label} (id={entry.tracker_id}, {entry.direction.value}) "
                    f"- absent: {time_absent:.2f}s, missed: {entry.consecutive_missed_frames}"
                )
            else:
                retained.append(entry)
        self._entries = retained

    @staticmethod
    def _extract_label(item: Any) -> str:
        """Nesne etiketini ayrıştırır ve standartlaştırır."""
        if hasattr(item, "label"):
            raw_label = getattr(item, "label")
        elif hasattr(item, "object"):
            raw_label = getattr(item, "object")
        else:
            raw_label = str(item)
        return " ".join(str(raw_label).casefold().split())

    @staticmethod
    def _calculate_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """İki sınır kutusu arasındaki IoU çakışma oranını hesaplar."""
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        union_area = float(box_a_area + box_b_area - inter_area)
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _severity_rank(severity: AlertSeverity) -> int:
        """Tehlike seviyesini sayısal önceliğe dönüştürür."""
        ranks = {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4,
        }
        return ranks.get(severity, 1)
