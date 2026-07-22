"""Erişilebilirlik uyarılarını doğal, önceliklendirilmiş Türkçe cümlelere dönüştüren servis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.response import Alert, Alerts
    from models.scene import SceneObject, StructuredScene


TURKISH_OBJECT_MAP: dict[str, tuple[str, bool]] = {
    "person": ("kişi", True),
    "car": ("araba", True),
    "bus": ("otobüs", True),
    "motorcycle": ("motosiklet", True),
    "bicycle": ("bisiklet", True),
    "stairs": ("merdiven", False),
    "door": ("kapı", True),
    "traffic light": ("trafik ışığı", True),
    "pole": ("direk", True),
    "chair": ("sandalye", True),
    "bottle": ("şişe", True),
    "cup": ("fincan", True),
    "keyboard": ("klavye", True),
    "table": ("masa", True),
    "bench": ("bank", True),
    "suitcase": ("bavul", True),
    "potted plant": ("saksı bitkisi", True),
    "knife": ("bıçak", True),
    "scissors": ("makas", True),
    "truck": ("kamyon", True),
    "train": ("tren", True),
}

HIGH_PRIORITY_LABELS: set[str] = {
    "person",
    "car",
    "bus",
    "motorcycle",
    "bicycle",
    "knife",
    "scissors",
    "stairs",
    "door",
    "traffic light",
    "pole",
    "truck",
    "train",
}

DIRECTION_TURKISH_MAP: dict[str, str] = {
    "left": "Solunuzda",
    "center": "Önünüzde",
    "right": "Sağınızda",
}


class SentenceBuilder:
    """Yapılandırılmış uyarıları önceliklendirilmiş ve birleştirilmiş doğal Türkçe seslendirme cümlelerine çevirir."""

    def build_sentence(self, alert: Alert | Any) -> str:
        """Tek bir Alert veya SceneObject nesnesini doğal Türkçe cümleye dönüştürür."""
        if alert is None:
            return ""

        raw_label = getattr(alert, "object", getattr(alert, "label", str(alert))).casefold().strip()
        direction_val = getattr(alert, "direction", "")
        direction_str = (
            direction_val.value
            if hasattr(direction_val, "value")
            else str(direction_val)
        ).casefold().strip()

        direction_tr = DIRECTION_TURKISH_MAP.get(direction_str, "Önünüzde")
        object_phrase = self._format_object_phrase(raw_label)

        return f"{direction_tr} {object_phrase} var."

    def build_sentences(self, alerts: Alerts | list[Alert] | Any) -> list[str]:
        """Tüm uyarıları önceliklendirip yönlerine göre birleştirerek doğal Türkçe cümleler üretir."""
        if alerts is None:
            return []

        alert_list: list[Any] = (
            alerts.alerts if hasattr(alerts, "alerts") else list(alerts)
        )
        if not alert_list:
            return []

        sorted_items = sorted(alert_list, key=self._get_item_sort_key, reverse=True)

        grouped_by_dir: dict[str, list[str]] = {}
        for item in sorted_items:
            direction_val = getattr(item, "direction", "")
            direction_str = (
                direction_val.value
                if hasattr(direction_val, "value")
                else str(direction_val)
            ).casefold().strip()
            raw_label = getattr(item, "object", getattr(item, "label", str(item))).casefold().strip()

            direction_tr = DIRECTION_TURKISH_MAP.get(direction_str, "Önünüzde")
            phrase = self._format_object_phrase(raw_label)

            if direction_tr not in grouped_by_dir:
                grouped_by_dir[direction_tr] = []

            if phrase not in grouped_by_dir[direction_tr]:
                grouped_by_dir[direction_tr].append(phrase)

        sentences: list[str] = []
        verbs = ["var", "bulunuyor", "görünüyor"]
        verb_idx = 0

        for direction_tr, phrases in list(grouped_by_dir.items())[:3]:
            if not phrases:
                continue

            if len(phrases) == 1:
                object_str = phrases[0]
            elif len(phrases) == 2:
                object_str = f"{phrases[0]} ve {phrases[1]}"
            else:
                object_str = f"{', '.join(phrases[:-1])} ve {phrases[-1]}"

            verb = verbs[verb_idx % len(verbs)]
            verb_idx += 1

            sentences.append(f"{direction_tr} {object_str} {verb}.")

        return sentences

    def build_text(self, alerts: Alerts | list[Alert] | Any) -> str:
        """Tüm uyarıları TTS servisinin okuyabileceği doğal tek bir metin bloğuna dönüştürür."""
        sentences = self.build_sentences(alerts)
        return " ".join(sentences)

    @staticmethod
    def _format_object_phrase(raw_label: str) -> str:
        """Nesne etiketini Türkçe karşılığına çevirir ve belgisiz zamir ekler."""
        if raw_label in TURKISH_OBJECT_MAP:
            tr_label, use_bir = TURKISH_OBJECT_MAP[raw_label]
            return f"bir {tr_label}" if use_bir else tr_label
        return f"bir {raw_label}" if raw_label else "bir nesne"

    @staticmethod
    def _get_item_sort_key(item: Any) -> tuple[int, float, float]:
        """Nesnenin öncelik sırasını belirler (Tehlike > Öncelik Skoru > Yakınlık)."""
        raw_label = getattr(item, "object", getattr(item, "label", "")).casefold().strip()
        priority = getattr(item, "priority", 0) or 0
        confidence = getattr(item, "confidence", 1.0) or 1.0
        is_high_priority = 1 if raw_label in HIGH_PRIORITY_LABELS else 0

        return (is_high_priority, priority, confidence)
