"""Erişilebilirlik uyarılarını doğal, önceliklendirilmiş ve Türkçe cümlelere dönüştüren servis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.response import Alert, Alerts
    from models.scene import ReasonedObject, SceneObject, StructuredScene


TURKISH_OBJECT_MAP: dict[str, tuple[str, bool]] = {
    # İnsan ve Ulaşım
    "person": ("kişi", True),
    "car": ("araba", True),
    "bus": ("otobüs", True),
    "truck": ("kamyon", True),
    "motorcycle": ("motosiklet", True),
    "bicycle": ("bisiklet", True),
    "train": ("tren", True),
    "airplane": ("uçak", True),
    "boat": ("tekne", True),
    # Tehlikeli ve Engeller
    "knife": ("bıçak", True),
    "scissors": ("makas", True),
    "stairs": ("merdiven", False),
    "pole": ("direk", True),
    "traffic light": ("trafik ışığı", True),
    "stop sign": ("dur tabelası", True),
    "fire hydrant": ("yangın musluğu", True),
    "bench": ("bank", True),
    # Mobilya ve Ev
    "chair": ("sandalye", True),
    "couch": ("koltuk", True),
    "sofa": ("koltuk", True),
    "bed": ("yatak", True),
    "dining table": ("yemek masası", True),
    "table": ("masa", True),
    "desk": ("çalışma masası", True),
    "toilet": ("tuvalet", True),
    "door": ("kapı", True),
    "window": ("pencere", True),
    # Elektronik ve Kişisel
    "cell phone": ("telefon", True),
    "phone": ("telefon", True),
    "laptop": ("dizüstü bilgisayar", True),
    "computer": ("bilgisayar", True),
    "mouse": ("fare", True),
    "remote": ("kumanda", True),
    "keyboard": ("klavye", True),
    "tv": ("televizyon", True),
    "monitor": ("monitör", True),
    "clock": ("saat", True),
    "book": ("kitap", True),
    "backpack": ("sırt çantası", True),
    "handbag": ("el çantası", True),
    "suitcase": ("bavul", True),
    "umbrella": ("şemsiye", True),
    "bottle": ("şişe", True),
    "cup": ("bardak", True),
    "glass": ("bardak", True),
    "bowl": ("kase", True),
    "fork": ("çatal", True),
    "spoon": ("kaşık", True),
    "potted plant": ("saksı bitkisi", True),
    "plant": ("bitki", True),
    # Hayvanlar
    "dog": ("köpek", True),
    "cat": ("kedi", True),
    "bird": ("kuş", True),
    "horse": ("at", True),
    "sheep": ("koyun", True),
    "cow": ("inek", True),
    "elephant": ("fil", True),
    "bear": ("ayı", True),
    "zebra": ("zebra", True),
    "giraffe": ("zürafa", True),
}

DIRECTION_TURKISH_MAP: dict[str, str] = {
    "left": "Sol tarafınızda",
    "center": "Önünüzde",
    "right": "Sağ tarafınızda",
}

DIRECTION_VARIATIONS: dict[str, list[str]] = {
    "left": ["Sol tarafınızda", "Solunuzda", "Sol tarafta"],
    "center": ["Önünüzde", "Tam karşınızda", "Ön tarafta"],
    "right": ["Sağ tarafınızda", "Sağınızda", "Sağ tarafta"],
}


class SentenceBuilder:
    """Yapılandırılmış uyarıları akıcı, doğal ve Türkçe seslendirme cümlelerine çeviren servis."""

    def build_sentence(self, alert: Alert | SceneObject | ReasonedObject | Any) -> str:
        """Tek bir nesne uyarısını doğal Türkçe cümleye dönüştürür."""
        if alert is None:
            return ""

        raw_label = (
            getattr(alert, "object", getattr(alert, "label", str(alert)))
            .casefold()
            .strip()
        )
        direction_val = getattr(alert, "direction", "")
        direction_str = (
            direction_val.value
            if hasattr(direction_val, "value")
            else str(direction_val)
        ).casefold().strip()

        direction_tr = DIRECTION_TURKISH_MAP.get(direction_str, "Önünüzde")
        phrase = self._format_object_phrase(raw_label)

        # Check if it is direction change:
        is_direction_change = bool(getattr(alert, "is_direction_change", False))
        if is_direction_change:
            prev_dir_val = getattr(alert, "prev_direction", None)
            prev_dir_str = (
                prev_dir_val.value
                if hasattr(prev_dir_val, "value")
                else str(prev_dir_val)
            ).casefold().strip() if prev_dir_val else "center"

            prev_dir_adj = {
                "center": "Önünüzdeki",
                "left": "Solunuzdaki",
                "right": "Sağınızdaki",
            }.get(prev_dir_str, "Önünüzdeki")

            new_dir_verb = {
                "center": "önünüze geçti",
                "left": "sola geçti",
                "right": "sağa geçti",
            }.get(direction_str, "önünüze geçti")

            tr_label = self._get_turkish_label(raw_label)
            return f"{prev_dir_adj} {tr_label} {new_dir_verb}."

        is_warning = bool(getattr(alert, "is_warning", False))
        if is_warning:
            return f"Dikkat! {direction_tr} {phrase} var."

        return f"{direction_tr} {phrase} var."

    def build_sentences(
        self, alerts: Alerts | StructuredScene | list[Any] | Any
    ) -> list[str]:
        """Tüm nesneleri öncelik sırasına koyup yönlerine göre doğal Türkçe cümleler halinde birleştirir."""
        if alerts is None:
            return []

        alert_list: list[Any] = (
            alerts.alerts
            if hasattr(alerts, "alerts")
            else (alerts.objects if hasattr(alerts, "objects") else list(alerts))
        )
        if not alert_list:
            return []

        # Separate direction changes and normal alerts
        dir_change_alerts = [item for item in alert_list if getattr(item, "is_direction_change", False)]
        normal_alerts = [item for item in alert_list if not getattr(item, "is_direction_change", False)]

        sentences: list[str] = []
        for item in dir_change_alerts:
            sent = self.build_sentence(item)
            if sent:
                sentences.append(sent)

        if not normal_alerts:
            return sentences

        # Yönlerine göre grupla
        grouped_by_dir: dict[str, list[dict[str, Any]]] = {}
        for item in normal_alerts:
            direction_val = getattr(item, "direction", "")
            direction_str = (
                direction_val.value
                if hasattr(direction_val, "value")
                else str(direction_val)
            ).casefold().strip()
            raw_label = (
                getattr(item, "object", getattr(item, "label", str(item)))
                .casefold()
                .strip()
            )

            phrase = self._format_object_phrase(raw_label)
            is_warning = bool(getattr(item, "is_warning", False))

            if direction_str not in grouped_by_dir:
                grouped_by_dir[direction_str] = []

            # Aynı nesneyi tekrarlamadan ekle
            if not any(
                d["phrase"] == phrase for d in grouped_by_dir[direction_str]
            ):
                grouped_by_dir[direction_str].append(
                    {"phrase": phrase, "is_warning": is_warning}
                )

        verbs = ["var", "bulunuyor", "yer alıyor"]
        verb_idx = 0

        for direction_str, item_info_list in list(grouped_by_dir.items())[:3]:
            if not item_info_list:
                continue

            phrases = [d["phrase"] for d in item_info_list]
            has_warning = any(d["is_warning"] for d in item_info_list)

            direction_options = DIRECTION_VARIATIONS.get(
                direction_str, ["Önünüzde"]
            )
            dir_tr = direction_options[
                len(sentences) % len(direction_options)
            ]

            if len(phrases) == 1:
                object_str = phrases[0]
            elif len(phrases) == 2:
                object_str = f"{phrases[0]} ve {phrases[1]}"
            else:
                object_str = f"{', '.join(phrases[:-1])} ve {phrases[-1]}"

            verb = verbs[verb_idx % len(verbs)]
            verb_idx += 1

            if has_warning:
                sentences.append(f"Dikkat! {dir_tr} {object_str} {verb}.")
            else:
                sentences.append(f"{dir_tr} {object_str} {verb}.")

        return sentences

    def build_text(
        self, alerts: Alerts | StructuredScene | list[Any] | Any
    ) -> str:
        """Tüm uyarıları akıcı tek bir Türkçe seslendirme metnine birleştirir."""
        sentences = self.build_sentences(alerts)
        return " ".join(sentences)

    @staticmethod
    def _format_object_phrase(raw_label: str) -> str:
        """İngilizce nesne etiketini Türkçe karşılığına çevirir."""
        label_norm = " ".join(raw_label.casefold().split())
        if label_norm in TURKISH_OBJECT_MAP:
            tr_label, use_bir = TURKISH_OBJECT_MAP[label_norm]
            return f"bir {tr_label}" if use_bir else tr_label
        return f"bir {label_norm}" if label_norm else "bir nesne"

    @staticmethod
    def _get_turkish_label(raw_label: str) -> str:
        """İngilizce nesne etiketinin yalın Türkçe karşılığını döndürür (bir takısı olmadan)."""
        label_norm = " ".join(raw_label.casefold().split())
        if label_norm in TURKISH_OBJECT_MAP:
            return TURKISH_OBJECT_MAP[label_norm][0]
        return label_norm if label_norm else "nesne"
