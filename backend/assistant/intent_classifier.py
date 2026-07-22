"""Kullanıcı sesli/metinsel komutlarının amacını belirleyen kural tabanlı sınıflandırıcı."""

from __future__ import annotations

from enum import StrEnum


class UserIntent(StrEnum):
    """Kullanıcı isteklerinin kural tabanlı niyet türleri."""

    DescribeScene = "DescribeScene"
    ReadText = "ReadText"
    GeneralQuestion = "GeneralQuestion"
    Unknown = "Unknown"


class IntentClassifier:
    """Metin girdilerini genişletilebilir kural haritasıyla sınıflandıran servis."""

    # Öncelik sırasına göre kural haritası: ReadText > DescribeScene > GeneralQuestion
    _INTENT_RULES: list[tuple[UserIntent, set[str]]] = [
        (
            UserIntent.ReadText,
            {
                "yazıyı oku",
                "metni oku",
                "tabelayı oku",
                "kitabı oku",
                "yazıları oku",
                "üzerindeki yazıyı oku",
                "yazı",
                "metin",
                "tabela",
                "okur musun",
                "read text",
                "read the text",
                "read",
            },
        ),
        (
            UserIntent.DescribeScene,
            {
                "neler var",
                "sahneyi anlat",
                "çevremi anlat",
                "çevremi betimle",
                "ne görüyorsun",
                "etrafımda ne var",
                "etrafı anlat",
                "etrafımı anlat",
                "önümde ne var",
                "önümde ne görüyorsun",
                "karşımda ne var",
                "sağımda ne var",
                "solumda ne var",
                "önümü tarif et",
                "önümü anlat",
                "betimle",
                "describe scene",
                "describe",
            },
        ),
        (
            UserIntent.GeneralQuestion,
            {
                "saat kaç",
                "hava nasıl",
                "kim bu",
                "bu nedir",
                "ne yapıyorsun",
                "neredeyim",
                "nedir",
                "ne zaman",
                "nasıl",
                "kimdir",
                "nerede",
                "niçin",
                "neden",
                "kaç",
                "hangisi",
                "what is",
                "where am i",
                "who is",
                "how is",
            },
        ),
    ]

    def classify(self, text: str) -> UserIntent:
        """Düz metin komutunu kural tabanlı analiz ederek UserIntent döndürür."""
        if not text or not isinstance(text, str):
            return UserIntent.Unknown

        clean_text = self._normalize_text(text)
        if not clean_text:
            return UserIntent.Unknown

        for intent, keywords in self._INTENT_RULES:
            if any(keyword in clean_text for keyword in keywords):
                return intent

        if "?" in text:
            return UserIntent.GeneralQuestion

        return UserIntent.Unknown

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Türkçe karakterleri ve boşlukları duyarlı şekilde küçültüp düzeltir."""
        tr_normalized = (
            text.replace("İ", "i")
            .replace("I", "ı")
            .replace("Ğ", "ğ")
            .replace("Ü", "ü")
            .replace("Ş", "ş")
            .replace("Ö", "ö")
            .replace("Ç", "ç")
        )
        return " ".join(tr_normalized.casefold().split())
