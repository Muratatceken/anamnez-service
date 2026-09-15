"""Doğrulama Katmanı - LLM sonuçlarını anahtar kelime ve kural tabanlı kontrol."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .models import ClassificationResult, ValidationResult

logger = logging.getLogger(__name__)

_TR = str.maketrans({"İ": "i", "I": "ı"})


def _tr_fold(s: str) -> str:
    return s.translate(_TR).lower()


class Validator:
    """LLM sınıflandırma sonuçlarını doğrular."""

    def __init__(self, config: dict):
        self.config = config
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.require_review_below = config.get("require_human_review_below", 0.4)
        self.use_keywords = config.get("use_keyword_validation", True)

        # Anahtar kelime veritabanını yükle
        categories_path = Path(__file__).parent.parent / "config" / "categories.json"
        with open(categories_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.keyword_hints = data.get("keyword_hints", {})

    def validate(self, result: ClassificationResult, report_text: str) -> ValidationResult:
        """Sınıflandırma sonucunu doğrula."""
        warnings = []
        keyword_matches = []
        adjusted_confidence = result.confidence

        # 1. Anahtar kelime doğrulama
        if self.use_keywords:
            keyword_result = self._keyword_validation(result.category, report_text)
            keyword_matches = keyword_result["matches"]

            if keyword_result["best_category"] and keyword_result["best_category"] != result.category:
                # LLM ve anahtar kelime uyuşmazlığı
                if keyword_result["match_score"] > len(keyword_matches):
                    warnings.append(
                        f"Anahtar kelime analizi '{keyword_result['best_category']}' öneriyor, "
                        f"LLM '{result.category}' dedi. Eşleşen kelimeler: {keyword_result['best_keywords']}"
                    )
                    adjusted_confidence *= 0.7  # Güveni düşür

            # Uyumlu eşleşme güveni ARTIRMAZ (LLM'in kalibrasyonu bozulmasın); yalnızca çelişkide düşürülür

        # 2. Güven eşik kontrolü
        if adjusted_confidence < self.confidence_threshold:
            warnings.append(
                f"Güven skoru düşük ({adjusted_confidence:.2f} < {self.confidence_threshold})"
            )

        if adjusted_confidence < self.require_review_below:
            warnings.append("İnsan incelemesi önerilir")

        # 3. Mantıksal tutarlılık kontrolleri
        consistency_warnings = self._consistency_checks(result, report_text)
        warnings.extend(consistency_warnings)

        # 4. Boş/yetersiz metin kontrolü
        if len(report_text.strip()) < 50:
            warnings.append("Rapor metni çok kısa, sonuç güvenilir olmayabilir")
            adjusted_confidence *= 0.5

        # Sonuç: uyuşmazlık yoksa veya uyuşmazlık düşükse geçerli
        validated_category = result.category
        is_valid = len([w for w in warnings if "İnsan incelemesi" not in w]) == 0

        return ValidationResult(
            is_valid=is_valid,
            original_category=result.category,
            validated_category=validated_category,
            keyword_matches=keyword_matches,
            warnings=warnings,
            confidence_adjusted=round(adjusted_confidence, 3),
        )

    # Siteye özgü olmayan (birçok kanserde geçen) anahtarlar: kategori kanıtı sayılmaz
    GENERIC_KEYWORDS = {"er", "pr", "ki67", "ki-67", "p53", "lenf nodu", "lenf", "skuamöz hücreli karsinom", "adenokarsinom",
                        "karsinom", "metastaz", "metastatik", "tümör", "malign", "grade", "evre", "biyopsi", "her2"}

    @staticmethod
    def _kw_rx(kw: str) -> "re.Pattern":
        parts = [re.escape(w) for w in kw.split()]
        return re.compile(r"(?<![\wçğıöşü])" + r"\s+".join(parts) + r"(?![\wçğıöşü])")

    def _keyword_validation(self, llm_category: str, text: str) -> dict:
        """Metindeki anahtar kelimelere göre kategori öner (kelime sınırlı, Türkçe-duyarsız)."""
        text_lower = _tr_fold(text)
        category_scores = {}

        for category, keywords in self.keyword_hints.items():
            matches = []
            for kw in keywords:
                if kw.lower() in self.GENERIC_KEYWORDS:
                    continue
                if self._kw_rx(_tr_fold(kw)).search(text_lower):
                    matches.append(kw)
            if matches:
                category_scores[category] = {
                    "score": len(matches),
                    "keywords": matches,
                }

        if not category_scores:
            return {"best_category": None, "match_score": 0, "matches": [], "best_keywords": []}

        # En yüksek skorlu kategori
        best = max(category_scores.items(), key=lambda x: x[1]["score"])

        # LLM kategorisi için eşleşmeler
        llm_matches = []
        if llm_category in category_scores:
            llm_matches = category_scores[llm_category]["keywords"]

        return {
            "best_category": best[0],
            "match_score": best[1]["score"],
            "matches": llm_matches or best[1]["keywords"],
            "best_keywords": best[1]["keywords"],
        }

    def _consistency_checks(self, result: ClassificationResult, text: str) -> list[str]:
        """Mantıksal tutarlılık kontrolleri."""
        warnings = []
        text_lower = text.lower()

        # Metastaz kontrolü
        metastasis_keywords = ["metastaz", "metastatik", "sekonder", "yayılım"]
        has_metastasis = any(kw in text_lower for kw in metastasis_keywords)
        if has_metastasis and result.primary_site:
            # Metastaz varsa primer kaynak belirtilmiş mi kontrol et
            if "metastaz" in (result.primary_site or "").lower():
                warnings.append("Primer tümör yeri metastaz bölgesi olarak belirtilmiş olabilir")

        # Benign / malign kontrolü ('negatif' tek başına benign kanıtı DEĞİLDİR: "HER2 negatif")
        benign_keywords = ["benign", "iyi huylu", "selim", "reaktif değişiklik", "reaktif hiperplazi",
                           "malignite saptanmadı", "malignite izlenmedi", "malignite görülmedi", "malignite yok",
                           "malign bulgu saptanmadı", "displazi saptanmadı"]
        malign_keywords = ["malign", "kötü huylu", "karsinom", "sarkom", "lenfoma", "lösemi", "myelom", "seminom",
                           "melanom", "blastom", "adenokarsinom", "karsinoma", "mezotelyoma", "nöroendokrin tümör",
                           "germ hücreli", "polisitemi vera", "myelofibrozis", "gbm", "aml", "kll", "kml"]

        # Olumsuzlanmış ifadeleri ("malignite açısından negatif", "malignite saptanmadı")
        # malign sayma
        import re as _re
        negated = _re.sub(
            r"malign\w*\s+(?:açısından|yönünden|bulgusu)?\s*(?:negatif|saptanmadı|izlenmedi|görülmedi|yok)",
            "", text_lower,
        )
        has_benign = any(kw in text_lower for kw in benign_keywords)
        has_malign = any(kw in negated for kw in malign_keywords)

        if has_benign and not has_malign and result.category != "Other":
            warnings.append("Rapor benign bulgular içeriyor ancak kanser kategorisi atanmış")

        return warnings
