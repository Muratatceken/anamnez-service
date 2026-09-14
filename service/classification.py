"""Sınıflandırma — anonimize metin → kanser kategorisi (lokal LLM) + keyword doğrulama."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.models import CancerCategory, ClassificationResult
from src.validator import Validator

from .backends.llm import LLMBackend, extract_json

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "templates" / "classification_prompt.txt"
CATEGORIES_PATH = ROOT / "config" / "categories.json"

SYSTEM = "Sen bir tıbbi patoloji uzmanısın. Yanıtlarını her zaman geçerli JSON formatında ver."


@dataclass
class ClassificationOutput:
    category: str
    confidence: float
    reasoning: str
    histological_type: str | None
    primary_site: str | None
    validated_category: str
    adjusted_confidence: float
    keyword_matches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Classifier:
    def __init__(self, cfg: dict, llm: LLMBackend):
        self.llm = llm
        self.template = PROMPT_PATH.read_text(encoding="utf-8")
        with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
            self.categories: list[str] = json.load(f)["categories"]
        self.validator = Validator(cfg)
        self.valid = {c.value for c in CancerCategory}

    TR_TO_EN = {
        "akciğer": "Lung", "beyin": "Brain", "meme": "Breast", "over": "Ovary", "yumurtalık": "Ovary",
        "uterus": "Uterus", "rahim": "Uterus", "serviks": "Cervix", "karaciğer": "Liver", "böbrek": "Kidney",
        "pankreas": "Pancreas", "mide": "Stomach", "kolon": "Colorectal", "rektum": "Colorectal",
        "prostat": "Prostate", "mesane": "Bladder", "tiroid": "Thyroid", "deri": "Skin", "cilt": "Skin",
        "kemik iliği": "Bone_Marrow", "kemik": "Bone", "kan": "Blood", "lenf": "Lymph_Nodes",
        "özofagus": "Esophagus", "yemek borusu": "Esophagus", "testis": "Testis", "timus": "Thymus",
        "plevra": "Pleura", "yumuşak doku": "Soft_Tissue", "safra": "Bile_Duct", "adrenal": "Adrenal_Gland",
        "böbrek üstü": "Adrenal_Gland", "göz": "Eye", "baş boyun": "Head_and_Neck", "sinir": "Nervous_System",
    }

    def _normalize_category(self, raw: str) -> str:
        """LLM'in yazdığı kategoriyi geçerli enum değerine indirge (Türkçe/İngilizce, boşluk/tire toleranslı)."""
        raw = (raw or "").strip()
        if raw in self.valid:
            return raw
        low = raw.lower().replace(" ", "_").replace("-", "_")
        for c in self.valid:
            if c.lower() == low:
                return c
        for c in self.valid:
            # "lung_cancer" → Lung ; "bilinmeyen_kategori" → "eye" alt dizesine takılmasın (tam token şartı)
            if low and re.search(rf"(?:^|_){re.escape(c.lower())}(?:_|$)", low):
                return c
        # Türkçe organ adı → kategori (uzun anahtarlar önce: "kemik iliği" > "kemik")
        for tr, en in sorted(self.TR_TO_EN.items(), key=lambda kv: -len(kv[0])):
            if tr in raw.lower():
                logger.warning("Kategori düzeltildi: %r → %s", raw, en)
                return en
        logger.warning("Bilinmeyen kategori %r → Other", raw)
        return "Other"

    def classify(self, text: str) -> ClassificationOutput:
        prompt = self.template.format(categories=", ".join(self.categories), report_text=text)
        raw = self.llm.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            json_mode=True,
        )
        data = extract_json(raw)
        conf = data.get("confidence", 0.0)
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.0
        result = ClassificationResult(
            category=self._normalize_category(str(data.get("category", "Other"))),
            confidence=conf,
            reasoning=str(data.get("reasoning", "")),
            histological_type=data.get("histological_type"),
            primary_site=data.get("primary_site"),
        )
        val = self.validator.validate(result, text)
        return ClassificationOutput(
            category=result.category,
            confidence=result.confidence,
            reasoning=result.reasoning,
            histological_type=result.histological_type,
            primary_site=result.primary_site,
            validated_category=val.validated_category,
            adjusted_confidence=round(val.confidence_adjusted, 3),
            keyword_matches=val.keyword_matches,
            warnings=val.warnings,
            model=getattr(self.llm, "model", ""),
        )
