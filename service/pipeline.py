"""Uçtan uca pipeline: dosya → OCR → anonimizasyon → kapı → sınıflandırma → rapor.

İlkeler:
  - Ham metin hiçbir zaman loglanmaz, diske yazılmaz, sonuçta yer almaz.
  - Kapı geçilmezse (fail-closed) anonim metin ve sınıflandırma DÖNMEZ; yalnızca
    inceleme için gerekli bilgiler döner.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.anonymizer import ReportAnonymizer

from .anonymization.gate import AnonymizationGate, cross_ocr_candidates
from .backends.llm import make_llm
from .backends.ocr import OCRService
from .classification import Classifier

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    status: str                      # done | needs_review | failed
    file_sha256: str
    filename: str
    ocr: dict = field(default_factory=dict)
    anonymization: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    classification: Optional[dict] = None
    anonymized_text: Optional[str] = None
    error: Optional[str] = None
    timings: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ocr = OCRService(cfg.get("ocr", {}))
        self.llm = make_llm(cfg.get("llm", {}))
        anon_cfg = cfg.get("anonymization", {})
        self.anonymizer = ReportAnonymizer(anon_cfg)
        self.gate = AnonymizationGate(anon_cfg.get("gate", {}), self.llm)
        cls_cfg = cfg.get("classification", {})
        self.classifier = Classifier(cls_cfg, self.llm) if cls_cfg.get("enabled", True) else None
        st = cfg.get("storage", {})
        self.store_anonymized_text = bool(st.get("store_anonymized_text", True))
        # needs_review'da metin: "masked" (varsayılan: kapı bulguları da maskelenir) | "full" | "none"
        self.review_text_mode = str(st.get("review_text_mode", "masked"))
        self.min_chars = int(cfg.get("ocr", {}).get("min_chars", 20))

    def health(self) -> dict:
        return {"ocr": self.ocr.health(), "llm": self.llm.health()}

    def run(self, file_bytes: bytes, filename: str) -> PipelineResult:
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        res = PipelineResult(status="failed", file_sha256=hashlib.sha256(file_bytes).hexdigest(), filename=filename)
        timings = {}

        try:
            # 1. OCR
            t = time.time()
            ocr = self.ocr.extract(file_bytes, suffix)
            timings["ocr"] = round(time.time() - t, 2)
            res.ocr = {
                "engine": ocr.engine,
                "pages": ocr.pages,
                "page_engines": ocr.page_engines,
                "chars": len(ocr.text),
                "alt_engine": ocr.alt_engine,
                "completeness": ocr.completeness,
                "warnings": ocr.warnings,
            }
            if len(ocr.text.strip()) < self.min_chars:
                res.error = "Yeterli metin çıkarılamadı"
                res.timings = timings
                return res

            # 2. Regex/sözlük anonimizasyon
            t = time.time()
            anon_text, report = self.anonymizer.anonymize(ocr.text)
            timings["anonymize"] = round(time.time() - t, 2)
            res.anonymization = {
                "original_chars": report.original_length,
                "anonymized_chars": report.anonymized_length,
                "fields_removed": report.fields_removed,
                "fields_generalized": report.fields_generalized,
            }
            # 2b. Çapraz OCR: ikincil motorun gördüğü PII adayları
            cross = set()
            if ocr.alt_text:
                alt_anon, _ = self.anonymizer.anonymize(ocr.alt_text)
                cross = cross_ocr_candidates(ocr.alt_text, alt_anon)
                res.anonymization["cross_ocr_candidates"] = len(cross)
            del ocr  # ham metin referansını bırak

            # 3. Kapı
            t = time.time()
            gate = self.gate.check(anon_text, cross_candidates=cross)
            timings["gate"] = round(time.time() - t, 2)
            res.gate = gate.as_dict()
            if not gate.passed:
                res.status = "needs_review"
                if self.review_text_mode == "full":
                    res.anonymized_text = anon_text
                elif self.review_text_mode == "masked":
                    # Kapının yakaladığı kalıntılar da maskelenir; bulgu metni yerine yalnızca tür/uzunluk döner
                    masked = anon_text
                    for i, f in enumerate(gate.findings, 1):
                        if f.text:
                            masked = masked.replace(f.text, f"[KAPI_BULGUSU_{i}:{f.type}]")
                    res.anonymized_text = masked
                    res.gate["findings"] = [
                        {"type": f.type, "source": f.source, "chars": len(f.text), "reason": f.reason, "index": i}
                        for i, f in enumerate(gate.findings, 1)
                    ]
                res.timings = timings
                logger.warning("Kapı geçilemedi: %s bulgu, %s hata", len(gate.findings), len(gate.errors))
                return res

            # 4. Sınıflandırma (yalnızca kapıdan geçmiş metin)
            if self.classifier is not None:
                t = time.time()
                cls = self.classifier.classify(anon_text)
                timings["classify"] = round(time.time() - t, 2)
                res.classification = cls.as_dict()

            if self.store_anonymized_text:
                res.anonymized_text = anon_text
            res.status = "done"
            res.timings = timings
            return res

        except Exception as e:  # noqa: BLE001
            # exc_info YOK ve str(e) YOK: ikisi de ham metin/yanıt parçası taşıyabilir
            logger.error("Pipeline hatası (%s): %s", res.file_sha256[:12], type(e).__name__)
            res.error = type(e).__name__
            res.timings = timings
            return res
