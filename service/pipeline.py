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
from .anonymization.ner import NERAnonymizer
from .backends.llm import make_llm
from .backends.ocr import OCRService
from .classification import Classifier
from .egress import EgressBlocked, EgressGateway
from .report import ReportGenerator, report_to_markdown

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
    report: Optional[dict] = None          # bulut (Claude) raporu: yapılandırılmış + markdown
    egress: Optional[dict] = None          # egress kararı (hash, izin, gerekçe)
    anonymized_text: Optional[str] = None
    error: Optional[str] = None
    timings: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ocr = OCRService(cfg.get("ocr", {}))
        llm_cfg = cfg.get("llm", {}) or {}
        # Lokal LLM opsiyonel (CPU sunucuda yok): enabled=false → yargıç/lokal sınıflandırma kapalı
        self.llm = make_llm(llm_cfg) if llm_cfg.get("enabled", True) else None
        anon_cfg = cfg.get("anonymization", {})
        self.anonymizer = ReportAnonymizer(anon_cfg)
        # Motor: "ner+regex" (varsayılan: NER önce, regex taban) | "regex"
        self.engine = str(anon_cfg.get("engine", "ner+regex"))
        self.ner = NERAnonymizer(anon_cfg.get("ner", {})) if self.engine.startswith("ner") else None
        self.gate = AnonymizationGate(anon_cfg.get("gate", {}), self.llm)
        cls_cfg = cfg.get("classification", {})
        self.classifier = Classifier(cls_cfg, self.llm) if (cls_cfg.get("enabled", True) and self.llm is not None) else None
        cloud_cfg = cfg.get("cloud", {}) or {}
        self.egress = EgressGateway(cloud_cfg, audit_db_path=cfg.get("storage", {}).get("egress_audit_db"))
        self.reporter = ReportGenerator(cloud_cfg, validator_cfg=cls_cfg) if cloud_cfg.get("enabled", False) else None
        st = cfg.get("storage", {})
        self.store_anonymized_text = bool(st.get("store_anonymized_text", True))
        # needs_review'da metin: "masked" (varsayılan: kapı bulguları da maskelenir) | "full" | "none"
        self.review_text_mode = str(st.get("review_text_mode", "masked"))
        self.min_chars = int(cfg.get("ocr", {}).get("min_chars", 20))

    def health(self) -> dict:
        h = {"ocr": self.ocr.health(), "llm": self.llm.health() if self.llm is not None else {"ok": True, "disabled": True}}
        h["cloud"] = {"enabled": self.egress.enabled, "provider": self.egress.provider, "model": self.egress.model}
        if self.ner is not None:
            h["ner"] = self.ner.health()
        return h

    def anonymize_text(self, text: str) -> tuple[str, list[str], set[str]]:
        """NER (varsa) + regex. Döndürür: (anonim metin, silinen alan adları, kapı adayları=NER'in bulduğu ham dizeler)."""
        candidates: set[str] = set()
        fields: list[str] = []
        if self.ner is not None:
            n = self.ner.anonymize(text)
            text, fields, candidates = n.text, list(n.fields_removed), set(n.candidates)
        out, report = self.anonymizer.anonymize(text)
        return out, fields + list(report.fields_removed), candidates

    def run(self, file_bytes: bytes, filename: str, job_id: str = "") -> PipelineResult:
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

            # 2. Anonimizasyon: NER (GLiNER) + regex/sözlük tabanı
            t = time.time()
            anon_text, fields_removed, ner_candidates = self.anonymize_text(ocr.text)
            timings["anonymize"] = round(time.time() - t, 2)
            res.anonymization = {
                "engine": self.engine,
                "original_chars": len(ocr.text),
                "anonymized_chars": len(anon_text),
                "fields_removed": fields_removed,
                "ner_candidates": len(ner_candidates),
            }
            # 2b. Çapraz OCR: ikincil motorun gördüğü PII adayları + NER'in bulduğu ham dizeler
            cross = set(ner_candidates)
            if ocr.alt_text:
                alt_anon, _, alt_ner = self.anonymize_text(ocr.alt_text)
                cross |= cross_ocr_candidates(ocr.alt_text, alt_anon) | alt_ner
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

            # 4a. Bulut raporu (Claude) — egress gateway'den geçmeden HİÇBİR ŞEY çıkmaz
            if self.reporter is not None:
                t = time.time()
                try:
                    d = self.egress.authorize(anon_text, gate, cross, job_id=job_id)
                    res.egress = {"allowed": True, "text_sha256": d.text_sha256, "chars": d.chars,
                                  "host": self.egress.host, "model": self.egress.model}
                except EgressBlocked as e:
                    res.egress = {"allowed": False, "reasons": str(e).split("; ")}
                    res.status = "needs_review"
                    res.timings = timings
                    return res
                rep_ = self.reporter.generate(anon_text)
                rep_["markdown"] = report_to_markdown(rep_)
                res.report = rep_
                res.classification = {
                    "category": rep_["rapor"]["kategori"], "confidence": rep_["rapor"]["guven"],
                    "validated_category": rep_["validated_category"], "adjusted_confidence": rep_["adjusted_confidence"],
                    "keyword_matches": rep_["keyword_matches"], "warnings": rep_["warnings"], "model": rep_["model"],
                    "histological_type": rep_["rapor"].get("histolojik_tip"), "primary_site": rep_["rapor"].get("primer_bolge"),
                    "reasoning": rep_["rapor"].get("gerekce"),
                }
                timings["report"] = round(time.time() - t, 2)
            # 4b. Lokal sınıflandırma (bulut kapalıysa ve lokal LLM varsa)
            elif self.classifier is not None:
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
