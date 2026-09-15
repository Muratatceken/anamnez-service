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
from .egress import _FINAL_RULES, EgressBlocked, EgressGateway
from .report import CloudError, ModelRefusal, ReportGenerator, ReportTruncated, TransientCloudError, redact_report, report_to_markdown

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
    retryable: bool = False                # failed + retryable → istemci daha sonra yeniden gönderebilir
    review_recommended: bool = False       # done ama insan bakışı önerilir (okunabilirlik/güven/belirsizlik)
    review_reasons: list[str] = field(default_factory=list)
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
        self.reporter = ReportGenerator(cloud_cfg, validator_cfg=cls_cfg, host=self.egress.host) if cloud_cfg.get("enabled", False) else None
        self.cloud_attempts = int(cloud_cfg.get("attempts", 3))
        self.cloud_backoff = float(cloud_cfg.get("backoff_seconds", 5))
        self.review_below = float(cls_cfg.get("require_human_review_below", 0.4))
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
                masked = anon_text
                for i, f in enumerate(gate.findings, 1):
                    if f.text:
                        masked = masked.replace(f.text, f"[KAPI_BULGUSU_{i}:{f.type}]")
                if self.review_text_mode == "full":
                    res.anonymized_text = anon_text          # bulgu metni de kalır (yalnızca kapalı devre inceleme UI)
                elif self.review_text_mode == "masked":
                    res.anonymized_text = masked
                if self.review_text_mode != "full":
                    # Bulgu METNİ yalnızca 'full' modunda; diğerlerinde tür/uzunluk
                    res.gate["findings"] = [
                        {"type": f.type, "source": f.source, "chars": len(f.text), "reason": f.reason, "index": i}
                        for i, f in enumerate(gate.findings, 1)
                    ]
                res.timings = timings
                logger.warning("Kapı geçilemedi: %s bulgu, %s hata", len(gate.findings), len(gate.errors))
                return res

            # 4a. Bulut raporu (Claude) — egress gateway'den geçmeden HİÇBİR ŞEY çıkmaz;
            #     her deneme ayrı egress onayı + denetim kaydı alır (SDK retry kapalı)
            if self.reporter is not None:
                t = time.time()
                rep_ = None
                for attempt in range(1, self.cloud_attempts + 1):
                    try:
                        d = self.egress.authorize(anon_text, gate, cross, job_id=job_id)
                        res.egress = {"allowed": True, "text_sha256": d.text_sha256, "chars": d.chars,
                                      "host": self.egress.host, "model": self.egress.model, "attempts": attempt}
                    except EgressBlocked as e:
                        res.egress = {"allowed": False, "reasons": str(e).split("; ")}
                        res.status = "needs_review"
                        res.timings = timings
                        return res
                    try:
                        rep_ = self.reporter.generate(anon_text)
                        break
                    except TransientCloudError as e:
                        logger.warning("Bulut geçici hata (%s, deneme %d/%d): %s", job_id[:8], attempt, self.cloud_attempts, e)
                        if attempt == self.cloud_attempts:
                            res.error = f"TransientCloudError:{e}"
                            res.retryable = True
                            res.timings = timings
                            return res
                        time.sleep(self.cloud_backoff * attempt)
                    except ModelRefusal as e:
                        res.status = "needs_review"
                        res.review_reasons.append(f"model reddetti ({e.category})")
                        res.timings = timings
                        return res
                    except ReportTruncated as e:
                        res.error = f"ReportTruncated:{e}"
                        res.timings = timings
                        return res
                    except CloudError as e:
                        res.error = f"CloudError:{e}"
                        res.timings = timings
                        return res
                # Model yanıtı da PII taramasından geçer (serbest metne kalıntı sızmasın)
                hits = redact_report(rep_["rapor"], cross, _FINAL_RULES)
                if hits:
                    rep_["warnings"] = list(rep_.get("warnings", [])) + [f"rapor alanı maskelendi: {', '.join(hits)}"]
                    res.review_recommended = True
                    res.review_reasons.append("rapor çıktısında PII benzeri kalıntı maskelendi")
                rep_["markdown"] = report_to_markdown(rep_)
                res.report = rep_
                rap = rep_["rapor"]
                if rap.get("okunabilirlik") == "kotu":
                    res.review_recommended = True; res.review_reasons.append("okunabilirlik: kötü")
                if rap.get("malignite_durumu") == "belirsiz":
                    res.review_recommended = True; res.review_reasons.append("malignite belirsiz")
                if min(rap.get("guven", 0.0), rep_["adjusted_confidence"]) < self.review_below:
                    res.review_recommended = True; res.review_reasons.append("düşük güven")
                if any("İnsan incelemesi" in w for w in rep_.get("warnings", [])):
                    res.review_recommended = True; res.review_reasons.append("doğrulama uyarısı")
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
