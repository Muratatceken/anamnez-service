"""Bulut raporu + egress gateway testleri — sahte Anthropic istemcisi (ağ yok)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from service.anonymization.gate import AnonymizationGate, GateResult
from service.egress import EgressBlocked, EgressGateway
from service.pipeline import Pipeline
from service.report import DoctorReport, ReportGenerator, report_to_markdown

ANON = """[KURUM_SILINDI]
Hasta [HASTA_ADI_SILINDI] [YAS_ARALIGI: 55-64]
TANI
Sol akciğer alt lob, skuamöz hücreli karsinom. TTF-1 negatif, p40 pozitif. PDL-1 %60.
"""


class FakeClient:
    """client.messages.parse taklidi; gönderilen metni kaydeder."""

    def __init__(self, report: DoctorReport, stop_reason="end_turn"):
        self.report = report
        self.stop_reason = stop_reason
        self.sent = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.sent.append(kw)
        return SimpleNamespace(parsed_output=self.report, stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=500, output_tokens=200, cache_read_input_tokens=0),
                               _request_id="req_test", stop_details=None)


def sample_report(cat="Lung"):
    return DoctorReport(kategori=cat, guven=0.9, gerekce="Akciğer SCC", histolojik_tip="Skuamöz hücreli karsinom",
                        primer_bolge="Sol akciğer alt lob", evre_grade=None,
                        belirtecler=[{"ad": "TTF-1", "deger": "negatif"}, {"ad": "p40", "deger": "pozitif"}],
                        onemli_bulgular=["Sol alt lobda kitle"], tedavi_ve_plan=[], ozet="Akciğer skuamöz hücreli karsinom.",
                        belirsizlikler=[], okunabilirlik="iyi")


# ── Rapor ───────────────────────────────────────────────────────────────
def test_report_generator_uses_structured_output_and_validates():
    fake = FakeClient(sample_report())
    gen = ReportGenerator({"model": "claude-opus-5", "effort": "high"}, client=fake)
    r = gen.generate(ANON)
    assert r["validated_category"] == "Lung" and r["rapor"]["histolojik_tip"].startswith("Skuamöz")
    kw = fake.sent[0]
    assert kw["model"] == "claude-opus-5" and kw["output_format"] is DoctorReport
    assert kw["output_config"]["effort"] == "high"
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert ANON in kw["messages"][0]["content"]
    md = report_to_markdown(r)
    assert "Kategori:" in md and "TTF-1: negatif" in md


def test_report_refusal_raises():
    gen = ReportGenerator({}, client=FakeClient(sample_report(), stop_reason="refusal"))
    with pytest.raises(RuntimeError):
        gen.generate(ANON)


def test_report_schema_rejects_unknown_category():
    with pytest.raises(Exception):
        DoctorReport(kategori="Elma", guven=0.5, gerekce="x", ozet="y", okunabilirlik="iyi")


# ── Egress ──────────────────────────────────────────────────────────────
def test_egress_blocks_when_gate_failed_or_candidate_present(tmp_path):
    eg = EgressGateway({"enabled": True, "provider": "anthropic"}, audit_db_path=str(tmp_path / "a.db"))
    ok = GateResult(passed=True)
    assert eg.decide(ANON, ok, set()).allowed
    assert not eg.decide(ANON, GateResult(passed=False), set()).allowed
    assert not eg.decide(ANON + "\nMehmet Yılmaz", ok, {"Mehmet Yılmaz"}).allowed
    assert not eg.decide(ANON + "\nTC 12345678901", ok, set()).allowed
    assert not eg.decide(ANON + "\n05551234567", ok, set()).allowed
    with pytest.raises(EgressBlocked):
        eg.authorize(ANON, GateResult(passed=False), set(), job_id="j1")
    import sqlite3
    rows = sqlite3.connect(str(tmp_path / "a.db")).execute("select allowed, reasons, chars from egress_audit").fetchall()
    assert rows == [(0, "kapı geçilmedi", len(ANON))]   # metin değil, yalnızca metadata


def test_egress_disabled_blocks():
    assert not EgressGateway({"enabled": False}).decide(ANON, GateResult(passed=True), set()).allowed


def test_egress_rejects_unknown_provider():
    with pytest.raises(ValueError):
        EgressGateway({"enabled": True, "provider": "openai"})


# ── Pipeline: bulut yolu ────────────────────────────────────────────────
def _cloud_pipeline(tmp_path, fake, engine="regex"):
    cfg = {
        "ocr": {"backend": "tesseract", "min_chars": 20},
        "llm": {"enabled": False},
        "anonymization": {"engine": engine, "gate": {"enabled": True, "heuristics": True, "llm_judge": False, "fail_closed": True}},
        "classification": {"enabled": True, "confidence_threshold": 0.6, "require_human_review_below": 0.4},
        "cloud": {"enabled": True, "provider": "anthropic", "model": "claude-opus-5"},
        "storage": {"egress_audit_db": str(tmp_path / "egress.db")},
    }
    p = Pipeline(cfg)
    p.reporter = ReportGenerator(cfg["cloud"], client=fake)
    return p


def test_pipeline_cloud_path_sends_only_anonymized_text(tmp_path):
    fake = FakeClient(sample_report())
    p = _cloud_pipeline(tmp_path, fake)
    raw = "Hasta Adı Soyadı : MEHMET YILMAZ\nTC Kimlik No : 12345678901\nTANI\nSol akciğer alt lob, skuamöz hücreli karsinom. TTF-1 negatif, p40 pozitif.\n"
    res = p.run(raw.encode(), "r.txt", job_id="job1")
    assert res.status == "done", res.as_dict()
    assert res.egress["allowed"] and res.report["validated_category"] == "Lung"
    sent = fake.sent[0]["messages"][0]["content"]
    assert "MEHMET" not in sent and "12345678901" not in sent and "skuamöz" in sent
    assert "MEHMET" not in json.dumps(res.as_dict(), ensure_ascii=False)


def test_pipeline_cloud_path_blocked_when_gate_fails(tmp_path):
    fake = FakeClient(sample_report())
    p = _cloud_pipeline(tmp_path, fake)
    # kapıyı zorla düşür: heuristics 'TC' bulacak — regex kaçırsın diye etiketsiz TC'yi düz metne koy
    p.anonymizer = type("NoOp", (), {"anonymize": lambda self, t: (t, type("R", (), {"fields_removed": [], "original_length": len(t), "anonymized_length": len(t), "fields_generalized": []})())})()
    res = p.run(b"TANI karsinom\nkimlik 12345678901 sakli\nmetin uzun olsun diye biraz daha", "r.txt", job_id="job2")
    assert res.status == "needs_review"
    assert fake.sent == []   # Claude'a hiçbir şey gitmedi


@pytest.mark.slow
def test_pipeline_ner_engine_end_to_end(tmp_path):
    """GLiNER gerçek model (HF cache'te olmalı; yoksa atla)."""
    try:
        from service.anonymization.ner import NERAnonymizer
        NERAnonymizer()._load()
    except Exception:
        pytest.skip("GLiNER modeli yok")
    fake = FakeClient(sample_report())
    p = _cloud_pipeline(tmp_path, fake, engine="ner+regex")
    raw = "AVI SOXAOT: BURAK ERDOĞDU\nTC KİMLİK no 94323194875\nSamsun'dan sevk.\nTANI\nAML M4, NPM1 pozitif, FLT3 negatif. Rydapt eklendi.\n"
    res = p.run(raw.encode(), "r.txt", job_id="job3")
    sent = fake.sent[0]["messages"][0]["content"] if fake.sent else ""
    assert res.status == "done", res.as_dict()
    assert "ERDOĞDU" not in sent and "Samsun" not in sent and "NPM1" in sent
