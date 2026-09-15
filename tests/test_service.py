"""Servis testleri — sahte LLM backend ile (Ollama/GPU gerekmez)."""

import io
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from service.anonymization.gate import AnonymizationGate
from service.classification import Classifier
from service.pipeline import Pipeline

SAMPLE = """Hasta Adı Soyadı : MEHMET YILMAZ
TC Kimlik No : 12345678901
Yaş : 63
İsteyen Doktor : Prof. Dr. AYŞE KAYA
TANI
Sol akciğer alt lob, skuamöz hücreli karsinom. TTF-1 negatif, p40 pozitif.
"""


class FakeLLM:
    """Yargıç ve sınıflandırıcı için betikli yanıtlar."""

    name = "fake"
    model = "fake-model"

    def __init__(self, judge_findings=None, category="Lung", confidence=0.9, fail=False):
        self.judge_findings = judge_findings or []
        self.category = category
        self.confidence = confidence
        self.fail = fail
        self.calls = []

    def chat(self, messages, *, json_mode=False, images=None, max_tokens=None):
        self.calls.append(messages[-1]["content"][:60])
        if self.fail:
            raise RuntimeError("LLM down")
        user = messages[-1]["content"]
        if "<<<" in user and ">>>" in user:  # yargıç prompt işareti
            return json.dumps({"findings": self.judge_findings})
        return json.dumps({
            "category": self.category, "confidence": self.confidence,
            "reasoning": "test", "histological_type": "SCC", "primary_site": "akciğer",
        })

    def health(self):
        return {"ok": True, "backend": self.name, "model": self.model}


def make_pipeline(llm, **overrides):
    cfg = {
        "ocr": {"backend": "tesseract", "min_chars": 20},
        "llm": {"backend": "ollama", "model": "x"},
        "anonymization": {"engine": "regex", "gate": {"enabled": True, "llm_judge": True, "fail_closed": True}},
        "classification": {"enabled": True, "confidence_threshold": 0.6, "require_human_review_below": 0.4},
        "storage": {"store_review_text": True},
    }
    for k, v in overrides.items():
        cfg[k].update(v)
    p = Pipeline(cfg)
    p.llm = llm
    p.gate = AnonymizationGate(cfg["anonymization"]["gate"], llm)
    p.classifier = Classifier(cfg["classification"], llm)
    return p


# ── Kapı ────────────────────────────────────────────────────────────────
def test_gate_passes_when_no_findings():
    g = AnonymizationGate({"enabled": True, "llm_judge": True}, FakeLLM())
    assert g.check("[HASTA_ADI_SILINDI] karsinom").passed


def test_gate_blocks_on_real_finding():
    llm = FakeLLM(judge_findings=[{"text": "ERDEM", "type": "person", "reason": "soyad"}])
    r = AnonymizationGate({"enabled": True, "llm_judge": True}, llm).check("[DOKTOR_SILINDI] ERDEM\nTANI: karsinom")
    assert not r.passed and r.findings[0].text == "ERDEM"


def test_gate_ignores_hallucinated_or_masked_findings():
    llm = FakeLLM(judge_findings=[
        {"text": "[HASTA_ADI_SILINDI]", "type": "person"},   # zaten maskeli
        {"text": "AHMET", "type": "person"},                  # metinde yok
    ])
    assert AnonymizationGate({"enabled": True}, llm).check("[HASTA_ADI_SILINDI] karsinom").passed


def test_gate_fail_closed_on_layer_error():
    r = AnonymizationGate({"enabled": True, "fail_closed": True}, FakeLLM(fail=True)).check("metin")
    assert not r.passed and r.errors


# ── Pipeline ────────────────────────────────────────────────────────────
def test_pipeline_done_returns_anonymized_text_and_classification():
    llm = FakeLLM()
    res = make_pipeline(llm).run(SAMPLE.encode(), "rapor.txt")
    assert res.status == "done"
    assert "MEHMET" not in res.anonymized_text and "12345678901" not in res.anonymized_text
    assert "skuamöz hücreli karsinom" in res.anonymized_text
    assert res.classification["validated_category"] == "Lung"
    # yargıç ham metni değil, anonim metni görmeli
    assert all("MEHMET" not in c for c in llm.calls)


def test_pipeline_needs_review_withholds_classification():
    llm = FakeLLM(judge_findings=[{"text": "karsinom", "type": "person", "reason": "test"}])
    res = make_pipeline(llm).run(SAMPLE.encode(), "rapor.txt")
    assert res.status == "needs_review"
    assert res.classification is None
    assert res.gate["findings"]


def test_pipeline_result_never_contains_raw_text():
    res = make_pipeline(FakeLLM()).run(SAMPLE.encode(), "rapor.txt")
    dumped = json.dumps(res.as_dict(), ensure_ascii=False)
    for pii in ("MEHMET", "YILMAZ", "12345678901", "AYŞE KAYA"):
        assert pii not in dumped


def test_pipeline_too_short_fails_cleanly():
    res = make_pipeline(FakeLLM()).run(b"kisa", "a.txt")
    assert res.status == "failed" and "Yeterli metin" in res.error


# ── API ─────────────────────────────────────────────────────────────────
@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from service import api

    monkeypatch.setenv("ANAMNEZ_STORAGE__DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("ANAMNEZ_SERVER__API_KEY", "secret")
    monkeypatch.setenv("ANAMNEZ_OCR__BACKEND", "tesseract")
    monkeypatch.setenv("ANAMNEZ_ANONYMIZATION__ENGINE", "regex")   # testlerde GLiNER yüklenmesin (yavaş)
    monkeypatch.setenv("ANAMNEZ_ANONYMIZATION__GATE__ENABLED", "true")
    monkeypatch.setenv("ANAMNEZ_SERVER__MAX_QUEUE", "2")
    monkeypatch.setenv("ANAMNEZ_SERVER__MAX_FILE_SIZE_MB", "1")

    llm = FakeLLM()
    orig = api.Pipeline

    class PatchedPipeline(orig):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.llm = llm
            self.gate = AnonymizationGate(cfg["anonymization"]["gate"], llm)
            self.classifier = Classifier(cfg["classification"], llm)

    monkeypatch.setattr(api, "Pipeline", PatchedPipeline)
    with TestClient(api.app) as c:
        yield c


def _wait(client, job_id, headers, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{job_id}", headers=headers).json()
        if j["status"] not in ("queued", "processing"):
            return j
        time.sleep(0.1)
    raise AssertionError("iş zamanında bitmedi")


def test_api_requires_key(client):
    assert client.get("/jobs").status_code == 401


def _post(client, name, data, ref=None):
    h = {"X-API-Key": "secret", "X-Filename": name, "Content-Type": "application/octet-stream"}
    if ref:
        h["X-Ref"] = ref
    return client.post("/jobs", content=data, headers=h)


def test_api_rejects_unsupported_type(client):
    assert _post(client, "x.exe", b"abc").status_code == 415
    assert client.post("/jobs", content=b"abc", headers={"X-API-Key": "secret"}).status_code == 415  # X-Filename yok


def test_api_rejects_oversize_early(client):
    big = b"x" * (1024 * 1024 + 1)
    assert _post(client, "a.pdf", big).status_code == 413


def test_api_job_roundtrip(client):
    h = {"X-API-Key": "secret"}
    r = _post(client, "MEHMET_YILMAZ_rapor.txt", SAMPLE.encode(), ref="hastane-ref-42")
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"], h)
    assert job["status"] == "done"
    assert job["result"]["classification"]["validated_category"] == "Lung"
    dumped = json.dumps(job, ensure_ascii=False)
    assert "MEHMET" not in dumped and "YILMAZ" not in dumped     # ne metinde ne dosya adında
    assert job["filename"].endswith(".txt") and job["ref"] == "hastane-ref-42"
    assert client.get("/jobs", headers=h).json()[0]["id"] == job["id"]
    st = client.get("/stats", headers=h).json()
    assert st["jobs"]["done"] == 1 and st["workers_alive"] == 1
    assert client.get("/health").status_code == 200


# ── Deterministik kapı katmanı ──────────────────────────────────────────
def test_heuristic_layer_catches_residual_surname_even_if_llm_misses():
    """Gerçek vaka: OCR 'EÇTİM' okudu, regex kaçırdı, LLM yargıç da kaçırdı."""
    llm = FakeLLM(judge_findings=[])  # LLM hiçbir şey görmüyor
    g = AnonymizationGate({"enabled": True, "heuristics": True, "llm_judge": True}, llm)
    r = g.check("[DOKTOR_SILINDI] ERDEM\nPatoloji Servis : Tibbi Patoloji")
    assert not r.passed
    assert any(f.text == "ERDEM" and f.source == "residual_heuristics" for f in r.findings)


def test_heuristic_layer_ignores_medical_caps():
    g = AnonymizationGate({"enabled": True, "heuristics": True, "llm_judge": False}, None)
    assert g.check("KLİNİK ÖYKÜ\nAKCIĞER CA TANILI HASTANIN EGFR, ALK, ROS-1\n[YAS_ARALIGI: 45-54]").passed


# ── Çapraz OCR katmanı ──────────────────────────────────────────────────
def test_cross_ocr_catches_pii_seen_only_by_secondary_engine():
    """Gerçek vaka: GLM-OCR sol sütunu atladı; Tesseract 'MEHMET KORKMAZ' gördü."""
    from service.anonymization.gate import cross_ocr_candidates
    from src.anonymizer import ReportAnonymizer

    a = ReportAnonymizer()
    alt = "Hasta Adi Soyadi : MEHMET KORKMAZ\nTC Kimlik No : 10000000146\nTANI\nGlioblastom"
    alt_anon, _ = a.anonymize(alt)
    cands = cross_ocr_candidates(alt, alt_anon)
    assert {"MEHMET", "KORKMAZ", "10000000146"} <= cands
    assert "TANI" not in cands and "GLİOBLASTOM" not in cands

    g = AnonymizationGate({"enabled": True, "heuristics": True, "llm_judge": False}, None)
    assert g.check("MAKROSKOPİ\nGlioblastom, DSÖ derece 4", cross_candidates=cands).passed
    r = g.check("Hasta KORKMAZ\nGlioblastom", cross_candidates=cands)
    assert not r.passed and r.findings[0].source == "cross_ocr"


def test_judge_retries_on_invalid_json():
    from service.anonymization.gate import LLMJudge

    class FlakyLLM:
        model = "flaky"
        def __init__(self): self.n = 0
        def chat(self, messages, **kw):
            self.n += 1
            return "{bozuk json" if self.n == 1 else '{"findings": []}'
        def health(self): return {"ok": True}

    llm = FlakyLLM()
    assert LLMJudge(llm).check("metin") == [] and llm.n == 2


def test_stale_jobs_marked_failed_on_restart(tmp_path):
    from service.jobs import JobStore

    db = str(tmp_path / "j.db")
    s1 = JobStore(db)
    jid = s1.create("x.pdf", 10)
    s1.mark_processing(jid)
    s2 = JobStore(db)  # yeniden başlatma
    assert s2.get(jid)["status"] == "failed"



# ── İnceleme bulguları (regresyon) ──────────────────────────────────────
def test_fields_removed_never_contains_raw_values():
    """C1: silinen isim/yaş 'fields_removed' listesinde ham olarak yer almamalı."""
    from src.anonymizer import ReportAnonymizer

    a = ReportAnonymizer()
    _, rep = a.anonymize("Hasta: MEHMET YILMAZ\nMEHMET YILMAZ\nDR ÖZTÜRK\nYaş: 57\n[DOKTOR_SILINDI] ERDEM")
    dumped = json.dumps(rep.fields_removed + rep.fields_generalized, ensure_ascii=False)
    for pii in ("MEHMET", "YILMAZ", "ÖZTÜRK", "ERDEM", "57"):
        assert pii not in dumped, dumped


def test_needs_review_masks_gate_findings_by_default():
    """H: needs_review'da kalıntı PII metni ve bulgu metni dönmemeli (review_text_mode=masked)."""
    llm = FakeLLM(judge_findings=[{"text": "karsinom", "type": "person", "reason": "test"}])
    res = make_pipeline(llm).run(SAMPLE.encode(), "rapor.txt")
    assert res.status == "needs_review"
    assert "karsinom" not in res.anonymized_text and "[KAPI_BULGUSU_1:person]" in res.anonymized_text
    assert "text" not in res.gate["findings"][0] and res.gate["findings"][0]["chars"] == len("karsinom")


def test_gate_finding_with_mask_tag_is_not_dropped():
    from service.anonymization.gate import Finding, _finding_is_real

    t = "[DOKTOR_SILINDI] ERDEM\nTANI"
    assert _finding_is_real(Finding("[DOKTOR_SILINDI] ERDEM", "person", "llm"), t)
    assert _finding_is_real(Finding("Dr. Erdem", "person", "llm"), t)          # Title-case / unvanlı
    assert not _finding_is_real(Finding("[DOKTOR_SILINDI]", "person", "llm"), t)


def test_gate_with_no_layers_fails_closed():
    r = AnonymizationGate({"enabled": True, "heuristics": False, "llm_judge": False, "fail_closed": True}, None).check("x")
    assert not r.passed and "no_layers" in r.errors


def test_judge_non_list_findings_is_error():
    from service.anonymization.gate import LLMJudge

    class BadLLM:
        model = "bad"
        def chat(self, messages, **kw): return '{"findings": "yok"}'
        def health(self): return {"ok": True}

    with pytest.raises(ValueError):
        LLMJudge(BadLLM()).check("metin")


def test_queue_full_returns_503(client):
    """H: kuyruk sınırlı; dolunca 503."""
    from service import api

    # worker'ı durdurup kuyruğu doldur
    api.state.runner._stop.set()
    for t in api.state.runner._threads:
        t.join(timeout=2)
    codes = [_post(client, f"r{i}.txt", SAMPLE.encode()).status_code for i in range(4)]
    assert codes[:2] == [202, 202] and 503 in codes[2:]


def test_pipeline_error_message_has_no_raw_text():
    class BoomLLM(FakeLLM):
        def chat(self, messages, **kw): raise RuntimeError("HAM METİN: MEHMET YILMAZ 12345678901")

    p = make_pipeline(BoomLLM())
    p.gate = AnonymizationGate({"enabled": True, "heuristics": True, "llm_judge": True, "fail_closed": True}, p.llm)
    res = p.run(SAMPLE.encode(), "rapor.txt")
    assert res.status == "needs_review"  # katman hatası → fail-closed
    assert "MEHMET" not in json.dumps(res.as_dict(), ensure_ascii=False)


def test_judge_ignores_sut_and_icd_codes():
    llm = FakeLLM(judge_findings=[
        {"text": "G101951-KRAS Geni Dizi Analizi", "type": "id", "reason": "tetkik no"},
        {"text": "C34.9", "type": "id", "reason": "kod"},
        {"text": "5014591049", "type": "id", "reason": "işlem no"},   # gerçek kimlik → kalmalı
    ])
    t = "G101951-KRAS Geni Dizi Analizi\nC34.9\nislem 5014591049"
    r = AnonymizationGate({"enabled": True, "heuristics": False, "llm_judge": True}, llm).check(t)
    assert [f.text for f in r.findings] == ["5014591049"]


def test_judge_ignores_label_only_findings():
    llm = FakeLLM(judge_findings=[
        {"text": "Hasta / Velisi", "type": "person", "reason": "etiket"},
        {"text": "İsteyen Doktor", "type": "person", "reason": "etiket"},
        {"text": "Hasta Ali Yücel", "type": "person", "reason": "gerçek"},
    ])
    t = "Hasta / Velisi\nİsteyen Doktor\nHasta Ali Yücel"
    r = AnonymizationGate({"enabled": True, "heuristics": False, "llm_judge": True}, llm).check(t)
    assert [f.text for f in r.findings] == ["Hasta Ali Yücel"]


def test_judge_sees_neutral_masks_and_bare_tag_names_are_dropped():
    from service.anonymization.gate import Finding, LLMJudge, _finding_is_real

    seen = {}
    class SpyLLM(FakeLLM):
        def chat(self, messages, **kw):
            seen["prompt"] = messages[-1]["content"]; return '{"findings": [{"text": "DIPLOMA_SILINDI", "type": "other"}, {"text": "Rapor Revizyon No", "type": "other"}, {"text": "ASİSTAN DR", "type": "institution"}]}'
    t = "[DIPLOMA_SILINDI]\nRapor Revizyon No :\nASİSTAN DR.\n[DOKTOR_SILINDI]"
    assert LLMJudge(SpyLLM()).check(t) == []
    assert "_SILINDI" not in seen["prompt"] and "■" in seen["prompt"]
    assert not _finding_is_real(Finding("[[KAPI_BULGUSU_2:other]]", "other", "x"), t)


def test_judge_drops_self_refuting_findings():
    llm = FakeLLM(judge_findings=[
        {"text": "Ki67 %75 oranda pozitiftir", "type": "other", "reason": "Tıbbi terim, PII değil"},
        {"text": "AYŞE KAYA", "type": "person", "reason": "hasta adı"},
    ])
    r = AnonymizationGate({"enabled": True, "heuristics": False, "llm_judge": True}, llm).check("Ki67 %75 oranda pozitiftir\nAYŞE KAYA")
    assert [f.text for f in r.findings] == ["AYŞE KAYA"]
