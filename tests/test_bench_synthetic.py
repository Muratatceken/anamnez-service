"""Sentetik vaka regresyonu — gerçek hasta verisi gerektirmez, CI'da koşar.

1. Temiz metin (OCR hatasız): regex katmanı 20 vakada hiçbir PII'yi kaçırmamalı, hiçbir tıbbi terimi silmemeli.
2. El yazısı görüntüsü + Tesseract: LLM'siz kapı (heuristics + cross_ocr) geçtiği halde PII kalan görüntü = 0.
"""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from bench.run_bench import cer, fuzzy_contains  # noqa: E402
from bench.synth_cases import make_cases  # noqa: E402
from service.anonymization.gate import AnonymizationGate, cross_ocr_candidates  # noqa: E402
from src.anonymizer import ReportAnonymizer  # noqa: E402


@pytest.mark.parametrize("case", make_cases(20), ids=lambda c: c.id)
def test_clean_text_no_leak_no_overdeletion(case):
    out, _ = ReportAnonymizer().anonymize(case.text)
    leaked = [p for p in case.pii if fuzzy_contains(out, p)]
    lost = [k for k in case.keep if not fuzzy_contains(out, k)]
    assert leaked == [], f"sızıntı: {leaked}"
    assert lost == [], f"aşırı silme: {lost}"
    # kapı da temiz metinde bulgu üretmemeli (yanlış pozitif kontrolü, LLM'siz)
    g = AnonymizationGate({"enabled": True, "heuristics": True, "llm_judge": False}, None).check(out)
    assert g.passed, [f.text for f in g.findings]


def _ner_or_skip():
    try:
        from service.anonymization.ner import NERAnonymizer
        n = NERAnonymizer()
        n._load()
        return n
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"GLiNER modeli yüklenemedi: {type(e).__name__}")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract yok")
def test_handwriting_tesseract_gate_never_passes_with_leak(tmp_path):
    """Üretim motoru (NER+regex) ile: kapıdan geçen hiçbir görüntüde PII kalmamalı."""
    from bench.render_handwriting import build_dataset
    from service.backends.ocr import TesseractOCR

    ner = _ner_or_skip()
    items = build_dataset(str(tmp_path), n_cases=8, styles=("scan", "phone", "bad"))
    tess = TesseractOCR({"lang": "tur+eng"})
    rx = ReportAnonymizer()
    gate = AnonymizationGate({"enabled": True, "heuristics": True, "llm_judge": False, "fail_closed": True}, None)
    bad = []
    for it in items:
        png = (tmp_path / it["file"]).read_bytes()
        hyp = tess.ocr_image(png)
        n = ner.anonymize(hyp)
        out, _ = rx.anonymize(n.text)
        g = gate.check(out, cross_candidates=cross_ocr_candidates(hyp, out) | n.candidates)
        leaked = [p for p in it["pii"] if fuzzy_contains(out, p)]
        if g.passed and leaked:
            bad.append((it["file"], leaked))
    assert bad == [], f"kapı geçti ama PII kaldı: {bad}"


def test_cer_metric_sanity():
    assert cer("abc", "abc") == 0.0
    assert 0.3 < cer("Hasta Adı: Ali", "Hasta Adi: Alx yz") < 0.6
