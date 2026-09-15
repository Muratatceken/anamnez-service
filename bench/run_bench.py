"""El yazısı OCR + anonimizasyon benchmark'ı.

Her görüntü × her OCR backend için:
  cer            karakter hata oranı (ground truth'a göre; Türkçe-duyarsız, boşluk normalize)
  pii_read       PII parçalarının OCR metninde (bulanık eşleşmeyle) okunma oranı  → OCR "PII görüyor mu?"
  keep_read      korunması gereken tıbbi terimlerin okunma oranı                  → OCR tıbbi içeriği görüyor mu?
  leak           anonimize metinde (bulanık) hâlâ görünen PII sayısı              → 0 OLMALI
  over_del       anonimize metinde kaybolan tıbbi terim sayısı                    → aşırı silme
  gate           kapı sonucu (heuristics + cross_ocr[+ llm_judge])
  cls_ok         sınıflandırma beklenen kategoriyle uyuştu mu (LLM verilirse)

Kullanım:
  python bench/run_bench.py --data bench/data --ocr tesseract --ocr ollama:glm-ocr
  python bench/run_bench.py --data bench/data --ocr openai:http://localhost:8000/v1:glm-ocr \
        --ocr openai:http://localhost:8001/v1:paddleocr-vl --llm openai:http://localhost:8002/v1:qwen3-8b
"""

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service.anonymization.gate import AnonymizationGate, cross_ocr_candidates, tr_fold  # noqa: E402
from service.anonymization.ner import NERAnonymizer  # noqa: E402
from service.backends.llm import make_llm  # noqa: E402
from service.backends.ocr import TesseractOCR, VisionLLMOCR  # noqa: E402
from src.anonymizer import ReportAnonymizer  # noqa: E402


# ── metrikler ──────────────────────────────────────────────────────────────
def norm(s: str) -> str:
    s = tr_fold(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a or b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(gt: str, hyp: str) -> float:
    g, h = norm(gt).replace("\n", " "), norm(hyp).replace("\n", " ")
    return round(levenshtein(g, h) / max(1, len(g)), 4)


def fuzzy_contains(hay: str, needle: str, thr: float = 0.85) -> bool:
    """needle, hay içinde (normalize) en az thr benzerlikle geçiyor mu? (OCR hatalarına toleranslı)"""
    h, n = norm(hay).replace("\n", " "), norm(needle).replace("\n", " ")
    if not n:
        return False
    if n in h:
        return True
    # yalnızca rakamlardan oluşan PII: rakam dizisi karşılaştır
    if n.replace(" ", "").replace(".", "").isdigit():
        digits = re.sub(r"\D", "", n)
        return digits in re.sub(r"\D", "", h)
    L = len(n)
    best = 0.0
    for w in (L - 2, L, L + 2):
        if w <= 0:
            continue
        for i in range(0, max(1, len(h) - w + 1), 1):
            r = difflib.SequenceMatcher(None, n, h[i:i + w]).ratio()
            if r > best:
                best = r
                if best >= thr:
                    return True
    return best >= thr


# ── OCR backend'leri ───────────────────────────────────────────────────────
def make_ocr(spec: str, timeout: float):
    """tesseract | ollama:<model> | openai:<base_url>:<model>"""
    if spec == "tesseract":
        t = TesseractOCR({"lang": "tur+eng", "tesseract_timeout": timeout})
        return "tesseract", t.ocr_image
    if spec.startswith("ollama:"):
        model = spec.split(":", 1)[1]
        v = VisionLLMOCR({"backend": "ollama", "base_url": "http://127.0.0.1:11434", "model": model,
                          "timeout": timeout, "max_dimension": 2000, "num_ctx": 8192})
        return f"ollama:{model}", v.ocr_image
    if spec.startswith("openai:"):
        _, rest = spec.split(":", 1)
        base, model = rest.rsplit(":", 1)
        v = VisionLLMOCR({"backend": "openai", "base_url": base, "model": model, "timeout": timeout, "max_dimension": 2000})
        return f"vllm:{model}", v.ocr_image
    raise ValueError(spec)


def make_llm_from_spec(spec: str, timeout: float):
    if spec.startswith("ollama:"):
        return make_llm({"backend": "ollama", "base_url": "http://127.0.0.1:11434", "model": spec.split(":", 1)[1],
                         "timeout": timeout, "num_ctx": 8192, "think": False, "temperature": 0.0})
    _, rest = spec.split(":", 1)
    base, model = rest.rsplit(":", 1)
    return make_llm({"backend": "openai", "base_url": base, "model": model, "timeout": timeout, "temperature": 0.0})


# ── ana döngü ──────────────────────────────────────────────────────────────
def run(data_dir: Path, ocr_specs: list[str], llm_spec: str | None, timeout: float, limit: int | None,
        out_dir: Path, classify: bool, engine: str = "ner+regex", cloud: str | None = None) -> dict:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if limit:
        meta = meta[:limit]
    rx = ReportAnonymizer()
    ner = NERAnonymizer() if engine.startswith("ner") else None

    class _Anon:
        """Servisteki Pipeline.anonymize_text ile aynı sıra: NER → regex; adayları döndürür."""
        def anonymize(self, text):
            cands = set()
            if ner is not None:
                n = ner.anonymize(text)
                text, cands = n.text, n.candidates
            out, rep_ = rx.anonymize(text)
            rep_.ner_candidates = cands
            return out, rep_

    anon = _Anon()
    llm = make_llm_from_spec(llm_spec, timeout) if llm_spec else None
    gate = AnonymizationGate({"enabled": True, "heuristics": True, "llm_judge": bool(llm), "fail_closed": True}, llm)
    classifier = None
    if llm and classify:
        from service.classification import Classifier
        classifier = Classifier({"confidence_threshold": 0.6, "require_human_review_below": 0.4}, llm)
    reporter = egress = None
    if cloud:
        from service.egress import _FINAL_RULES, EgressGateway
        from service.report import ReportGenerator, redact_report
        prov, model = cloud.split(":", 1)
        key_file = {"gemini": "deploy/gemini_key.txt", "anthropic": "deploy/anthropic_key.txt"}[prov]
        cfg = {"enabled": True, "provider": prov, "model": model, "effort": "high", "api_key_file": key_file, "timeout": timeout}
        egress = EgressGateway(cfg)
        reporter = ReportGenerator(cfg, host=egress.host)

    tess = TesseractOCR({"lang": "tur+eng", "tesseract_timeout": timeout})
    backends = [make_ocr(s, timeout) for s in ocr_specs]
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in meta:
        png = (data_dir / item["file"]).read_bytes()
        gt = (data_dir / item["file"].replace(".png", ".gt.txt")).read_text(encoding="utf-8")
        # çapraz kontrol için tesseract metni (servisteki gibi) — bir kez
        try:
            alt_text = tess.ocr_image(png)
        except Exception:  # noqa: BLE001
            alt_text = ""
        alt_anon, alt_rep = anon.anonymize(alt_text)
        cross = cross_ocr_candidates(alt_text, alt_anon) | getattr(alt_rep, "ner_candidates", set())

        for name, fn in backends:
            t0 = time.time()
            try:
                hyp = fn(png)
                err = ""
            except Exception as e:  # noqa: BLE001
                hyp, err = "", type(e).__name__
            dt = round(time.time() - t0, 1)
            anon_text, rep = anon.anonymize(hyp)
            cands = (cross if name != "tesseract" else set()) | getattr(rep, "ner_candidates", set())
            g = gate.check(anon_text, cross_candidates=cands)
            leaks = [p for p in item["pii"] if fuzzy_contains(anon_text, p)]
            over = [k for k in item["keep"] if fuzzy_contains(hyp, k) and not fuzzy_contains(anon_text, k)]
            row = {
                "file": item["file"], "case": item["case_id"], "kind": item["kind"], "font": item["font"],
                "style": item["style"], "ocr": name, "seconds": dt, "error": err,
                "chars": len(hyp), "cer": cer(gt, hyp) if hyp else 1.0,
                "pii_total": len(item["pii"]), "pii_read": sum(fuzzy_contains(hyp, p) for p in item["pii"]),
                "keep_total": len(item["keep"]), "keep_read": sum(fuzzy_contains(hyp, k) for k in item["keep"]),
                "leak": len(leaks), "leaked": leaks, "over_del": len(over), "over_deleted": over,
                "gate_pass": g.passed, "gate_findings": [f.text for f in g.findings][:5], "gate_errors": g.errors,
                "fields_removed": len(rep.fields_removed),
            }
            if classifier and g.passed and len(anon_text) > 40:
                try:
                    c = classifier.classify(anon_text)
                    row["cls"] = c.validated_category
                    row["cls_ok"] = c.validated_category == item["category"]
                except Exception as e:  # noqa: BLE001
                    row["cls"] = f"ERR:{type(e).__name__}"
                    row["cls_ok"] = False
            if reporter and g.passed and len(anon_text) > 40:
                t1 = time.time()
                d = egress.decide(anon_text, g, cands)
                if not d.allowed:
                    row["cls"] = "EGRESS_BLOCK"; row["cls_ok"] = False; row["egress_reasons"] = d.reasons
                else:
                    try:
                        r = reporter.generate(anon_text)
                        hits = redact_report(r["rapor"], cands, _FINAL_RULES)
                        row["cls"] = r["validated_category"]; row["cls_ok"] = r["validated_category"] == item["category"]
                        row["guven"] = r["rapor"]["guven"]; row["okunabilirlik"] = r["rapor"]["okunabilirlik"]
                        row["belirsizlik_n"] = len(r["rapor"]["belirsizlikler"]); row["report_redacted"] = hits
                        row["tokens"] = r["usage"]; row["report_seconds"] = round(time.time() - t1, 1)
                        row["rapor"] = r["rapor"]
                    except Exception as e:  # noqa: BLE001
                        row["cls"] = f"ERR:{type(e).__name__}"; row["cls_ok"] = False
            rows.append(row)
            print(f"{item['file'][:44]:44} {name:18} cer={row['cer']:.2f} pii={row['pii_read']}/{row['pii_total']} "
                  f"keep={row['keep_read']}/{row['keep_total']} leak={row['leak']} over={row['over_del']} "
                  f"gate={'PASS' if g.passed else 'FAIL'} {row.get('cls', '')} {dt}s {err}", flush=True)
            # ham OCR ve anonim çıktıyı sakla (sentetik veri → sorun yok)
            (out_dir / f"{Path(item['file']).stem}__{name.replace(':', '_')}.ocr.txt").write_text(hyp, encoding="utf-8")
            (out_dir / f"{Path(item['file']).stem}__{name.replace(':', '_')}.anon.txt").write_text(anon_text, encoding="utf-8")

    summary = summarize(rows)
    (out_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print("\n" + to_markdown(summary))
    return summary


def summarize(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["ocr"], []).append(r)
    out = {}
    for name, rs in by.items():
        n = len(rs)
        cls_rows = [r for r in rs if "cls_ok" in r]
        out[name] = {
            "n": n,
            "errors": sum(1 for r in rs if r["error"]),
            "cer_mean": round(sum(r["cer"] for r in rs) / n, 3),
            "cer_median": round(sorted(r["cer"] for r in rs)[n // 2], 3),
            "pii_read_rate": round(sum(r["pii_read"] for r in rs) / max(1, sum(r["pii_total"] for r in rs)), 3),
            "keep_read_rate": round(sum(r["keep_read"] for r in rs) / max(1, sum(r["keep_total"] for r in rs)), 3),
            "leak_images": sum(1 for r in rs if r["leak"]),
            "leak_items": sum(r["leak"] for r in rs),
            "over_del_items": sum(r["over_del"] for r in rs),
            "gate_pass": sum(1 for r in rs if r["gate_pass"]),
            "gate_fail_with_leak": sum(1 for r in rs if not r["gate_pass"] and r["leak"]),
            "gate_pass_with_leak": sum(1 for r in rs if r["gate_pass"] and r["leak"]),   # ← KRİTİK: 0 olmalı
            "cls_acc": round(sum(r["cls_ok"] for r in cls_rows) / len(cls_rows), 3) if cls_rows else None,
            "cls_n": len(cls_rows),
            "report_redacted": sum(1 for r in rs if r.get("report_redacted")),
            "okunabilirlik": {k: sum(1 for r in rs if r.get("okunabilirlik") == k) for k in ("iyi", "orta", "kotu")} if any("okunabilirlik" in r for r in rs) else None,
            "report_sec_mean": round(sum(r.get("report_seconds", 0) for r in rs) / max(1, sum(1 for r in rs if "report_seconds" in r)), 1),
            "sec_mean": round(sum(r["seconds"] for r in rs) / n, 1),
            "by_style": {
                st: {"cer": round(sum(r["cer"] for r in rs if r["style"] == st) / max(1, sum(1 for r in rs if r["style"] == st)), 3),
                     "pii_read": round(sum(r["pii_read"] for r in rs if r["style"] == st) / max(1, sum(r["pii_total"] for r in rs if r["style"] == st)), 3)}
                for st in sorted({r["style"] for r in rs})
            },
        }
    return out


def to_markdown(summary: dict) -> str:
    lines = ["| OCR | n | CER ort | CER med | PII okuma | tıbbi okuma | sızıntı (img/parça) | **kapı PASS + sızıntı** | aşırı silme | kapı PASS | sınıf. doğruluk | sn/sayfa |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['n']} | {s['cer_mean']} | {s['cer_median']} | {s['pii_read_rate']:.0%} | {s['keep_read_rate']:.0%} | "
                     f"{s['leak_images']}/{s['leak_items']} | **{s['gate_pass_with_leak']}** | {s['over_del_items']} | {s['gate_pass']}/{s['n']} | "
                     f"{'-' if s['cls_acc'] is None else f'{s['cls_acc']:.0%}'} | {s['sec_mean']} |")
    lines.append("")
    for name, s in summary.items():
        lines.append(f"- {name} stil bazında: " + ", ".join(f"{st}: CER {v['cer']} / PII {v['pii_read']:.0%}" for st, v in s["by_style"].items()))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/data")
    ap.add_argument("--ocr", action="append", required=True, help="tesseract | ollama:<model> | openai:<base>:<model>")
    ap.add_argument("--llm", help="yargıç/sınıflandırıcı: ollama:<model> | openai:<base>:<model>")
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--engine", default="ner+regex", choices=["ner+regex", "regex"])
    ap.add_argument("--cloud", help="bulut raporu: gemini:<model> | anthropic:<model> (anahtar: deploy/*_key.txt)")
    a = ap.parse_args()
    run(Path(a.data), a.ocr, a.llm, a.timeout, a.limit, Path(a.out), classify=not a.no_classify, engine=a.engine, cloud=a.cloud)
