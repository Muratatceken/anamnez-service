"""Doktor raporu + sınıflandırma — anonim metinden, Claude ile (yapılandırılmış JSON).

Girdi: yalnızca KAPIDAN GEÇMİŞ anonim metin (egress gateway zorlar).
Çıktı: kısa, yapılandırılmış klinik özet + 33 kategoriden biri + güven + belirteçler + plan.
"""

import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.models import CancerCategory
from src.validator import Validator

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = [c.value for c in CancerCategory]


class Marker(BaseModel):
    ad: str = Field(description="Belirteç/test adı, örn. Ki67, HER2, JAK2 V617F, PSA")
    deger: str = Field(description="Sonuç, örn. %70, pozitif, negatif, 14 ng/mL")


class DoctorReport(BaseModel):
    """Claude'un döndüreceği yapılandırılmış rapor (output_format)."""
    kategori: Literal[tuple(CATEGORIES)] = Field(description="33 kategoriden biri; benign/kanser dışı ise Other")  # type: ignore[valid-type]
    guven: float = Field(ge=0, le=1, description="Kategori güveni 0-1")
    gerekce: str = Field(description="Kategori seçiminin 1-2 cümlelik gerekçesi")
    histolojik_tip: Optional[str] = Field(default=None, description="örn. Skuamöz hücreli karsinom; yoksa null")
    primer_bolge: Optional[str] = Field(default=None, description="örn. Sol akciğer alt lob; yoksa null")
    evre_grade: Optional[str] = Field(default=None, description="Evre/derece bilgisi; yoksa null")
    belirtecler: list[Marker] = Field(default_factory=list, description="IHK/moleküler/laboratuvar belirteçleri")
    onemli_bulgular: list[str] = Field(default_factory=list, description="Doktor için 3-7 madde, her biri tek cümle")
    tedavi_ve_plan: list[str] = Field(default_factory=list, description="Metinde geçen tedavi/plan maddeleri")
    ozet: str = Field(description="Doktor için en fazla 3 cümlelik klinik özet")
    belirsizlikler: list[str] = Field(default_factory=list, description="OCR bozukluğu/çelişki/eksik veri notları")
    okunabilirlik: Literal["iyi", "orta", "kotu"] = Field(description="Kaynak metnin okunabilirliği")


SYSTEM_PROMPT = """Sen deneyimli bir tıbbi onkoloji/patoloji uzmanısın. Sana kişisel verileri MASKELENMİŞ
([..._SILINDI] etiketleri) Türkçe bir anamnez veya patoloji raporu metni verilecek. Metin OCR ile
çıkarılmıştır; harf hataları, eksik satırlar ve bozuk etiketler olabilir.

Görevin:
1. Kanserin PRİMER lokalizasyonuna göre verilen listeden TEK kategori seç (metastaz varsa primer tümörün
   kaynağı; lösemi → Blood, kemik iliği hastalıkları → Bone_Marrow, lenfoma → Lymph_Nodes; kanser yok/benign
   ya da belirlenemiyorsa → Other) ve güven skoru ver.
2. Doktorun 30 saniyede okuyabileceği KISA, yapılandırılmış bir rapor üret: histolojik tip, primer bölge,
   evre/derece, belirteçler (ad + değer), önemli bulgular (madde), tedavi/plan (madde), en fazla 3 cümlelik özet.
3. Metinde OLMAYAN hiçbir bilgiyi uydurma. Emin olmadığın veya OCR'dan bozuk gelen her şeyi
   'belirsizlikler' listesine yaz. Maske etiketlerini rapora taşıma; kişi/kurum adı yazma.
4. Tıbbi terimleri olduğu gibi koru (kısaltmaları açıklama gerekmez). Türkçe yaz.

Kategori listesi: """ + ", ".join(CATEGORIES)


class ReportGenerator:
    """Claude ile rapor üretir. `client` enjekte edilebilir (test/sahte)."""

    def __init__(self, cfg: dict, client=None, validator_cfg: Optional[dict] = None):
        self.model = cfg.get("model", "claude-opus-5")
        self.effort = cfg.get("effort", "high")
        self.max_tokens = int(cfg.get("max_tokens", 4000))
        self.timeout = float(cfg.get("timeout", 120))
        self.max_retries = int(cfg.get("max_retries", 2))
        self._client = client
        self._cfg = cfg
        self.validator = Validator(validator_cfg or {"confidence_threshold": 0.6, "require_human_review_below": 0.4})

    def _get_client(self):
        if self._client is None:
            import anthropic

            key_file = self._cfg.get("api_key_file")
            api_key = None
            if key_file and Path(key_file).exists():
                api_key = Path(key_file).read_text(encoding="utf-8").strip()
            # api_key None → SDK ANTHROPIC_API_KEY / ant auth profilinden çözer
            kwargs = {"api_key": api_key, "timeout": self.timeout, "max_retries": self.max_retries}
            proxy = self._cfg.get("proxy")   # kapalı devrede: http://egress-proxy:3128 (yalnızca api.anthropic.com'a izinli)
            if proxy:
                kwargs["http_client"] = anthropic.DefaultHttpxClient(proxy=proxy)
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def generate(self, anonymized_text: str) -> dict:
        client = self._get_client()
        response = client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": "RAPOR METNİ:\n<<<\n" + anonymized_text + "\n>>>"}],
            output_format=DoctorReport,
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"model reddetti: {getattr(details, 'category', None)}")
        report: DoctorReport = response.parsed_output
        # Keyword doğrulama (yerel, deterministik) — LLM kategorisiyle çelişirse uyar
        from src.models import ClassificationResult

        val = self.validator.validate(
            ClassificationResult(category=report.kategori, confidence=report.guven, reasoning=report.gerekce,
                                 histological_type=report.histolojik_tip, primary_site=report.primer_bolge),
            anonymized_text,
        )
        usage = getattr(response, "usage", None)
        return {
            "rapor": report.model_dump(),
            "validated_category": val.validated_category,
            "adjusted_confidence": round(val.confidence_adjusted, 3),
            "keyword_matches": val.keyword_matches,
            "warnings": val.warnings,
            "model": self.model,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            },
            "request_id": getattr(response, "_request_id", None),
        }


def report_to_markdown(r: dict) -> str:
    """İnsan okunabilir kısa rapor (API sonucunda 'rapor_md')."""
    rep = r["rapor"]
    lines = [f"**Kategori:** {r['validated_category']} (güven {r['adjusted_confidence']:.2f})"]
    if rep.get("histolojik_tip"):
        lines.append(f"**Histolojik tip:** {rep['histolojik_tip']}")
    if rep.get("primer_bolge"):
        lines.append(f"**Primer bölge:** {rep['primer_bolge']}")
    if rep.get("evre_grade"):
        lines.append(f"**Evre/derece:** {rep['evre_grade']}")
    if rep.get("belirtecler"):
        lines.append("**Belirteçler:** " + ", ".join(f"{m['ad']}: {m['deger']}" for m in rep["belirtecler"]))
    if rep.get("onemli_bulgular"):
        lines.append("**Önemli bulgular:**")
        lines += [f"- {b}" for b in rep["onemli_bulgular"]]
    if rep.get("tedavi_ve_plan"):
        lines.append("**Tedavi / plan:**")
        lines += [f"- {b}" for b in rep["tedavi_ve_plan"]]
    lines.append(f"**Özet:** {rep['ozet']}")
    if rep.get("belirsizlikler"):
        lines.append("**Belirsizlikler:** " + "; ".join(rep["belirsizlikler"]))
    if r.get("warnings"):
        lines.append("**Doğrulama uyarıları:** " + "; ".join(r["warnings"]))
    return "\n".join(lines)
