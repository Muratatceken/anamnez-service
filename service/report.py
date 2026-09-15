"""Doktor raporu + sınıflandırma — anonim metinden, Claude ile (yapılandırılmış JSON).

Girdi: yalnızca KAPIDAN GEÇMİŞ anonim metin (egress gateway zorlar).
Çıktı: kısa, yapılandırılmış klinik özet + 33 kategoriden biri + güven + belirteçler + plan.

Tasarım notları:
  - `messages.create` + `output_config.format` (json_schema) kullanılır; `parse()` yerine elle doğrulama yapılır ki
    `stop_reason` (max_tokens/refusal) JSON'a bakmadan ÖNCE kontrol edilebilsin.
  - base_url egress host'una sabitlenir; ANTHROPIC_BASE_URL/ANTHROPIC_LOG/proxy env'leri yok sayılır.
  - Model yanıtı da PII taramasından geçer (rapor serbest metnine kalıntı sızmasın).
  - Geçici API hataları (429/5xx/bağlantı/zaman aşımı) TransientCloudError olarak ayrılır; pipeline yeniden dener.
"""

import json
import logging
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.models import CancerCategory
from src.validator import Validator

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = [c.value for c in CancerCategory]
EFFORTS = ("low", "medium", "high", "xhigh", "max")


# ── hata sınıfları (CloudError/TransientCloudError: cloud_backends) ─────
from .cloud_backends import CloudError, TransientCloudError, make_backend  # noqa: E402


class ModelRefusal(CloudError):
    def __init__(self, category: Optional[str]):
        super().__init__(f"model reddetti: {category}")
        self.category = category


class ReportTruncated(CloudError):
    pass


# ── şema ────────────────────────────────────────────────────────────────
class Marker(BaseModel):
    ad: str = Field(description="Belirteç/test adı: Ki67, HER2, JAK2 V617F, PSA, CA-125 …")
    deger: str = Field(description="Sonuç: %70, pozitif, negatif, 14 ng/mL …")
    kanit: str = Field(default="", description="Metinden birebir kısa alıntı (≤80 karakter)")


class Bulgu(BaseModel):
    madde: str = Field(description="Tek cümle, doktor için")
    kanit: str = Field(default="", description="Metinden birebir kısa alıntı (≤80 karakter)")


class DoctorReport(BaseModel):
    """Claude'un döndüreceği yapılandırılmış rapor (json_schema)."""
    malignite_durumu: Literal["malign", "benign", "belirsiz"] = Field(description="Metne göre malignite")
    kategori: Literal[tuple(CATEGORIES)] = Field(description="33 kategoriden biri")  # type: ignore[valid-type]
    guven: float = Field(ge=0, le=1, description="Kategori güveni 0-1 (rubrik: sistem promptu)")
    gerekce: str = Field(description="Kategori seçiminin 1-2 cümlelik gerekçesi; metinden kanıt içersin")
    yas_araligi: Optional[str] = Field(default=None, description="[YAS_ARALIGI: a-b] etiketinden 'a-b'; yoksa null")
    cinsiyet: Optional[Literal["kadin", "erkek"]] = Field(default=None)
    histolojik_tip: Optional[str] = Field(default=None, description="örn. Skuamöz hücreli karsinom; yoksa null")
    primer_bolge: Optional[str] = Field(default=None, description="örn. Sol akciğer alt lob; yoksa null")
    evre_tnm: Optional[str] = Field(default=None, description="pT/pN/pM veya klinik evre, metinde varsa")
    derece: Optional[str] = Field(default=None, description="Grade / Gleason / DSÖ derece, metinde varsa")
    cerrahi_sinir: Optional[str] = Field(default=None, description="pozitif/negatif/… metinde varsa")
    lenf_nodu: Optional[str] = Field(default=None, description="örn. 2/12 metastatik; metinde varsa")
    belirtecler: list[Marker] = Field(default_factory=list, description="IHK / moleküler / laboratuvar")
    onemli_bulgular: list[Bulgu] = Field(default_factory=list, description="3-7 madde")
    tedavi_ve_plan: list[str] = Field(default_factory=list, description="Metinde geçen tedavi/plan maddeleri")
    onerilen_ek_tetkik: list[str] = Field(default_factory=list, description="Yalnızca metinde önerilenler")
    ozet: str = Field(description="Doktor için en fazla 3 cümle")
    belirsizlikler: list[str] = Field(default_factory=list, description="OCR bozukluğu / çelişki / eksik veri")
    okunabilirlik: Literal["iyi", "orta", "kotu"]

    @field_validator("guven", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("kategori", mode="before")
    @classmethod
    def _norm_cat(cls, v):
        s = str(v or "").strip()
        for c in CATEGORIES:
            if c.lower() == s.lower().replace(" ", "_").replace("-", "_"):
                return c
        return s


SYSTEM_PROMPT = """Sen deneyimli bir tıbbi onkoloji/patoloji uzmanısın. Sana kişisel verileri MASKELENMİŞ
([..._SILINDI] etiketleri) Türkçe bir anamnez veya patoloji raporu metni verilecek. Metin OCR ile çıkarılmıştır:
harf hataları, bozuk etiketler, eksik satırlar olabilir ("HERD NEGATIE" → HER2 negatif, "4 PYLORI" → H. pylori).

GÖREV
1. Malignite durumu ve KATEGORİ: primer tümörün kaynak organına göre tek kategori. Karar tablosu:
   - Metastaz: metastazın değil PRİMER tümörün organı. Primer belirlenemiyorsa (CUP) → Other, güven ≤0.5.
   - Lösemiler (AML, ALL, KLL, KML) ve kan kaynaklı hematolojik maligniteler → Blood;
     kemik iliği hastalıkları: MDS, MPN (polisitemi vera, esansiyel trombositoz, myelofibrozis), multipl myelom,
     kemik iliği biyopsisiyle konan tanılar → Bone_Marrow.
   - Lenfoma (Hodgkin, NHL, DBBHL, foliküler…) → Lymph_Nodes (ekstranodal olsa da).
   - Glioblastom/astrositom/menenjiyom → Brain; periferik sinir tümörleri → Nervous_System.
   - Kolon/rektum → Colorectal; serviks → Cervix; endometrium → Uterus; intrahepatik kolanjiyokarsinom → Intrahepatic;
     ekstrahepatik safra yolu/safra kesesi → Bile_Duct; baş-boyun skuamöz (larenks, oral kavite) → Head_and_Neck.
   - Benign, reaktif, displazi (yüksek dereceli olsa da), kanser yok → malignite_durumu 'benign' ve kategori Other.
   Cancer_all ve Inflammatory kategorilerini KULLANMA.
2. GÜVEN rubriği: ≥0.9 primer organ VE histoloji açık; 0.7-0.89 organ açık, histoloji/OCR kısmen belirsiz;
   0.5-0.69 organ IHK/bağlamdan çıkarım; <0.5 primer belirsiz veya metin okunamıyor.
3. RAPOR: doktorun 30 saniyede okuyacağı kısalıkta. Histolojik tip, primer bölge, evre (TNM), derece, cerrahi
   sınır, lenf nodu, belirteçler (ad+değer), 3-7 önemli bulgu, tedavi/plan, yalnızca metinde önerilen ek tetkikler,
   ≤3 cümle özet.
4. KANIT: her belirteç ve önemli bulgu için metinden BİREBİR kısa alıntı (`kanit`, ≤80 karakter) ver. Alıntı
   gösteremediğin hiçbir şeyi yazma. Sayısal değerleri OCR'dan şüpheli okuduysan 'belirsizlikler'e yaz ve değeri "?" ile
   işaretle. Metinde olmayan bilgiyi ASLA uydurma; yoksa alanı null/boş bırak.
5. BELİRSİZLİKLER: OCR bozuk satırlar, çelişen değerler, eksik bölümler, tereddütlü normalizasyonlar. Metin büyük
   ölçüde okunamıyorsa okunabilirlik 'kotu' ve güven <0.5.
6. MASKELER: [YAS_ARALIGI: a-b] → yas_araligi "a-b". Diğer [..._SILINDI] etiketlerini rapora TAŞIMA; hiçbir kişi,
   kurum, şehir adı, tarih veya numara yazma. Cinsiyet metinde açıkça varsa yaz.
7. DİL: Türkçe; tıbbi terimleri metindeki gibi koru (WHO/DSÖ tanı adları İngilizce kalabilir); pozitif/negatif
   yazımını normalize et; birimleri koru.

Kategori listesi: """ + ", ".join(CATEGORIES)

USER_PREFIX = "RAPOR METNİ:\n<<<\n"
USER_SUFFIX = "\n>>>"


class ReportGenerator:
    """Claude ile rapor üretir. `client` enjekte edilebilir (test/sahte)."""

    def __init__(self, cfg: dict, client=None, validator_cfg: Optional[dict] = None, host: str = "api.anthropic.com",
                 backend=None):
        self.provider = cfg.get("provider", "anthropic")
        self.model = cfg.get("model", "claude-opus-5")
        self.effort = cfg.get("effort", "high")
        if self.effort not in EFFORTS:
            raise ValueError(f"cloud.effort geçersiz: {self.effort!r} (izinli: {EFFORTS})")
        self.max_tokens = int(cfg.get("max_tokens", 16000))
        self.host = host
        self._cfg = cfg
        self.validator = Validator(validator_cfg or {"confidence_threshold": 0.6, "require_human_review_below": 0.4})
        self.schema = DoctorReport.model_json_schema()
        # Test/sahte: `client` (Anthropic messages.create taklidi) veya `backend` (complete() taklidi)
        self._backend = backend
        if client is not None:
            self._backend = _ClientBackend(client, self.model, self.effort, self.max_tokens)

    def _get_backend(self):
        if self._backend is None:
            self._backend = make_backend(self._cfg)
        return self._backend

    def generate(self, anonymized_text: str) -> dict:
        response = self._get_backend().complete(SYSTEM_PROMPT, USER_PREFIX + anonymized_text + USER_SUFFIX, self.schema)
        if response.stop_reason == "refusal":
            raise ModelRefusal(response.refusal_category)
        if response.stop_reason == "max_tokens":
            raise ReportTruncated(f"max_tokens={self.max_tokens}")
        try:
            report = DoctorReport.model_validate_json(response.text)
        except (ValidationError, ValueError) as e:
            raise CloudError(f"şema doğrulama: {type(e).__name__}") from e

        from src.models import ClassificationResult

        val = self.validator.validate(
            ClassificationResult(category=report.kategori, confidence=report.guven, reasoning=report.gerekce,
                                 histological_type=report.histolojik_tip, primary_site=report.primer_bolge),
            anonymized_text,
        )
        return {
            "rapor": report.model_dump(),
            "validated_category": val.validated_category,
            "adjusted_confidence": round(val.confidence_adjusted, 3),
            "keyword_matches": val.keyword_matches,
            "warnings": val.warnings,
            "provider": self.provider,
            "model": self.model,
            "usage": response.usage,
            "request_id": response.request_id,
        }


class _ClientBackend:
    """Anthropic `messages.create` uyumlu bir istemciyi (gerçek/sahte) backend arayüzüne sarar (testler)."""

    def __init__(self, client, model, effort, max_tokens):
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens

    def complete(self, system, user, schema):
        from .cloud_backends import CloudResponse

        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in r.content if getattr(b, "type", "") == "text"), "")
        stop = {"end_turn": "end_turn", "max_tokens": "max_tokens", "refusal": "refusal"}.get(r.stop_reason, "other")
        d = getattr(r, "stop_details", None); u = getattr(r, "usage", None)
        return CloudResponse(text=text, stop_reason=stop, refusal_category=getattr(d, "category", None) if stop == "refusal" else None,
                             usage={"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None)},
                             request_id=getattr(r, "_request_id", None))


# ── rapor çıktısı PII taraması ──────────────────────────────────────────
def _walk_strings(obj, path=""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")


def _set_path(obj, path: str, value):
    parts = re.findall(r"\[(\d+)\]|([^.\[\]]+)", path)
    cur = obj
    for i, (idx, key) in enumerate(parts):
        last = i == len(parts) - 1
        k = int(idx) if idx else key
        if last:
            cur[k] = value
        else:
            cur = cur[k]


def redact_report(rapor: dict, candidates: set[str], final_rules) -> list[str]:
    """Raporun tüm string alanlarını PII adayları + son kurallarla tara; isabet olan alanı maskele.
    Döndürür: maskelenen alan yolları."""
    from .anonymization.gate import tr_fold

    hits = []
    folded_cands = [tr_fold(c.strip()) for c in candidates if len(c.strip()) >= 3]
    tokens = {t for c in folded_cands for t in re.split(r"[^\wçğıöşü]+", c) if len(t) >= 4}
    for path, s in list(_walk_strings(rapor)):
        f = tr_fold(s)
        bad = False
        for c in folded_cands:
            if re.search(r"(?<![\wçğıöşü])" + r"\s+".join(map(re.escape, c.split())) + r"(?![\wçğıöşü])", f):
                bad = True
                break
        if not bad:
            for t in tokens:
                if re.search(r"(?<![\wçğıöşü])" + re.escape(t) + r"(?![\wçğıöşü])", f):
                    bad = True
                    break
        if not bad:
            bad = any(rx.search(s) for _, rx in final_rules)
        if bad:
            _set_path(rapor, path, "[RAPOR_KALINTI_SILINDI]")
            hits.append(path)
    return hits


# ── markdown ────────────────────────────────────────────────────────────
def report_to_markdown(r: dict) -> str:
    rep = r["rapor"]
    blocks = []
    head = (f"**Otomatik özet ({r.get('model', 'LLM')})** — kategori **{r['validated_category']}** · güven {rep['guven']:.2f}"
            + (f" (doğrulama sonrası {r['adjusted_confidence']:.2f})" if r.get("adjusted_confidence") != rep.get("guven") else "")
            + f" · malignite: {rep['malignite_durumu']} · okunabilirlik: {rep['okunabilirlik']}")
    blocks.append(head)
    facts = [("Histolojik tip", rep.get("histolojik_tip")), ("Primer bölge", rep.get("primer_bolge")),
             ("Evre (TNM)", rep.get("evre_tnm")), ("Derece", rep.get("derece")), ("Cerrahi sınır", rep.get("cerrahi_sinir")),
             ("Lenf nodu", rep.get("lenf_nodu")), ("Yaş aralığı", rep.get("yas_araligi")), ("Cinsiyet", rep.get("cinsiyet"))]
    facts = [f"- **{k}:** {v}" for k, v in facts if v]
    if facts:
        blocks.append("\n".join(facts))
    if rep.get("belirtecler"):
        blocks.append("**Belirteçler**\n\n| Belirteç | Sonuç |\n|---|---|\n" + "\n".join(f"| {m['ad']} | {m['deger']} |" for m in rep["belirtecler"]))
    if rep.get("onemli_bulgular"):
        blocks.append("**Önemli bulgular**\n\n" + "\n".join(f"- {b['madde']}" for b in rep["onemli_bulgular"]))
    if rep.get("tedavi_ve_plan"):
        blocks.append("**Tedavi / plan**\n\n" + "\n".join(f"- {b}" for b in rep["tedavi_ve_plan"]))
    if rep.get("onerilen_ek_tetkik"):
        blocks.append("**Önerilen ek tetkik**\n\n" + "\n".join(f"- {b}" for b in rep["onerilen_ek_tetkik"]))
    blocks.append(f"**Özet:** {rep['ozet']}\n\n*Gerekçe:* {rep['gerekce']}")
    if rep.get("belirsizlikler"):
        blocks.append("**Belirsizlikler**\n\n" + "\n".join(f"- {b}" for b in rep["belirsizlikler"]))
    if r.get("warnings"):
        blocks.append("**Doğrulama uyarıları**\n\n" + "\n".join(f"- {w}" for w in r["warnings"]))
    return "\n\n".join(blocks)
