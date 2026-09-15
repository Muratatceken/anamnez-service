"""Doğrulama kapısı — anonimize metinde kalan PII'yi bağımsız katmanlarla arar.

Katmanlar:
  1. LLM yargıç (Qwen3 vb.): "bu metinde kalan kişisel veri var mı?" → JSON bulgular
  2. GLiNER Türkçe PII modeli (opsiyonel; neondijital/neonredact-tr-model)

fail_closed=True ise herhangi bir katman bulgu üretirse kapı KAPALI sayılır ve
pipeline sonucu döndürmez (needs_review).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..backends.llm import LLMBackend, extract_json

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "Sen KVKK uyumluluk denetçisisin. Görevin, anonimleştirilmiş Türkçe tıbbi metinde "
    "KALAN kişisel verileri bulmaktır. Sadece JSON döndür."
)

JUDGE_PROMPT = """Aşağıdaki metin bir tıbbi rapordan anonimleştirilmiştir. ■ işaretleri zaten maskelenmiş
alanlardır; bunları ve yanlarındaki ETİKET kelimelerini ("Hasta Adı", "TC Kimlik No", "Numune Kabul Tarihi",
"Asistan Dr", "Rapor Revizyon No", "Hasta / Velisi") RAPORLAMA — yalnızca açıkta kalan DEĞERİ raporla.

Metinde hâlâ açıkta kalan şu türde verileri bul:
- kişi adı veya soyadı (hasta, hasta yakını, doktor, patolog, hemşire — unvanlı veya unvansız)
- TC kimlik no, protokol/işlem/dosya/tescil numarası
- tam tarih (gün/ay/yıl) veya doğum tarihi
- telefon, e-posta, adres, web adresi
- hastane/kurum/laboratuvar/şehir/ilçe adı
- imza, diploma, sicil bilgisi

Bir ifadeyi yalnızca GERÇEKTEN bir kişiyi/kurumu tanımlıyorsa raporla. ■ işaretinin yanındaki kelime
tek başına şüphe nedeni DEĞİLDİR: "■ AML", "■ MOKSEFEN", "■ YAŞINDA", "■ MAKRO" gibi tanı/ilaç/etiket
kelimeleri raporlanmaz. Ama "■ ERDEM" gibi Türk soyadı/adı olan kelime raporlanır.

TIBBİ İÇERİK PII DEĞİLDİR: organ adları, tanılar (AML, KLL, GBM), ilaç/ticari adlar (Moksefen, Suprax, Rydapt),
test/antikor adları (NPM1, FLT3, TTF-1, Anti-HBc IgG), bölüm/servis adları (Tıbbi Onkoloji, Jin. Onk.),
tıbbi terimler (karsinom, blast, metastaz), ölçümler, yaş aralıkları, "Umut verici", "Deniz seviyesi"
gibi sıradan Türkçe ifadeler, SUT/tetkik kodları ("G101951-KRAS Geni Dizi Analizi", "C34.9", ICD kodları),
gen/mutasyon adları (JAK2, EXON12). Bunları raporlama. Sadece BİR KİŞİYİ veya KURUMU tanımlayan veriyi raporla.

Yanıt formatı (yalnızca JSON):
{{"findings": [{{"text": "<metindeki tam ifade>", "type": "<person|id|date|contact|institution|other>", "reason": "<kısa gerekçe>"}}]}}
Hiçbir şey bulamazsan: {{"findings": []}}

METİN:
<<<
{text}
>>>"""

MASK_TAG = re.compile(r"\[[A-Z_]+_SILINDI\]|\[YAS_ARALIGI:[^\]]*\]|\[COKLU_ALAN_SILINDI\]|\[KAPI_BULGUSU_\d+:[a-z]+\]")
BARE_TAG = re.compile(r"\b[A-Z_]*_SILINDI\b|\bYAS_ARALIGI\b|\bKAPI_BULGUSU_\d+\b")


@dataclass
class Finding:
    text: str
    type: str
    source: str
    reason: str = ""


@dataclass
class GateResult:
    passed: bool
    findings: list[Finding] = field(default_factory=list)
    layers_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "layers_run": self.layers_run,
            "findings": [f.__dict__ for f in self.findings],
            "errors": self.errors,
        }


_TR_FOLD = str.maketrans({"İ": "i", "I": "ı", "ı": "ı"})


def tr_fold(s: str) -> str:
    """Türkçe-duyarlı küçük harf: 'İ'.lower() Python'da 'i̇' (nokta + i) döner; İ→i, I→ı eşle."""
    return s.translate(_TR_FOLD).lower()


def _finding_is_real(f: Finding, text: str) -> bool:
    """LLM'in uydurduğu / zaten maskelenmiş bulguları ele.

    Bulgu bir maske etiketi İÇERİYORSA ("[DOKTOR_SILINDI] ERDEM") etiketi sıyırıp kalanı değerlendir;
    yalnızca etiketten ibaretse ele.
    """
    t = BARE_TAG.sub(" ", MASK_TAG.sub(" ", f.text)).strip(" \t\n.:;,-[]")
    if len(t) < 2:
        return False
    f.text = t
    ft, tt = tr_fold(t), tr_fold(text)
    if ft in tt:
        return True
    # Kelime bazlı tolerans: LLM "Dr. Ayşe Kaya" yazdı, metinde "AYŞE KAYA" var
    words = [w for w in re.split(r"[^\wçğıöşüÇĞİÖŞÜ]+", ft) if len(w) >= 3 and w not in {"dr", "prof", "uzm", "doç"}]
    return bool(words) and all(w in tt for w in words)


class LLMJudge:
    name = "llm_judge"

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    RETRIES = 2

    def check(self, text: str) -> list[Finding]:
        judge_view = MASK_TAG.sub("■", text)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_PROMPT.format(text=judge_view)},
        ]
        data = None
        last_err: Optional[Exception] = None
        for attempt in range(self.RETRIES):
            raw = self.llm.chat(messages, json_mode=True, max_tokens=2048)
            try:
                data = extract_json(raw)
                break
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                logger.warning("Yargıç geçersiz JSON döndürdü (deneme %d)", attempt + 1)
                messages = messages + [
                    {"role": "assistant", "content": raw[:500]},
                    {"role": "user", "content": 'Yanıtın geçerli JSON değildi. SADECE {"findings": [...]} biçiminde, kısa ve geçerli JSON ver.'},
                ]
        if data is None:
            raise last_err or ValueError("yargıç yanıtı çözümlenemedi")
        findings = data.get("findings")
        if findings is None:
            findings = []
        if not isinstance(findings, list):
            raise ValueError("yargıç 'findings' alanı liste değil")
        out = []
        for item in findings:
            if not isinstance(item, dict):
                continue
            f = Finding(
                text=str(item.get("text", "")),
                type=str(item.get("type", "other")),
                source=self.name,
                reason=str(item.get("reason", "")),
            )
            if _is_medical_code(f) or _is_label_only(f) or _self_refuting(f):
                continue  # SUT/ICD/tetkik kodu veya "Hasta / Velisi" gibi etiket — küçük modeller prompt'a rağmen raporluyor
            if _finding_is_real(f, text):
                out.append(f)
        return out


_SUT_CODE_RX = re.compile(r"^[A-Z]\d{5,7}(?:\.\d+)?\b")                 # G101951, G101860.3, P520030
_ICD_RX = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,2})?$")                         # C34.9, D37
_MEDICAL_WORDS = re.compile(r"\b(?:analiz|panel|dizileme|test|tetkik|mutasyon|gen(?:i|ler)?|sekans|pcr|fish|ihc|immün)\w*", re.IGNORECASE)


_LABEL_WORDS = {
    "hasta", "velisi", "veli", "doktor", "hekim", "imza", "kaşe", "kase", "adı", "adi", "soyadı", "soyadi",
    "isim", "ismi", "tarih", "tarihi", "onay", "onaylayan", "raporlayan", "patolog", "uzman", "uzmanı", "asistan",
    "servis", "isteyen", "gönderen", "gonderen", "klinik", "bölüm", "bolum", "no", "tc", "kimlik", "numarası",
    "numara", "randevu", "numune", "kabul", "saat", "saati", "protokol", "rapor", "revizyon", "revize", "diploma",
    "dipl", "tescil", "ve", "bilgileri", "bilgi", "doğum", "dogum", "yeri", "cinsiyet", "yaş", "yas", "barkod",
    "istem", "alım", "alim", "alındığı", "alindigi", "alınış", "alinis", "doku", "kurum", "adres", "adresi",
    "web", "tel", "telefon", "faks", "posta", "e-posta", "email", "sorumlu", "gönd", "gond", "dokt", "kaydeden",
    "hazırlayan", "hazirlayan", "form", "formu", "sayfa", "baba", "anne", "eş", "es", "yakını", "yakini",
}


def _is_label_only(f: Finding) -> bool:
    """Bulgu yalnızca etiket/rol kelimelerinden ibaretse (kişi adı yok) → PII değil."""
    words = [tr_fold(w).replace("ı", "i") for w in re.split(r"[^\wçğıöşüÇĞİÖŞÜ]+", f.text) if w]
    return bool(words) and all(w in {x.replace("ı", "i") for x in _LABEL_WORDS} or w in {"dr", "prof", "uzm"} for w in words)


_SELF_REFUTE_RX = re.compile(r"tıbbi|tibbi|medikal|pii değil|kişisel veri değil|değildir|not pii|ilaç|tanı|tetkik|ölçüm|yüzde|oran", re.IGNORECASE)


def _self_refuting(f: Finding) -> bool:
    """Model gerekçesinde 'tıbbi terim' / 'PII değil' diyorsa bulgu kendi kendini çürütüyordur."""
    return bool(f.reason) and bool(_SELF_REFUTE_RX.search(f.reason)) and not re.search(r"\d{7,}", f.text)


def _is_medical_code(f: Finding) -> bool:
    t = f.text.strip()
    if _SUT_CODE_RX.match(t) or _ICD_RX.match(t):
        return True
    return f.type in {"id", "other"} and bool(_MEDICAL_WORDS.search(t)) and not re.search(r"\d{7,}", t)


class GLiNERLayer:
    name = "gliner"
    LABELS = [
        "kişi adı", "hasta adı", "doktor adı", "TC kimlik numarası", "telefon numarası",
        "e-posta", "adres", "doğum tarihi", "tarih", "kurum adı", "hastane adı", "şehir",
    ]

    def __init__(self, model_name: str, threshold: float = 0.5):
        from gliner import GLiNER  # opsiyonel bağımlılık

        self.model = GLiNER.from_pretrained(model_name)  # HF_HUB_OFFLINE=1 → lokal cache zorunlu
        self.threshold = threshold

    def check(self, text: str) -> list[Finding]:
        ents = self.model.predict_entities(text, self.LABELS, threshold=self.threshold)
        out = []
        for e in ents:
            f = Finding(text=e["text"], type=e["label"], source=self.name, reason=f"score={e['score']:.2f}")
            if _finding_is_real(f, text):
                out.append(f)
        return out


class ResidualHeuristicLayer:
    """LLM'siz, deterministik sızıntı avcısı. Amaç: LLM yargıcın kaçırabileceği bariz kalıntılar."""

    name = "residual_heuristics"

    RULES = [
        ("id", re.compile(r"\b[1-9]\d{10}\b")),                                   # 11 haneli TC
        ("date", re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-](?:19|20)\d{2}\b")),       # tam tarih
        ("contact", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),                    # e-posta
        ("contact", re.compile(r"(?<!\d)0?\s?\(?5\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")),  # cep
        ("contact", re.compile(r"https?://\S+|www\.\S+")),
        # Maske etiketinden hemen sonra aynı satırda kalan BÜYÜK HARFLİ kelime(ler)
        ("person", re.compile(r"\[(?:DOKTOR|HASTA_ADI|ISIM|BABA_ADI|PII)_SILINDI\][ \t]+(?:(?:Dr|Prof|Uzm|Doç)\.?[ \t]*)?((?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]{2,}[ \t]*){1,3})", re.M)),   # ≥3 harf: 'Tc', 'Te' gibi OCR kırıntıları değil
    ]

    def __init__(self):
        try:
            from src.anonymizer import NAME_DICT  # isim sözlüğü: ad + soyad çifti
            self.first = NAME_DICT["first_names"]
            self.surnames = NAME_DICT["surnames"]
        except Exception:  # noqa: BLE001
            self.first, self.surnames = set(), set()
        self.pair = re.compile(r"\b([A-ZÇĞİÖŞÜ]{2,20})[ \t]+([A-ZÇĞİÖŞÜ]{2,20})\b")

    def check(self, text: str) -> list[Finding]:
        out = []
        _n = lambda w: tr_fold(w).replace("ı", "i")   # OCR "Isteyen"/"İsteyen"/"ISTEYEN" hepsi aynı
        stop = {_n(w) for w in CROSS_STOPLIST} | {
            "isteyen", "servis", "gönd", "gönderen", "yaş", "cinsiyet", "doku", "klinik", "numune",
            "tetkik", "tetkikler", "biyopsi", "materyal", "alındığı", "alinis", "alınış", "yeri",
        }
        for typ, rx in self.RULES:
            for m in rx.finditer(text):
                hit = (m.group(1) if m.groups() else m.group(0)).strip()
                if typ == "person":
                    first = _n(hit.split()[0]) if hit.split() else ""
                    if first in stop:
                        continue  # "[HASTA_ADI_SILINDI] Isteyen Servis" → etiket kelimesi, isim değil
                out.append(Finding(text=hit, type=typ, source=self.name, reason=f"kural:{typ}"))
        # Sözlük: BÜYÜK HARFLİ "AD SOYAD" çifti (ad sözlükte + soyad sözlükte)
        for m in self.pair.finditer(text):
            if m.group(1) in self.first and m.group(2) in self.surnames:
                out.append(Finding(text=m.group(0), type="person", source=self.name, reason="sözlük ad+soyad"))
        # "<bozuk etiket>: <Ad> <Soyad>" — ad sözlükte, soyad ne olursa olsun (OCR etiketi bozduğunda regex kaçırır)
        for m in re.finditer(r"^[^:\n]{1,30}:[ \t]*([A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]{1,19})[ \t]+([A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]{1,19})[ \t]*$", text, re.M):
            if m.group(1).upper() in self.first and _n(m.group(2)) not in stop:
                out.append(Finding(text=f"{m.group(1)} {m.group(2)}", type="person", source=self.name, reason="etiket sonrası sözlük adı"))
        return [f for f in out if _finding_is_real(f, text)]


TOKEN_RX = re.compile(r"[A-ZÇĞİÖŞÜ]{3,}|\d{5,}")
CROSS_STOPLIST = {
    # etiket/sık kelimeler — PII değil
    "ADI", "SOYADI", "HASTA", "DOKTOR", "TARIH", "TARİH", "TARİHİ", "SAAT", "TEL", "FAKS", "ADRES",
    "KURUM", "WEB", "POSTA", "MAIL", "KİMLİK", "KIMLIK", "TC", "NO", "SAĞLIK", "SAGLIK", "BAKANLIĞI",
    "BAKANLIGI", "HASTANESİ", "HASTANESI", "ŞEHİR", "SEHIR", "DEVLET", "ÜNİVERSİTESİ", "TIP",
    "FAKÜLTESİ", "LABORATUVARI", "LABORATUVAR", "PATOLOJİ", "PATOLOJI", "TIBBİ", "TIBBI", "RAPORU",
    "SONUÇ", "TETKİK", "FORMU", "TALEP", "NUMUNE", "BİLGİLERİ", "PROF", "UZM", "DOÇ", "ASİSTAN",
    "DİPL", "TESCİL", "ONAY", "KABUL", "İSTEM", "ISTEM", "RAPOR", "REVİZE", "PROTOKOL", "DOSYA",
}


def cross_ocr_candidates(alt_original: str, alt_anonymized: str) -> set[str]:
    """İkincil OCR'da regex katmanının SİLDİĞİ büyük harfli/sayısal tokenler = PII adayları."""
    before = set(TOKEN_RX.findall(alt_original))
    after = set(TOKEN_RX.findall(alt_anonymized))
    return {tok for tok in before - after if tok not in CROSS_STOPLIST}


class CrossOCRLayer:
    """Birincil OCR'ın atladığı ama ikincil OCR'ın gördüğü PII, nihai metinde var mı?"""

    name = "cross_ocr"

    def __init__(self, candidates: set[str]):
        self.candidates = candidates

    def check(self, text: str) -> list[Finding]:
        out = []
        from ..egress import candidate_patterns  # tam + token bazlı, Türkçe-katlanmış kalıplar

        folded = tr_fold(text)
        seen = set()
        for label, rx in candidate_patterns(self.candidates):
            if label in seen:
                continue
            if rx.search(folded):
                seen.add(label)
                out.append(Finding(text=label, type="person" if label.replace(" ", "").isalpha() else "id", source=self.name,
                                   reason="aday (ikincil OCR/NER) nihai metinde mevcut"))
        return out


class AnonymizationGate:
    def __init__(self, cfg: dict, llm: Optional[LLMBackend]):
        self.enabled = bool(cfg.get("enabled", True))
        self.fail_closed = bool(cfg.get("fail_closed", True))
        self.layers = []
        self.init_errors: list[str] = []
        if cfg.get("heuristics", True):
            self.layers.append(ResidualHeuristicLayer())
        if cfg.get("llm_judge", True) and llm is not None:
            self.layers.append(LLMJudge(llm))
        if cfg.get("gliner", False):
            try:
                self.layers.append(GLiNERLayer(cfg.get("gliner_model"), float(cfg.get("gliner_threshold", 0.5))))
            except Exception as e:  # noqa: BLE001
                logger.warning("GLiNER katmanı yüklenemedi, atlanıyor: %s", type(e).__name__)
                self.init_errors.append(f"gliner: {type(e).__name__}")

    def check(self, text: str, cross_candidates: Optional[set[str]] = None) -> GateResult:
        if not self.enabled:
            return GateResult(passed=True)
        result = GateResult(passed=True, errors=list(self.init_errors))
        layers = list(self.layers)
        if cross_candidates:
            layers.insert(0, CrossOCRLayer(cross_candidates))
        if not layers:
            result.errors.append("no_layers")
            result.passed = not self.fail_closed
            return result
        for layer in layers:
            try:
                findings = layer.check(text)
                result.layers_run.append(layer.name)
                result.findings.extend(findings)
            except Exception as e:  # noqa: BLE001
                logger.error("Kapı katmanı hatası (%s): %s", layer.name, type(e).__name__)  # mesaj yok: ham yanıt taşıyabilir
                result.errors.append(f"{layer.name}: {type(e).__name__}")
        if result.findings:
            result.passed = False
        elif result.errors and self.fail_closed:
            # Katman çalışamadıysa güvenli taraf: geçirme
            result.passed = False
        return result
