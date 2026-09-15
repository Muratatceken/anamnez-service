"""NER tabanlı anonimleştirici — GLiNER (Türkçe PII modeli), CPU'da çalışır.

Regex'in yapısal etiketlere bağımlı olduğu yerde NER, etiketi bozuk/eksik olan veya serbest metindeki
isimleri, şehirleri, kurumları yakalar. Regex katmanı NER'den SONRA çalışır (tarih/TC/tescil gibi
yapısal alanlar için deterministik taban). Kapı, NER'in bulduğu her dizeyi nihai çıktıda tekrar arar.

Model: neondijital/neonredact-tr-model (GLiNER, mDeBERTa, Apache-2.0). Offline: HF cache veya `model_path`.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# GLiNER etiketi → maske
LABEL_TAGS = {
    "kişi adı": "[ISIM_SILINDI]",
    "hasta adı": "[HASTA_ADI_SILINDI]",
    "doktor adı": "[DOKTOR_SILINDI]",
    "TC kimlik numarası": "[TC_KIMLIK_SILINDI]",
    "telefon numarası": "[ILETISIM_SILINDI]",
    "e-posta": "[EMAIL_SILINDI]",
    "adres": "[ADRES_SILINDI]",
    "tarih": "[TARIH_SILINDI]",
    "doğum tarihi": "[DOGUM_TARIHI_SILINDI]",
    "hastane": "[KURUM_SILINDI]",
    "kurum": "[KURUM_SILINDI]",
    "şehir": "[SEHIR_SILINDI]",
    "protokol numarası": "[REFERANS_SILINDI]",
    "diploma numarası": "[DIPLOMA_SILINDI]",
    "dosya numarası": "[REFERANS_SILINDI]",
}
DEFAULT_LABELS = list(LABEL_TAGS)

# NER'in tıbbi terimi yanlışlıkla isim/kurum sanmasına karşı küçük koruma (ölçümde 0 FP; yine de)
MEDICAL_GUARD = re.compile(
    r"^(?:ki-?67|p53|p40|ttf-?1|her2|er|pr|cd\d+|idh|atrx|gfap|npm1|flt3|jak2|kras|egfr|alk|ros1|pdl-?1|brca\d?|"
    r"msi|psa|afp|ca-?\d+|hbv|hcv|hpv|aml|kll|gbm|rcc|hsil|figo|bethesda|gleason|breslow|isup|tirads|dsö|who)$",
    re.IGNORECASE,
)

_TR_LOWER = {"İ": "i", "I": "ı"}


def _tr_lower_char(ch: str) -> str:
    """Uzunluk koruyan Türkçe küçük harf ('İ'.lower() Python'da 2 karakter döner!)."""
    if ch in _TR_LOWER:
        return _TR_LOWER[ch]
    low = ch.lower()
    return low if len(low) == 1 else ch


def _tr_fold(s: str) -> str:
    return "".join(_tr_lower_char(c) for c in s)


def _load_medical_vocab() -> set[str]:
    """Tıbbi kelime hazinesi: config/categories.json anahtar kelimeleri + patoloji/klinik çekirdek sözlük.
    NER bir span'ın herhangi bir kelimesi burada geçiyorsa (ör. 'Polisitemi VERA', 'BERRAK hücreli') o span PII değildir."""
    import json
    from pathlib import Path

    words = set()
    core = """
    hücreli hücre karsinom karsinomu karsinoma adenokarsinom adenokarsinomu sarkom sarkomu lenfoma lenfomu lösemi
    melanom melanomu blastom glioblastom nöroblastom seminom teratom vera polisitemi berrak papiller foliküler
    medüller invaziv duktal lobüler skuamöz ürotelyal seröz müsinöz nöroendokrin metastaz metastatik metastazı
    tümör tümörü tümoral kitle kitlesi lezyon lezyonu nodül nodülü biyopsi biyopsisi rezeksiyon rezeksiyonu
    eksizyon eksizyonu materyal materyali grade derece evre pozitif negatif mutasyon mutant wild tip tipi
    hodgkin non-hodgkin diffüz hücreli myeloid lenfoid akut kronik anemi trombositoz fibrozis
    hipersellüler hiposellüler displazi hiperplazi atipi atipik benign malign selim habis in situ
    invazyon perinöral lenfovasküler vasküler cerrahi sınır sınırı nekroz nekrotik mitoz mitotik
    immünohistokimya immünhistokimya boyama boyanma ekspresyon pozitiflik negatiflik oran oranında
    akciğer meme mide kolon rektum prostat mesane böbrek karaciğer pankreas over uterus serviks tiroid
    testis beyin deri kemik ilik iliği yumuşak doku plevra özofagus safra timus lenf nod nodu bezi
    lob lobu segment kadran medial lateral proksimal distal
    kemoterapi radyoterapi indüksiyon konsolidasyon nakil tedavi tedavisi takip kontrol sevk
    bethesda gleason breslow fuhrman isup figo tirads who dsö
    """.split()
    words.update(core)
    try:
        hints = json.loads((Path(__file__).resolve().parent.parent.parent / "config" / "categories.json").read_text(encoding="utf-8"))["keyword_hints"]
        for terms in hints.values():
            for t in terms:
                for w in t.replace("-", " ").split():
                    if len(w) >= 3:
                        words.add(_tr_fold(w))
    except Exception:  # noqa: BLE001
        pass
    return {_tr_fold(w) for w in words}


MEDICAL_VOCAB = _load_medical_vocab()


def looks_medical(span: str) -> bool:
    """Span PII değil, tıbbi ifade mi? Muhafazakâr: alfabetik (≥3 harf) tokenlerin HEPSİ tıbbi sözlükteyse.
    'Polisitemi Vera' → (polisitemi, vera) hepsi tıbbi → True;  'Selim Kaya' → kaya tıbbi değil → False."""
    if MEDICAL_GUARD.match(span.strip()):
        return True
    toks = [_tr_fold(t) for t in re.split(r"[^\wçğıöşüÇĞİÖŞÜ]+", span) if len(t) >= 3 and t.isalpha()]
    return bool(toks) and all(t in MEDICAL_VOCAB for t in toks)


def title_case_preserving_length(text: str) -> str:
    """BÜYÜK HARFLİ metni kelime başları büyük kalacak şekilde küçült; her karakter 1:1 eşlenir,
    böylece bulunan span ofsetleri orijinal metinde de geçerlidir."""
    out = []
    prev_alpha = False
    for ch in text:
        if ch.isalpha():
            out.append(ch if not prev_alpha else _tr_lower_char(ch))
            prev_alpha = True
        else:
            out.append(ch)
            prev_alpha = False
    return "".join(out)


@dataclass
class NERSpan:
    start: int
    end: int
    text: str
    label: str
    score: float


@dataclass
class NERResult:
    text: str
    spans: list[NERSpan] = field(default_factory=list)
    fields_removed: list[str] = field(default_factory=list)

    @property
    def candidates(self) -> set[str]:
        """Kapı için: bulunan her ham dize (çıktıda tekrar aranır)."""
        return {s.text.strip() for s in self.spans if len(s.text.strip()) >= 3}


class NERAnonymizer:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.model_name = cfg.get("model", "neondijital/neonredact-tr-model")
        self.model_path = cfg.get("model_path")            # offline dizin (bundle)
        self.threshold = float(cfg.get("threshold", 0.3))
        self.labels = list(cfg.get("labels") or DEFAULT_LABELS)
        self.chunk_tokens = int(cfg.get("chunk_tokens", 250))  # GLiNER max_len=384 kelime-token; pay bırak
        self.case_pass = bool(cfg.get("case_pass", True))    # büyük harfli metin için ikinci geçiş
        # Başlık geçişi: formun üst bölümünde (etiket-değer alanı) DAR etiket kümesiyle ek tahmin.
        # GLiNER'ın tahmini etiket kümesine bağlıdır; bozuk OCR satırlarında ("KDI SOYRDE BÜŞEK YAĞMURDERELİ")
        # dar küme isim/şehiri yakalarken tam küme kaçırabiliyor.
        self.header_pass = bool(cfg.get("header_pass", True))
        self.header_lines = int(cfg.get("header_lines", 14))
        self.header_labels = list(cfg.get("header_labels") or ["kişi adı", "hasta adı", "doktor adı"])
        self._model = None
        self._lock = __import__("threading").Lock()

    # ── model ──
    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from gliner import GLiNER  # ağır import: tembel

                    src = self.model_path or self.model_name
                    logger.info("GLiNER yükleniyor: %s", src)
                    self._model = GLiNER.from_pretrained(src)
        return self._model

    def health(self) -> dict:
        try:
            self._load()
            return {"ok": True, "engine": "gliner", "model": self.model_path or self.model_name}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "engine": "gliner", "model": self.model_name, "error": type(e).__name__}

    # ── parçalama: GLiNER'ın kelime-token sınırı (max_len=384) — sınır aşılırsa fazlası SESSİZCE atılır,
    #    bu yüzden token sayısıyla (karakterle değil) parçalanır; ofsetler korunur ──
    _TOK = re.compile(r"\w+(?:[-_]\w+)*|\S")   # GLiNER'ın kendi kelime ayırıcısıyla aynı

    def _pieces(self, text: str):
        """(ofset, parça) — satır satır; token sayısı büyük satırlar boşluktan bölünür."""
        pos = 0
        for line in text.splitlines(keepends=True):
            if len(self._TOK.findall(line)) <= self.chunk_tokens:
                yield pos, line
            else:
                sub_start = 0
                for m in re.finditer(r"\S+\s*", line):
                    if len(self._TOK.findall(line[sub_start:m.end()])) > self.chunk_tokens:
                        yield pos + sub_start, line[sub_start:m.start()]
                        sub_start = m.start()
                yield pos + sub_start, line[sub_start:]
            pos += len(line)

    def _chunks(self, text: str) -> list[tuple[int, str]]:
        chunks, start, buf, ntok = [], 0, [], 0
        for off, piece in self._pieces(text):
            t = len(self._TOK.findall(piece))
            if ntok + t > self.chunk_tokens and buf:
                chunks.append((start, "".join(buf)))
                start, buf, ntok = off, [], 0
            if not buf:
                start = off
            buf.append(piece)
            ntok += t
        if buf:
            chunks.append((start, "".join(buf)))
        return chunks or [(0, text)]

    def _predict(self, text: str) -> list[NERSpan]:
        model = self._load()
        spans: list[NERSpan] = []
        variants = [text]
        if self.case_pass and sum(ch.isupper() for ch in text) > 0.5 * max(1, sum(ch.isalpha() for ch in text)):
            variants.append(title_case_preserving_length(text))
        passes: list[tuple[str, int, list[str]]] = []   # (metin, ofset, etiketler)
        for variant in variants:
            for offset, chunk in self._chunks(variant):
                passes.append((chunk, offset, self.labels))
            if self.header_pass:
                head = "".join(variant.splitlines(keepends=True)[: self.header_lines])
                for offset, chunk in self._chunks(head):
                    passes.append((chunk, offset, self.header_labels))
        for chunk, offset, labels in passes:
            for e in model.predict_entities(chunk, labels, threshold=self.threshold):
                s, en = offset + e["start"], offset + e["end"]
                raw = text[s:en]
                if not raw.strip() or looks_medical(raw):
                    continue
                spans.append(NERSpan(s, en, raw, e["label"], float(e["score"])))
        return spans

    @staticmethod
    def _merge(spans: list[NERSpan], text: str) -> list[NERSpan]:
        """Örtüşen span'ları birleştir; etiket = daha yüksek skorlu (eşitse uzun) span'ınki; metin kaynaktan."""
        out: list[NERSpan] = []
        for sp in sorted(spans, key=lambda x: (x.start, -(x.end - x.start), -x.score)):
            if out and sp.start < out[-1].end:
                last = out[-1]
                end = max(last.end, sp.end)
                label = sp.label if (sp.score, sp.end - sp.start) > (last.score, last.end - last.start) else last.label
                out[-1] = NERSpan(last.start, end, text[last.start:end], label, max(last.score, sp.score))
                continue
            out.append(NERSpan(sp.start, sp.end, text[sp.start:sp.end], sp.label, sp.score))
        return out

    def anonymize(self, text: str) -> NERResult:
        spans = self._merge(self._predict(text), text)
        res = NERResult(text=text)
        if not spans:
            return res
        out, last = [], 0
        for sp in spans:
            out.append(text[last:sp.start])
            out.append(LABEL_TAGS.get(sp.label, "[PII_SILINDI]"))
            last = sp.end
            res.fields_removed.append(f"NER: {sp.label}")
        out.append(text[last:])
        res.text = "".join(out)
        res.spans = spans
        return res
