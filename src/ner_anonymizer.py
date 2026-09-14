"""NER Tabanlı Akıllı Anonimleştirici — Format bağımsız, lokal çalışır.

spaCy veya GLiNER gibi hafif NER modelleri ile kişisel verileri tespit eder.
GPU gerektirmez, CPU'da 1-3 saniyede çalışır. ~300 MB model boyutu.

Pipeline:
  1. NER modeli ile entity tespiti (PERSON, ORG, DATE, ID, PHONE...)
  2. Regex ile ek kalıp tespiti (TC, protokol no, e-posta...)  
  3. Tıbbi terimleri koruma listesiyle koruma
  4. Tespit edilen her entity'yi maskeleme

Bu yaklaşım regex-only'den çok daha güvenilir çünkü:
  - "HALİT BUĞDAYCI" → NER modeli isim olduğunu anlar (etiket olmasa bile)
  - "SONER DEMİRBAŞ" → isteyen doktor, imzalayan doktor fark etmez
  - Tablo içinde, satır sonunda, herhangi bir yerde olabilir
"""

import re
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationResult:
    """Anonimleştirme sonucu."""
    anonymized_text: str = ""
    entities_found: list[dict] = field(default_factory=list)
    fields_removed: list[str] = field(default_factory=list)
    original_length: int = 0
    anonymized_length: int = 0


class NERAnonymizer:
    """NER + Regex hibrit anonimleştirici."""

    # Tıbbi terimler — bunları ASLA silme
    MEDICAL_WHITELIST = {
        # Kanser tipleri
        "karsinom", "karsinoma", "adenokarsinom", "skuamöz", "glioblastom",
        "melanom", "sarkom", "lenfoma", "lösemi", "myelom", "seminom",
        "kistadenokarsinom", "mezotelyoma", "blastom", "papiller",
        # Organlar & dokular
        "bronkus", "akciğer", "serebrum", "beyin", "over", "uterus",
        "meme", "kolon", "rektum", "mide", "karaciğer", "böbrek",
        "pankreas", "prostat", "tiroid", "plevra", "periton", "omentum",
        # Patoloji terimleri
        "makroskopi", "makroskobik", "mikroskopi", "mikroskobik",
        "immünohistokimya", "immunohistokimya", "immünohistokimyasal",
        "histokimya", "sitoloji", "biyopsi", "rezeksiyon", "frozen",
        "neoplastik", "malign", "malignite", "benign", "metastaz",
        "metastatik", "diferansiye", "nekrotik", "invazyon",
        # Markerlar
        "p40", "ttf-1", "ttf1", "ck5", "ck5/6", "her2", "er", "pr",
        "ki67", "ki-67", "idh", "atrx", "gfap", "p53", "egfr",
        "alk", "ros-1", "pdl-1", "pd-l1", "jak2", "cd34",
        # Diğer tıbbi
        "eritroblastik", "granülositer", "megakaryosit", "maturasyon",
        "nötrofil", "lökosit", "trombosit", "eritrosit", "blast",
        "karyotip", "myeloid", "lenfoid", "polisitemi",
        "pnömonektomi", "lobektomi", "nefrektomi", "mastektomi",
        "atrofik", "reaktif", "hiperplazi", "displazi",
        # Bölüm/Birim adları (tıbbi)
        "patoloji", "onkoloji", "hematoloji", "radyoloji",
        "göğüs", "hastalıkları", "cerrahi", "dahiliye",
    }

    def __init__(self, use_ner: bool = True, ner_backend: str = "spacy"):
        """
        Args:
            use_ner: NER modeli kullan (False=sadece regex)
            ner_backend: "spacy" veya "gliner"
        """
        self.use_ner = use_ner
        self.ner_backend = ner_backend
        self._ner_model = None

        if use_ner:
            self._load_ner_model()

    def _load_ner_model(self):
        """NER modelini yükle."""
        if self.ner_backend == "spacy":
            try:
                import spacy
                # Türkçe model yoksa multilingual kullan
                try:
                    self._ner_model = spacy.load("xx_ent_wiki_sm")
                    logger.info("spaCy multilingual NER modeli yüklendi")
                except OSError:
                    try:
                        self._ner_model = spacy.load("en_core_web_sm")
                        logger.info("spaCy İngilizce NER modeli yüklendi (fallback)")
                    except OSError:
                        logger.warning(
                            "spaCy modeli bulunamadı. Yüklemek için:\n"
                            "  python -m spacy download xx_ent_wiki_sm\n"
                            "Sadece regex tabanlı anonimleştirme kullanılacak."
                        )
                        self.use_ner = False
            except ImportError:
                logger.warning("spaCy yüklü değil. pip install spacy")
                self.use_ner = False

        elif self.ner_backend == "gliner":
            try:
                from gliner import GLiNER
                self._ner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
                logger.info("GLiNER multilingual NER modeli yüklendi")
            except ImportError:
                logger.warning("GLiNER yüklü değil. pip install gliner")
                self.use_ner = False

    def anonymize(self, text: str) -> AnonymizationResult:
        """Ana anonimleştirme pipeline'ı."""
        result = AnonymizationResult(original_length=len(text))

        # Tüm tespit edilen entity'leri topla
        entities = []

        # 1. NER ile entity tespiti
        if self.use_ner:
            ner_entities = self._ner_detect(text)
            entities.extend(ner_entities)

        # 2. Regex ile ek tespit
        regex_entities = self._regex_detect(text)
        entities.extend(regex_entities)

        # 3. Tıbbi terimleri koruma — yanlış pozitifler çıkar
        entities = self._filter_medical_terms(entities)

        # 4. Örtüşen entity'leri birleştir
        entities = self._merge_overlapping(entities)

        # 5. Maskeleme (sondan başa doğru, indeks kayması olmasın)
        anonymized = text
        for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
            label = ent["label"]
            mask = f"[{label}_SILINDI]"
            anonymized = anonymized[:ent["start"]] + mask + anonymized[ent["end"]:]
            result.entities_found.append(ent)
            result.fields_removed.append(f"{label}: {ent['text'][:30]}")

        # 6. Son temizlik
        anonymized = self._final_cleanup(anonymized)

        result.anonymized_text = anonymized
        result.anonymized_length = len(anonymized)

        logger.info(
            f"Anonimleştirme: {len(entities)} entity bulundu, "
            f"{result.original_length} → {result.anonymized_length} karakter"
        )
        return result

    def _ner_detect(self, text: str) -> list[dict]:
        """NER modeli ile entity tespiti."""
        entities = []

        if self.ner_backend == "spacy" and self._ner_model:
            doc = self._ner_model(text)
            label_map = {
                "PER": "KISI", "PERSON": "KISI",
                "ORG": "KURUM", "ORGANIZATION": "KURUM",
                "LOC": "KONUM", "GPE": "KONUM", "LOCATION": "KONUM",
                "DATE": "TARIH", "TIME": "TARIH",
            }
            for ent in doc.ents:
                mapped_label = label_map.get(ent.label_, None)
                if mapped_label:
                    entities.append({
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "text": ent.text,
                        "label": mapped_label,
                        "source": "ner",
                        "confidence": 0.8,
                    })

        elif self.ner_backend == "gliner" and self._ner_model:
            labels = [
                "person name", "organization", "hospital",
                "date", "phone number", "identification number",
                "address", "doctor name", "email",
            ]
            gliner_entities = self._ner_model.predict_entities(text, labels)
            label_map = {
                "person name": "KISI", "doctor name": "KISI",
                "organization": "KURUM", "hospital": "KURUM",
                "date": "TARIH",
                "phone number": "ILETISIM",
                "identification number": "KIMLIK",
                "address": "ADRES",
                "email": "ILETISIM",
            }
            for ent in gliner_entities:
                mapped = label_map.get(ent["label"], "DIGER")
                entities.append({
                    "start": ent["start"],
                    "end": ent["end"],
                    "text": ent["text"],
                    "label": mapped,
                    "source": "ner",
                    "confidence": ent.get("score", 0.5),
                })

        return entities

    def _regex_detect(self, text: str) -> list[dict]:
        """Regex ile ek kalıp tespiti (NER'in kaçırdıklarını yakalar)."""
        entities = []

        patterns = [
            # TC Kimlik No
            (r'\b[1-9]\d{10}\b', "TC_KIMLIK"),
            (r'\b\d{2,3}\*{4,6}\d{2,3}\b', "TC_KIMLIK"),

            # Protokol / dosya numaraları
            (r'\b[A-Z]-\d{4,6}/\d{4}\b', "PROTOKOL"),
            (r'(?:Dosya|Başvuru|Protokol|Numune)\s*(?:No|no|NO)\s*[:.]?\s*[\w\-/]{4,15}', "REFERANS"),

            # Telefon
            (r'\(?\d{3,4}\)?\s*\d{3}\s*\d{2}\s*\d{2}', "TELEFON"),

            # E-posta
            (r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b', "EMAIL"),

            # URL
            (r'https?://\S+', "URL"),

            # Tarihler (GG.AA.YYYY veya GG/AA/YYYY)
            (r'\b\d{1,2}[./]\d{1,2}[./]\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b', "TARIH"),

            # Diploma tescil
            (r'(?:Dipl?\.?\s*Tescil\s*No\s*[:.]?\s*)\d+', "DIPLOMA"),

            # Doktor unvanları + isim
            (r'(?:PROF|Prof)\.\s*(?:DR|Dr)\.?\s*[A-ZÇĞİÖŞÜa-zçğıöşü\.\s]{3,40}', "DOKTOR"),
            (r'(?:DO[CÇcç]|Uzm|Op|Yrd)\.\s*(?:Do[cç]\.\s*)?(?:DR|Dr)\.?\s*[A-ZÇĞİÖŞÜa-zçğıöşü\.\s]{3,40}', "DOKTOR"),
            (r'\bDr\.?\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+){0,3}', "DOKTOR"),

            # Yapısal alan etiketleri ile isim
            (r'(?:Ad[ıi]?\s*Soyad[ıi]?\s*[:.]?\s*)[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü\s]{2,40}', "HASTA"),
            (r'(?:Hasta\s*(?:Ad[ıi]?)?\s*[:.]?\s*)[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü\s]{2,40}', "HASTA"),
            (r'(?:İsteyen\s*Doktor\s*[:.]?\s*)[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü\s]{2,40}', "DOKTOR"),

            # İmza blokları
            (r'Bu belge.*?imzalanmıştır\.?', "IMZA"),
            (r'BU BELGEN[İI]N G[İI]ZL[İI]L[İI][ĞG][İI].*?KULLANILAMAZ\.?', "IMZA"),

            # Adres
            (r'(?:Adres)\s*[:.]?\s*[^\n]{10,100}', "ADRES"),
        ]

        for pattern, label in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "label": label,
                    "source": "regex",
                    "confidence": 0.95,
                })

        return entities

    def _filter_medical_terms(self, entities: list[dict]) -> list[dict]:
        """Tıbbi terimlerin yanlışlıkla silinmesini engelle."""
        filtered = []
        for ent in entities:
            text_lower = ent["text"].lower().strip()
            words = text_lower.split()

            # Entity'nin tüm kelimeleri tıbbi whitelist'te mi?
            all_medical = all(
                any(med in w for med in self.MEDICAL_WHITELIST)
                for w in words
            )
            if all_medical and ent["source"] == "ner":
                logger.debug(f"Tıbbi terim korundu: {ent['text']}")
                continue

            # Tek kelimelik entity tıbbi whitelist'te mi?
            if len(words) == 1 and text_lower in self.MEDICAL_WHITELIST:
                continue

            filtered.append(ent)
        return filtered

    def _merge_overlapping(self, entities: list[dict]) -> list[dict]:
        """Örtüşen entity'leri birleştir."""
        if not entities:
            return []

        sorted_ents = sorted(entities, key=lambda e: (e["start"], -e["end"]))
        merged = [sorted_ents[0]]

        for ent in sorted_ents[1:]:
            last = merged[-1]
            if ent["start"] < last["end"]:
                # Örtüşüyor — daha uzun olanı tut
                if ent["end"] > last["end"]:
                    last["end"] = ent["end"]
                    last["text"] = last["text"] + "..." # Birleştirildi
            else:
                merged.append(ent)

        return merged

    def _final_cleanup(self, text: str) -> str:
        """Son temizlik."""
        # Ardışık maskeleri birleştir
        text = re.sub(r'(\[\w+_SILINDI\]\s*){3,}', '[COKLU_ALAN_SILINDI]\n', text)
        # Fazla boş satırlar
        text = re.sub(r'\n{4,}', '\n\n', text)
        return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF'ten metin çıkar (lokal, veri dışarı çıkmaz)."""
    import fitz
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            all_text.append(text)
    doc.close()
    return "\n\n".join(all_text)


def extract_text_from_image(image_path: str) -> str:
    """Görüntüden metin çıkar (lokal Tesseract OCR)."""
    import pytesseract
    from PIL import Image
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang="tur+eng")


# ════════════════════════════════════════════
# Komut satırı kullanımı
# ════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Kullanım: python ner_anonymizer.py <dosya.pdf|dosya.png|metin>")
        print("\nKurulum:")
        print("  pip install spacy PyMuPDF pytesseract Pillow")
        print("  python -m spacy download xx_ent_wiki_sm")
        sys.exit(1)

    input_path = sys.argv[1]

    # Metin çıkar
    if input_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(input_path)
    elif input_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
        text = extract_text_from_image(input_path)
    else:
        text = Path(input_path).read_text(encoding="utf-8")

    print(f"Çıkarılan metin ({len(text)} karakter):")
    print(text[:200] + "...")
    print()

    # Anonimleştir
    anon = NERAnonymizer(use_ner=True, ner_backend="spacy")
    result = anon.anonymize(text)

    print("=" * 60)
    print("ANONİMLEŞTİRİLMİŞ METİN:")
    print("=" * 60)
    print(result.anonymized_text)
    print()
    print(f"Bulunan entity sayısı: {len(result.entities_found)}")
    print(f"Silinen alanlar: {result.fields_removed}")
