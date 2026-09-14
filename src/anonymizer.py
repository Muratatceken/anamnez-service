"""Anonimleştirme Modülü v3 — İsim sözlüğü tabanlı, bölüm farkındalıklı.

Strateji: Beyaz liste yaklaşımı.
  - Tıbbi terimleri silmek yerine, sadece Türk isim sözlüğündeki isimleri sil
  - TANI/MAKROSKOPİ/MİKROSKOPİ bölümlerinde agresif silme yapma
  - Yapısal etiketler (Adı Soyadı:, İsteyen Doktor:) ile PII yakala
  - Kelime sınırlarına (\b) dikkat et — kelime ortasından kesme yapma
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationReport:
    """Anonimleştirme sonuç raporu."""
    original_length: int = 0
    anonymized_length: int = 0
    fields_removed: list[str] = field(default_factory=list)
    fields_generalized: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalize_turkish(text: str) -> str:
    """Türkçe özel karakterleri ASCII karşılıklarına çevir (OCR toleransı için)."""
    tr_map = str.maketrans("ÇĞİÖŞÜçğıöşü", "CGIOSUcgiosu")
    return text.translate(tr_map)


def _load_name_dictionary() -> dict:
    """Türk isim sözlüğünü yükle."""
    dict_path = Path(__file__).parent.parent / "config" / "turkish_names.json"
    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Tüm isimleri büyük harfe çevir ve set yap
        # OCR-tolerant: Türkçe karakter varyasyonlarını da ekle
        all_first = set()
        for name in data.get("first_names_male", []) + data.get("first_names_female", []):
            n = name.upper().strip()
            all_first.add(n)
            all_first.add(_normalize_turkish(n))
        all_surnames = set()
        for name in data.get("surnames", []):
            n = name.upper().strip()
            all_surnames.add(n)
            all_surnames.add(_normalize_turkish(n))
        suffixes = [s.upper() for s in data.get("surname_suffixes", [])]
        return {
            "first_names": all_first,
            "surnames": all_surnames,
            "surname_suffixes": suffixes,
            "all_names": all_first | all_surnames,
        }
    except Exception as e:
        logger.warning(f"İsim sözlüğü yüklenemedi: {e}")
        return {"first_names": set(), "surnames": set(), "surname_suffixes": [], "all_names": set()}


# Sözlüğü modül yüklenirken bir kez oku
NAME_DICT = _load_name_dictionary()

# İsim sözlüğünde olup tıbbi/günlük anlamı da olan kelimeler: bunlar yalnızca
# diğer kelime de sözlükte AD veya SOYAD olarak varsa isim sayılır (soyadı-eki kuralı uygulanmaz)
AMBIGUOUS_NAME_WORDS = {
    "SELİM", "SELIM", "ORAL", "DEMİR", "DEMIR", "UMUT", "TUBA", "İNCE", "INCE", "UZUN", "ATEŞ", "ATES",
    "BAŞ", "BAS", "KARA", "SARI", "GENÇ", "GENC", "TEMEL", "DENİZ", "DENIZ", "CAN", "AYDIN", "GÜL", "GUL",
    "YAŞAR", "YASAR", "DOĞAN", "DOGAN", "AKSOY", "KURT", "ŞAHİN", "SAHIN", "BARIŞ", "BARIS", "SEVİM", "SEVIM",
    "ÖZ", "OZ", "ACAR", "SAĞLAM", "SAGLAM", "YILDIZ", "GÜNEŞ", "GUNES", "TAŞ", "TAS", "KAYA", "TOPRAK",
    "İLERİ", "ILERI", "SOL", "SAĞ", "SAG", "ORTA", "BÜYÜK", "BUYUK", "KÜÇÜK", "KUCUK", "YENİ", "YENI",
}

# 81 il — dolaylı tanımlayıcı (ör. "AKSARAYDAN yönlendirilen hasta")
TURKISH_PROVINCES = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Aksaray", "Amasya", "Ankara", "Antalya", "Ardahan",
    "Artvin", "Aydın", "Balıkesir", "Bartın", "Batman", "Bayburt", "Bilecik", "Bingöl", "Bitlis", "Bolu",
    "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır", "Düzce", "Edirne", "Elazığ",
    "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", "Hakkari", "Hatay", "Iğdır",
    "Isparta", "İstanbul", "Istanbul", "İzmir", "Izmir", "Kahramanmaraş", "Karabük", "Karaman", "Kars", "Kastamonu",
    "Kayseri", "Kilis", "Kırıkkale", "Kırklareli", "Kırşehir", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa",
    "Mardin", "Mersin", "Muğla", "Muş", "Nevşehir", "Niğde", "Ordu", "Osmaniye", "Rize", "Sakarya", "Samsun",
    "Siirt", "Sinop", "Sivas", "Şanlıurfa", "Şırnak", "Tekirdağ", "Tokat", "Trabzon", "Tunceli", "Uşak", "Van",
    "Yalova", "Yozgat", "Zonguldak",
]
def _prov_forms(x: str) -> set[str]:
    """Konya, KONYA, Konya (ASCII), KONYA (ASCII). Küçük harf biçimi ("istanbul" — OCR) yalnızca
    6+ harfli, gündelik anlamı olmayan illerde; 'van', 'ordu', 'muş', 'kars' gibi kısa/çok anlamlılar EŞLEŞMEZ."""
    forms = {x, _normalize_turkish(x)}
    up = x.upper()
    forms |= {up, _normalize_turkish(up)}
    low = x.lower().replace("i̇", "i")
    if len(x) >= 6 and x not in {"Batman", "Burdur", "Bartın", "Düzce", "Bolu"}:
        forms |= {low, _normalize_turkish(low)}
    return forms


_PROV_ALT = "|".join(sorted({re.escape(f) for x in TURKISH_PROVINCES for f in _prov_forms(x)}, key=len, reverse=True))
PROVINCE_RX = re.compile(
    rf"(?<![A-Za-zÇĞİÖŞÜçğıöşü])(?:{_PROV_ALT})(?:['’]?(?:DAN|DEN|TAN|TEN|DA|DE|TA|TE|NIN|NİN|NUN|NÜN|YA|YE|LI|Lİ|LU|LÜ|dan|den|tan|ten|da|de|ta|te|nın|nin|nun|nün|ya|ye|lı|li|lu|lü))?(?![A-Za-zÇĞİÖŞÜçğıöşü])",
)

# Tıbbi bölüm başlıkları — bu bölümlerde agresif isim tespiti YAPILMAZ
MEDICAL_SECTION_HEADERS = [
    "MAKROSKOBİK", "MAKROSKOPİ", "MAKROSKOPI", "MAKROSK",
    "MİKROSKOBİ", "MİKROSKOPİ", "MIKROSKOPI",
    "TANI", "TANILAR",
    "EPİKRİZ",
    "İMMÜNOHİSTOKİMYA", "İMMUNHİSTOKİMYA", "HİSTOKİMYA",
    "KLİNİK BİLGİ", "KLİNİK ÖYKÜ", "KLİNİK ÖZET",
    "NUMUNE BİLGİLERİ",
    "LOKALIZASYON",
    "HISTOLOJI",
]


class ReportAnonymizer:
    """Patoloji raporlarından kişisel verileri temizler — isim sözlüğü tabanlı."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.generalize_age = self.config.get("generalize_age", True)
        self.remove_dates = self.config.get("remove_dates", True)
        self.remove_institutions = self.config.get("remove_institutions", True)

    def anonymize(self, text: str) -> tuple[str, AnonymizationReport]:
        report = AnonymizationReport(original_length=len(text))

        # Katman 1: TC Kimlik
        text = self._remove_tc_kimlik(text, report)

        # Katman 2: Referans numaraları
        text = self._remove_reference_numbers(text, report)

        # Katman 3: Yapısal etiketli alanlar (Adı Soyadı:, İsteyen Doktor:, vb.)
        text = self._remove_labeled_fields(text, report)

        # Katman 3b: Diploma/tescil numaraları — doktor adı katmanından ÖNCE
        # (aksi halde unvanlı doktor regex'i "Dipl" kelimesini yutup tescil no'yu açıkta bırakıyor)
        text = self._remove_diploma_numbers(text, report)

        # Katman 4: Unvanlı doktor adları (Prof. Dr., Uzm. Dr., vb.)
        text = self._remove_doctor_names(text, report)

        # Katman 5: Tarihler
        if self.remove_dates:
            text = self._remove_dates(text, report)

        # Katman 6: Yaş genelleştirme
        if self.generalize_age:
            text = self._generalize_age(text, report)

        # Katman 7: Kurum/hastane adları
        if self.remove_institutions:
            text = self._remove_institutions(text, report)

        # Katman 8: İletişim bilgileri
        text = self._remove_contact_info(text, report)

        # Katman 10: İmza blokları
        text = self._remove_signature_blocks(text, report)

        # Katman 11: İsim sözlüğü tabanlı isim tespiti (bölüm farkındalıklı)
        text = self._remove_names_by_dictionary(text, report)

        # Katman 12: İşlem/protokol numaraları
        text = self._remove_numeric_identifiers(text, report)

        # Temizlik
        text = self._cleanup(text)

        report.anonymized_length = len(text)
        logger.info(
            f"Anonimleştirme tamamlandı: {report.original_length} → {report.anonymized_length} karakter, "
            f"{len(report.fields_removed)} alan silindi"
        )
        return text, report

    # ════════════════════════════════════════════
    # Bölüm farkındalığı
    # ════════════════════════════════════════════

    def _find_medical_sections(self, text: str) -> list[tuple[int, int]]:
        """Tıbbi bölümlerin başlangıç-bitiş pozisyonlarını bul.
        Bu bölümlerde agresif isim tespiti yapılmaz."""
        sections = []
        lines = text.split('\n')
        in_section = False
        section_start = 0
        pos = 0

        for line in lines:
            line_upper = line.strip().upper()
            # Bölüm başlığı mı?
            if any(line_upper.startswith(h) for h in MEDICAL_SECTION_HEADERS):
                if not in_section:
                    section_start = pos
                    in_section = True
            # Boş satır veya yeni yapısal alan → bölüm sonu olabilir
            # Ama tıbbi bölümler genelde birden fazla satır, devam edelim
            pos += len(line) + 1  # +1 for \n

        # Son bölüm dosya sonuna kadar
        if in_section:
            sections.append((section_start, len(text)))

        # Daha basit yaklaşım: header'dan sonraki satırları da dahil et
        # Her header'ı bul ve sonraki header'a veya dosya sonuna kadar section say
        sections = []
        # Bölüm sonu: imza/onay/iletişim/etiketli PII satırı
        section_end_rx = re.compile(
            r'^[ \t]*(?:(?:UZM|PROF|DO[ÇC]|OP|YRD|DR|AS[İI]STAN)\.?\s*(?:DR|DOÇ)?\.?\s+[A-ZÇĞİÖŞÜ]|'
            r'Patoloji\s+Uzman|Dipl?[.\-\s]*Tescil|Raporlayan|Onaylayan|İmza|Imza|Kurum|Tel\b|Adres|E-?posta|'
            r'Hasta\s+Ad|Ad[ıi]\s+Soyad|[İI]steyen\s+Doktor|G[öo]nd(?:eren)?\.?\s*Dokt|TC\s*Kim)',
            re.IGNORECASE | re.MULTILINE,
        )
        for header in MEDICAL_SECTION_HEADERS:
            for m in re.finditer(rf'^[ \t]*{re.escape(header)}\b', text, re.IGNORECASE | re.MULTILINE):
                start = m.start()
                e = section_end_rx.search(text, m.end())
                end = e.start() if e else len(text)
                sections.append((start, end))

        return sections

    def _is_in_medical_section(self, pos: int, sections: list[tuple[int, int]]) -> bool:
        """Pozisyonun tıbbi bölüm içinde olup olmadığını kontrol et."""
        return any(start <= pos < end for start, end in sections)

    # ════════════════════════════════════════════
    # Katman 1: TC Kimlik
    # ════════════════════════════════════════════

    def _remove_tc_kimlik(self, text: str, report: AnonymizationReport) -> str:
        # Etiketli TC (OCR-tolerant)
        pattern = r'(?:T\.?C\.?\s*(?:Kimli[kğ]|Kimt[ıi]k|K\.?)\s*(?:No|Numaras[ıi])?\s*[:.]?\s*)[\d\*]{7,11}'
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, "[TC_KIMLIK_SILINDI]", text, flags=re.IGNORECASE)
            report.fields_removed.append("TC Kimlik No")

        # Bağımsız 11 haneli (TC olabilir)
        text = re.sub(r'\b[1-9]\d{10}\b', '[KIMLIK_SILINDI]', text)

        # Maskeli TC: 12345678***, 25******976 (satır sonunda \b tutmaz → lookahead)
        text = re.sub(r'\b\d{2,8}\*{2,8}\d{0,5}(?!\d)', '[TC_KIMLIK_SILINDI]', text)

        return text

    # ════════════════════════════════════════════
    # Katman 2: Referans numaraları
    # ════════════════════════════════════════════

    def _remove_reference_numbers(self, text: str, report: AnonymizationReport) -> str:
        patterns = [
            (r'(?:Dosya\s*[Nn]o\s*[:.]?\s*)[\w\-/]+', "Dosya No"),
            (r'(?:Ba[şs]vuru\s*[Nn]o\s*[:.]?\s*)[\w\-/]+', "Başvuru No"),
            (r'(?:Patoloji\s*(?:Protokol|No)\s*(?:No)?\s*[:.]?\s*)[\w\-/]+', "Patoloji No"),
            (r'(?:Protokol\s*[\\\/]?\s*[Nn][eo]\s*[:.]?\s*)[\w\-/]+', "Protokol No"),
            (r'(?:Biyopsi\s*[Nn]o\s*[:.]?\s*)[\w\-/]+', "Biyopsi No"),
            (r'(?:Ruhsat\s*[Nn]o\s*[:.]?\s*)[\w\-/]+', "Ruhsat No"),
            (r'(?:Rapor[ \t]*[Nn]o\b[ \t]*[:.]?[ \t]*)[\w\-/ \t]{3,20}', "Rapor No"),
            (r'(?:Defter\s*[Nn]o\s*[:.]?\s*)[\w\-/]*', "Defter No"),
            (r'\b[A-Z]-\d{4,7}/?\d{0,4}\b', "Protokol No (format)"),
        ]
        for pattern, name in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, "[REFERANS_SILINDI]", text, flags=re.IGNORECASE)
                report.fields_removed.append(name)
        return text

    # ════════════════════════════════════════════
    # Katman 3: Yapısal etiketli PII alanları
    # ════════════════════════════════════════════

    def _remove_labeled_fields(self, text: str, report: AnonymizationReport) -> str:
        """Yapısal alan etiketlerinden sonra gelen değerleri sil.
        Kelime sınırlarına dikkat eder — ortadan kesmez."""

        changes_made = True
        max_iterations = 20
        iteration = 0

        while changes_made and iteration < max_iterations:
            changes_made = False
            iteration += 1

            # Hasta adı etiketleri
            name_patterns = [
                # "Adı Soyadı : MEHMET YILMAZ" — satır sonuna veya sonraki etikete kadar
                r'(?:Ad[ıi]?\s*Soyad[ıi]?(?:[ \t]*[:=.]){0,2}[ \t]*[>©o]?[ \t]*)([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:[ \t]+(?![A-ZÇĞİÖŞÜa-zçğıöşü]+[ \t]*[:=»>])(?!(?:[İIi]steyen|G[öo]nd|Servis|Doktor|Dokt|TC|Ya[şs]|Cinsiyet|Do[gğ]um|Baba|Anne|Protokol|Rapor|Biyopsi|Dosya|Kabul|Numune|Tarih|Klinik|Tetkik|Patoloji|Hasta)\b)[A-ZÇĞİÖŞÜa-zçğıöşü]+){0,3})',
                # "Hasta Adı Soyadı o : CANER AKSU"
                r'(?:Hasta\s+Ad[ıi]\s*(?:Soyad[ıi]?)?(?:[ \t]*[:=.]){0,2}[ \t]*[>©o]?[ \t]*)([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:[ \t]+(?![A-ZÇĞİÖŞÜa-zçğıöşü]+[ \t]*[:=»>])(?!(?:[İIi]steyen|G[öo]nd|Servis|Doktor|Dokt|TC|Ya[şs]|Cinsiyet|Do[gğ]um|Baba|Anne|Protokol|Rapor|Biyopsi|Dosya|Kabul|Numune|Tarih|Klinik|Tetkik|Patoloji|Hasta)\b)[A-ZÇĞİÖŞÜa-zçğıöşü]+){0,3})',
                # "Adt- Soyad: > Sevim Kaya"
                r'(?:Adt?-?\s*Soyad?(?:[ \t]*[:=.]){0,2}[ \t]*[>©o]?[ \t]*)([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:[ \t]+(?![A-ZÇĞİÖŞÜa-zçğıöşü]+[ \t]*[:=»>])(?!(?:[İIi]steyen|G[öo]nd|Servis|Doktor|Dokt|TC|Ya[şs]|Cinsiyet|Do[gğ]um|Baba|Anne|Protokol|Rapor|Biyopsi|Dosya|Kabul|Numune|Tarih|Klinik|Tetkik|Patoloji|Hasta)\b)[A-ZÇĞİÖŞÜa-zçğıöşü]+){0,3})',
                # "Ad Ad Soyad: OK KÜBRA YEL"
                r'(?:Ad\s+(?:Ad\s+)?Soyad\s*[:=.]?\s*)([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:[ \t]+(?![A-ZÇĞİÖŞÜa-zçğıöşü]+[ \t]*[:=»>])(?!(?:[İIi]steyen|G[öo]nd|Servis|Doktor|Dokt|TC|Ya[şs]|Cinsiyet|Do[gğ]um|Baba|Anne|Protokol|Rapor|Biyopsi|Dosya|Kabul|Numune|Tarih|Klinik|Tetkik|Patoloji|Hasta)\b)[A-ZÇĞİÖŞÜa-zçğıöşü]+){0,3})',
                # "Adı-Soyadı:", "Adı ve Soyadı:", "İsim:", "İsmi:", tek başına "Adı:" / "Soyadı:"
                r'(?:^|[ \t|])(?:Ad[ıi]?[ \t]*[-/,]?[ \t]*(?:ve[ \t]+)?Soyad[ıi]?|[İI]s(?:im|mi)|Ad[ıi]|Soyad[ıi])[ \t]*[:=][ \t]*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:[ \t]+(?![A-ZÇĞİÖŞÜa-zçğıöşü]+[ \t]*[:=»>])(?!(?:[İIi]steyen|G[öo]nd|Servis|Doktor|Dokt|TC|Ya[şs]|Cinsiyet|Do[gğ]um|Baba|Anne|Protokol|Rapor|Biyopsi|Dosya|Kabul|Numune|Tarih|Klinik|Tetkik|Patoloji|Hasta)\b)[A-ZÇĞİÖŞÜa-zçğıöşü]+){0,3})',
                # "Hasta: BERKANT TOSUNOĞULLARI", "Hasta yakını: Ali Yücel"
                r'(?:(?<![a-zçğıöşü])Hasta(?:[ \t]+Yak[ıi]n[ıi])?[ \t]*[:=][ \t]*)([A-ZÇĞİÖŞÜa-zçğıöşü]{2,}(?:[ \t]+(?![A-ZÇĞİÖŞÜa-zçğıöşü]+[ \t]*[:=»>])(?!(?:[İIi]steyen|G[öo]nd|Servis|Doktor|Dokt|TC|Ya[şs]|Cinsiyet|Do[gğ]um|Baba|Anne|Protokol|Rapor|Biyopsi|Dosya|Kabul|Numune|Tarih|Klinik|Tetkik|Patoloji|Hasta)\b)[A-ZÇĞİÖŞÜa-zçğıöşü]{2,}){0,3})',
                # "Anne Adı: NATALIA"
                r'(?:Anne[ \t]*Ad[ıi]?[ \t]*[:.]?[ \t]*)([A-ZÇĞİÖŞÜa-zçğıöşü]{2,20})',
            ]
            for pattern in name_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    text = text[:m.start()] + "[HASTA_ADI_SILINDI]" + text[m.end():]
                    report.fields_removed.append(f"Hasta adı")
                    changes_made = True
                    break

            if changes_made:
                continue

            # Doktor etiketleri
            _name = r'([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü]+(?:[ \t]+(?![A-ZÇĞİÖŞÜa-zçğıöşü]+[ \t]*[:=»>])(?!(?:[İIi]steyen|G[öo]nd|Servis|Doktor|Dokt|TC|Ya[şs]|Cinsiyet|Do[gğ]um|Baba|Anne|Protokol|Rapor|Biyopsi|Dosya|Kabul|Numune|Tarih|Klinik|Tetkik|Patoloji|Hasta)\b)[A-ZÇĞİÖŞÜa-zçğıöşü]+){0,4})'
            _title = r'(?:(?:E[ĞGÇC][İI]?T[İI]M[ \t]+G[ÖO]REVL[İI]S[İI]|PROF\.?[ \t]*DR\.?|UZM\.?[ \t]*DR\.?|DO[ÇC]\.?[ \t]*DR\.?|OP\.?[ \t]*DR\.?|DR\.?)[ \t]+)?'
            doc_patterns = [
                r'(?:[İI]steyen[ \t]*Doktor[ \t]*[:.]?[ \t]*)' + _title + _name,
                r'(?:G[öo]nd(?:eren)?\.?[ \t]*Dokt?(?:or)?\.?[ \t]*[:;.]?[ \t]*)' + _title + _name,
                r'(?:Sorumlu[ \t]+Hekim[ \t]*[:.]?[ \t]*)' + _title + _name,
                r'(?:Raporlayan(?:[ \t]+(?:Doktor|Hekim|Patolog))?[ \t]*[:.]?[ \t]*)' + _title + _name,
                r'(?:Onaylayan(?:[ \t]+(?:Doktor|Hekim|Patolog))?[ \t]*[:.]?[ \t]*)' + _title + _name,
                r'(?:Patolog[ \t]*[:.][ \t]*)' + _title + _name,
                r'(?:(?:Klinisyen|Hekim|Doktor|Uzman|Cerrah)[ \t]*[:=][ \t]*)' + _title + _name,
            ]
            for pattern in doc_patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    text = text[:m.start()] + "[DOKTOR_SILINDI]" + text[m.end():]
                    report.fields_removed.append("Doktor adı")
                    changes_made = True
                    break

            if changes_made:
                continue

            # Baba Adı
            m = re.search(r'(?:Baba\s*Ad[ıi]?\s*[:.]?\s*)([A-ZÇĞİÖŞÜa-zçğıöşü]{2,20})', text, re.IGNORECASE)
            if m:
                text = text[:m.start()] + "[BABA_ADI_SILINDI]" + text[m.end():]
                report.fields_removed.append("Baba adı")
                changes_made = True
                continue

            # Doğum Yeri
            m = re.search(r'(?:Do[gğ]um\s*Yeri?\s*[:>.]?\s*)([A-ZÇĞİÖŞÜa-zçğıöşü]{2,25})', text, re.IGNORECASE)
            if m:
                text = text[:m.start()] + "[DOGUM_YERI_SILINDI]" + text[m.end():]
                report.fields_removed.append("Doğum yeri")
                changes_made = True
                continue

            # Doğum Tarihi
            m = re.search(r'(?:Do[gğ](?:um)?\s*\.?\s*Tar(?:ih)?[i]?\s*[:.]?\s*)([\d./\-\s]{6,20})', text, re.IGNORECASE)
            if m:
                text = text[:m.start()] + "[DOGUM_TARIHI_SILINDI]" + text[m.end():]
                report.fields_removed.append("Doğum tarihi")
                changes_made = True
                continue

            # Cinsiyet/Doğ.Tar/Yaş birleşik satır
            m = re.search(r'(?:Cinsiyet\s*[/,]?\s*Do[gğ]\.?\s*Tar[./]?\s*[/,]?\s*Ya[sş]\s*[:.]?\s*)([^\n]{5,40})', text, re.IGNORECASE)
            if m:
                text = text[:m.start()] + "[CINSIYET_YAS_SILINDI]" + text[m.end():]
                report.fields_removed.append("Cinsiyet/Doğum/Yaş")
                changes_made = True
                continue

        return text

    # ════════════════════════════════════════════
    # Katman 4: Unvanlı doktor adları
    # ════════════════════════════════════════════

    def _remove_doctor_names(self, text: str, report: AnonymizationReport) -> str:
        # İsim: en fazla 4 büyük harfle başlayan kelime; ':' ile biten etiket kelimesini yutmaz; satır aşmaz
        _NAME4 = r'[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü\.]+(?:[ \t]+(?![A-ZÇĞİÖŞÜa-zçğıöşü]+[ \t]*[:=»>])(?!(?:[İIi]steyen|G[öo]nd|Servis|Doktor|Dokt|TC|Ya[şs]|Cinsiyet|Do[gğ]um|Baba|Anne|Protokol|Rapor|Biyopsi|Dosya|Kabul|Numune|Tarih|Klinik|Tetkik|Patoloji|Hasta)\b)[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü\.]+){0,3}'
        title_patterns = [
            r'(?:PROF|Prof)\.?[ \t]*(?:DR|Dr)\.?[ \t]*' + _NAME4,
            r'(?:DO[CÇcç]|Do[cç])\.?[ \t]*(?:DR|Dr)\.?[ \t]*' + _NAME4,
            r'(?:UZM|Uzm)\.?[ \t]*(?:DR|Dr)\.?[ \t]*' + _NAME4,
            r'(?:OP|Op)\.?[ \t]*(?:DR|Dr)\.?[ \t]*' + _NAME4,
            r'(?:YRD|Yrd)\.?[ \t]*(?:DO[CÇcç]|Do[cç])\.?[ \t]*(?:DR|Dr)\.?[ \t]*' + _NAME4,
            r'(?:E[ĞGÇC][İI]?T[İI]M)[ \t]+G[ÖO]REVL[İI]S[İI][ \t]+' + _NAME4,
            r'(?:DR|Dr|dr)\.[ \t]*' + _NAME4,   # "Dr. Ayşe Kaya" / "DR. AYŞE KAYA"
        ]
        for pattern in title_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, "[DOKTOR_SILINDI]", text, flags=re.IGNORECASE)
                report.fields_removed.append("Doktor adı (unvanlı)")

        # ASİSTAN DR: + isim
        text = re.sub(
            r'AS[İI]STAN\s*DR[:.]\s*[A-ZÇĞİÖŞÜa-zçğıöşü\. \t]{2,40}',
            "[DOKTOR_SILINDI]", text, flags=re.IGNORECASE
        )

        # DR. + isim (mixed case)
        text = re.sub(
            r'\bDR\.?\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+){0,3}',
            "[DOKTOR_SILINDI]", text
        )

        # Patoloji Uzmanı satırı
        text = re.sub(r'Patoloji\s+Uzman[ıi]', "[UZMAN_SILINDI]", text)

        return text

    # ════════════════════════════════════════════
    # Katman 5: Tarihler
    # ════════════════════════════════════════════

    def _remove_dates(self, text: str, report: AnonymizationReport) -> str:
        count = 0

        # GG.AA.YYYY veya GG/AA/YYYY (saat dahil)
        p1 = r'\b\d{1,2}[./]\d{1,2}[./]\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b'
        count += len(re.findall(p1, text))
        text = re.sub(p1, "[TARIH_SILINDI]", text)

        # ISO: 2024-12-09
        p0 = r'\b(?:19|20)\d{2}-\d{2}-\d{2}\b'
        count += len(re.findall(p0, text))
        text = re.sub(p0, "[TARIH_SILINDI]", text)

        # GG-AA-YYYY (boşluklu/OCR)
        p2 = r'\b\d{1,2}\s*-\s*\d{1,2}\s*-\s*\d{4}\b'
        count += len(re.findall(p2, text))
        text = re.sub(p2, "[TARIH_SILINDI]", text)

        # GG,AA,YYYY (virgüllü OCR)
        p3 = r'\b\d{1,2},\d{1,2},\d{4}\b'
        count += len(re.findall(p3, text))
        text = re.sub(p3, "[TARIH_SILINDI]", text)

        # DD/MM/YY veya DD.MM.YY (kısa yıl)
        p4 = r'\b\d{1,2}[./]\d{1,2}[./]\d{2}\b'
        count += len(re.findall(p4, text))
        text = re.sub(p4, "[TARIH_SILINDI]", text)

        # Parantez içi tarih: (28/07/25)
        p5 = r'\(\d{1,2}/\d{1,2}/\d{2,4}\)'
        count += len(re.findall(p5, text))
        text = re.sub(p5, "[TARIH_SILINDI]", text)

        # MM/YY: 02/25, 04/24 — yalnızca tarih bağlamı olan satırlarda (lenf nodu "10/15" korunur)
        p6 = r'\b(?:0[1-9]|1[0-2])/(?:\d{2})\b'
        def _p6_line(m):
            nonlocal count
            line = m.group(0)
            if re.search(r'tarih|kabul|onay|istem|rapor|\bte\b|\bta\b|\bda\b|\bde\b', line, re.IGNORECASE) and not re.search(r'lenf|nod|/\d{2}\s*(?:pozitif|metast|adet)', line, re.IGNORECASE):
                n = len(re.findall(p6, line)); count += n
                return re.sub(p6, "[TARIH_SILINDI]", line)
            return line
        text = re.sub(r'^.*$', _p6_line, text, flags=re.MULTILINE)

        # DDMM/YY: 0112/25, 2001/26
        p7 = r'\b\d{4}/\d{2}\b'
        count += len(re.findall(p7, text))
        text = re.sub(p7, "[TARIH_SILINDI]", text)

        # Türkçe ay isimleri ile tarih: "12 şubat", "temmuzda"
        turkish_months = r'(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)'
        p8 = rf'\b\d{{1,2}}\s+{turkish_months}\b'
        count += len(re.findall(p8, text, re.IGNORECASE))
        text = re.sub(p8, "[TARIH_SILINDI]", text, flags=re.IGNORECASE)
        # "temmuzda", "martında" gibi ek almış aylar
        p9 = rf'\b{turkish_months}(?:da|de|ında|inde|ta|te)\b'
        count += len(re.findall(p9, text, re.IGNORECASE))
        text = re.sub(p9, "[TARIH_SILINDI]", text, flags=re.IGNORECASE)

        # Tarih etiketli satırlar
        text = re.sub(
            r'(?:Rapor|Lab\s*Geli[şs]|Onay|Numune\s*(?:Al[ıi]m|Kabul)|Sonu[çc]|Kabul|Al[ıi]nd[ıi][gğ][ıi]|Geldi[gğ]i|Revize|Rapor\s*Yazd[ıi]rma)[ \t]*Tarih[i]?[ \t]*[/:]?[ \t]*(?:Saat[i]?[ \t]*[:.]?[ \t]*)?(?:\[TARIH_SILINDI\][ \t]*)?[^\n]{0,30}',
            "[TARIH_ALANI_SILINDI]",
            text, flags=re.IGNORECASE
        )

        if count > 0:
            report.fields_removed.append(f"Tarih ({count} adet)")

        return text

    # ════════════════════════════════════════════
    # Katman 6: Yaş genelleştirme
    # ════════════════════════════════════════════

    def _generalize_age(self, text: str, report: AnonymizationReport) -> str:
        patterns = [
            r'(?:Ya[şs][ıi]?\s*[-/,]?\s*(?:Cinsiyet[i]?)?\s*[:.]?\s*)(\d{1,3})\s*(?:[-/]\s*([EKekMFmf]))?',
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                age = int(m.group(1))
                if 0 < age < 120:
                    gender = m.group(2)
                    decade = (age // 10) * 10
                    lower = max(0, decade - 5)
                    upper = lower + 9
                    replacement = f"[YAS_ARALIGI: {lower}-{upper}]"
                    if gender:
                        replacement += f" / {gender.upper()}"
                    text = text[:m.start()] + replacement + text[m.end():]
                    report.fields_generalized.append(f"Yaş → {lower}-{upper}")
        # Metin içi yaş: "45 yaşında", "57 yaşındaki", "63 y"
        def _age_repl(m):
            age = int(m.group(1))
            if not 0 < age < 120:
                return m.group(0)
            lower = max(0, (age // 10) * 10 - 5)
            report.fields_generalized.append(f"Yaş → {lower}-{lower + 9}")
            return f"[YAS_ARALIGI: {lower}-{lower + 9}] {m.group(2)}"
        text = re.sub(r'\b(\d{1,3})[ \t]*(ya[şs](?:[ıi]nda(?:ki)?|l[ıi])?)\b', _age_repl, text, flags=re.IGNORECASE)
        return text

    # ════════════════════════════════════════════
    # Katman 7: Kurum/hastane adları
    # ════════════════════════════════════════════

    def _remove_institutions(self, text: str, report: AnonymizationReport) -> str:
        inst_keywords = [
            r"[ÜU]niversites?i(?:n?[dn]e[n]?)?", r"Hastanes?i(?:n?[dn]e[n]?)?",
            r"Fak[üu]ltes?i", r"Laborat[uü]var[ıi](?:n?[dn]e[n]?)?|LABORATUVARI?",   # yalnızca özel ad hali ("... Laboratuvarı"), genel "laboratuvar" değil
            r"T[ıi]p\s+Fak", r"Devlet\s+Hastanesi",
            r"Sa[ğg]l[ıi]k\s+(?:Merkezi|Müdürlüğü|Bakanlığı|BAKANLIGI)",
            r"Şehir\s+Hastanesi",
        ]
        for keyword in inst_keywords:
            # Anahtar kelime harf-duyarsız (?i:), bağlam ise yalnızca Büyük harfle başlayan kelimeler
            kw = keyword if keyword.startswith("Laborat") else f"(?i:{keyword})"
            # Ön bağlam: en fazla 5 kelime; Büyük harfle başlayan VEYA "ve/ile" bağlacı ("Başakşehir Çam ve Sakura Şehir Hastanesi")
            pattern = rf'(?:(?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü\.]*|ve|ile)[ \t]+){{0,5}}\b(?:{kw})\b(?:[ \t]+(?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü\.]*|ve|Araştırma|Uygulama|Eğitim)){{0,3}}'
            if re.search(pattern, text):
                text = re.sub(pattern, "[KURUM_SILINDI]", text)
                report.fields_removed.append("Kurum/hastane adı")

        # T.C. ile başlayan satırlar
        text = re.sub(r'^T\s*\.?\s*C\.?(?:\s+Fi)?.*$', '[KURUM_SILINDI]', text, flags=re.MULTILINE)

        # İl adları (dolaylı tanımlayıcı): "AKSARAYDAN", "Konya'da"
        n = len(PROVINCE_RX.findall(text))
        if n:
            text = PROVINCE_RX.sub('[SEHIR_SILINDI]', text)
            report.fields_removed.append(f"Şehir ({n} adet)")

        # İl Sağlık Müdürlüğü
        text = re.sub(r'[A-Za-zçğıöşüÇĞİÖŞÜ]+\s+[İI]l\s+Sa[ğg]l[ıi]k\s+M[üu]d[üu]rl[üu][gğ][üu]', '[KURUM_SILINDI]', text)

        # Kurum-Adresi / Kurum Web
        text = re.sub(r'Kurum[--]?\s*[Aa]dres[i]?\s*[:.]?\s*[^\n]{5,80}', '[KURUM_ADRES_SILINDI]', text)
        text = re.sub(r'Kurum\s+Web\s+[Aa]dres\s*[:.]?\s*[^\n]{5,80}', '[KURUM_WEB_SILINDI]', text)

        return text

    # ════════════════════════════════════════════
    # Katman 8: İletişim bilgileri
    # ════════════════════════════════════════════

    def _remove_contact_info(self, text: str, report: AnonymizationReport) -> str:
        # Telefon/Faks
        tel_patterns = [
            r'(?:Tel|Tal|Tek|Faks?|Fax|GSM|Cep)[ \t]*[O(]?[ \t]*[:.]?[ \t]*[\d \t\(\)\-\+\']{7,25}',
            r'\(0\d{3}\)\s*[\d\s\-]{7,15}',
            r'\b0[ \t]?\(?5\d{2}\)?[ \t]?\d{3}[ \t]?\d{2}[ \t]?\d{2}\b',
            r'\+90[ \t]?\(?\d{3}\)?[ \t]?\d{3}[ \t]?\d{2}[ \t]?\d{2}\b',
            r'\b0[ \t]?\(?[2-4]\d{2}\)?[ \t]\d{3}[ \t]?\d{2}[ \t]?\d{2}\b',   # sabit hat "0332 310 50 00"
        ]
        for p in tel_patterns:
            if re.search(p, text, re.IGNORECASE):
                text = re.sub(p, "[ILETISIM_SILINDI]", text, flags=re.IGNORECASE)
                report.fields_removed.append("Telefon/Faks")

        # E-posta
        text = re.sub(r'(?:Posta|E-?mail|E-?posta)\s*[:.]?\s*\S+@\S+', "[EMAIL_SILINDI]", text, flags=re.IGNORECASE)
        text = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b', "[EMAIL_SILINDI]", text)

        # Web URL
        text = re.sub(r'https?://\S+', "[WEB_SILINDI]", text)
        text = re.sub(r'hteps[ıi]?//\S+', "[WEB_SILINDI]", text)
        text = re.sub(r'\b\w+\.(?:edu|gov|org|com)\.tr\b', "[WEB_SILINDI]", text)
        # Hab : URL (OCR bozuk)
        text = re.sub(r'Hab\s*[:.]?\s*\S+\.\S+', "[WEB_SILINDI]", text)

        # Adres
        text = re.sub(r'(?:Adres)\s*[:.]?\s*[^\n]{10,100}', "[ADRES_SILINDI]", text, flags=re.IGNORECASE)

        return text

    # ════════════════════════════════════════════
    # Katman 9: Diploma/tescil
    # ════════════════════════════════════════════

    def _remove_diploma_numbers(self, text: str, report: AnonymizationReport) -> str:
        pattern = r'(?:Dipl?[.:\-]?\s*Tes[cç]il\s*(?:No|Wo|Ng)?\s*[:.]?\s*)\d*'
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, "[DIPLOMA_SILINDI]", text, flags=re.IGNORECASE)
            report.fields_removed.append("Diploma tescil no")
        return text

    # ════════════════════════════════════════════
    # Katman 10: İmza blokları
    # ════════════════════════════════════════════

    def _remove_signature_blocks(self, text: str, report: AnonymizationReport) -> str:
        patterns = [
            r'Bu belge.*?imzalanm[ıi][şs]t[ıi]r\.?',
            r'Bu bilgiler.*?kullan[ıi]lamaz\.?',
            r'BU BELGEN[İI]N G[İI]ZL[İI]L[İI][ĞG][İI].*?KULLANILAMAZ\.?',
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
                report.fields_removed.append("İmza/güvenlik bloğu")
        return text

    # ════════════════════════════════════════════
    # Katman 11: İsim sözlüğü tabanlı tespit
    # ════════════════════════════════════════════

    def _remove_names_by_dictionary(self, text: str, report: AnonymizationReport) -> str:
        """Türk isim sözlüğü ile isim tespiti.

        Sadece sözlükte bulunan isimleri siler.
        Tıbbi bölümlerde (TANI, MAKROSKOPİ vb.) çalışmaz.
        """
        if not NAME_DICT["all_names"]:
            return text

        medical_sections = self._find_medical_sections(text)

        # Büyük harfli kelime gruplarını bul (2-4 kelime)
        tr_chars = r'A-ZÇĞİÖŞÜa-zçğıöşü'
        # Örtüşen çiftler: "X AHMET YILMAZ" → (X,AHMET) ve (AHMET,YILMAZ) ikisi de denenir
        pattern = rf'(?=\b([{tr_chars}]{{2,20}})[ \t]+([{tr_chars}]{{2,20}})(?:[ \t]+([{tr_chars}]{{2,20}}))?\b)'

        matches = list(re.finditer(pattern, text))

        # Sondan başa (indeks kayması olmasın)
        for m in reversed(matches):
            # Tıbbi bölüm içindeyse atla
            if self._is_in_medical_section(m.start(), medical_sections):
                continue

            # Zaten maskelenmiş mi?
            if '_SILINDI]' in text[max(0, m.start()-5):m.start()]:
                continue

            w1 = m.group(1).strip()
            w2 = m.group(2).strip()
            w3 = m.group(3).strip() if m.group(3) else None

            # "Umut verici": ilk kelime büyük, ikinci küçük harfle başlıyorsa özel isim değildir
            if w1[0].isupper() and w2[0].islower():
                continue

            w1_upper = w1.upper()
            w2_upper = w2.upper()
            w1_norm = _normalize_turkish(w1_upper)
            w2_norm = _normalize_turkish(w2_upper)

            # İsim sözlüğünde var mı? (hem orijinal hem normalize edilmiş hali)
            w1_is_name = w1_upper in NAME_DICT["all_names"] or w1_norm in NAME_DICT["all_names"]
            w2_is_name = w2_upper in NAME_DICT["all_names"] or w2_norm in NAME_DICT["all_names"]

            # Soyadı suffix kontrolü
            w2_has_suffix = any(w2_upper.endswith(s) for s in NAME_DICT["surname_suffixes"])

            is_name = False
            ambiguous = w1_upper in AMBIGUOUS_NAME_WORDS or w2_upper in AMBIGUOUS_NAME_WORDS

            # Durum 1: Her iki kelime de isim sözlüğünde
            if w1_is_name and w2_is_name:
                is_name = True

            # Durum 2: İlk kelime ad sözlüğünde, ikinci soyadı suffix'i ile bitiyor
            # ("UMUT VERİCİ", "SELİM LEZYONLAR" gibi belirsiz kelimelerde uygulanmaz)
            elif w1_upper in NAME_DICT["first_names"] and w2_has_suffix and not ambiguous:
                is_name = True

            # Durum 3: İlk kelime ad sözlüğünde, ikinci soyadı sözlüğünde
            elif w1_upper in NAME_DICT["first_names"] and w2_upper in NAME_DICT["surnames"]:
                is_name = True

            # Durum 4: İlk kelime soyadı sözlüğünde, ikinci ad sözlüğünde (ters sıra)
            elif w1_upper in NAME_DICT["surnames"] and w2_upper in NAME_DICT["first_names"]:
                is_name = True

            if is_name:
                # Zaten (önceki, sağdaki) bir eşleşmeyle silinmiş bölgeye denk geldiyse atla
                if text[m.start():m.start() + len(w1)] != w1:
                    continue
                # Eşleşme lookahead olduğu için uzunluğu metinden yeniden ölç
                seg = text[m.start():m.start() + len(w1) + len(w2) + (len(w3) if w3 else 0) + 8]
                pair = re.match(rf'{re.escape(w1)}[ \t]+{re.escape(w2)}', seg)
                if not pair:
                    continue
                end = m.start() + pair.end()
                # 3. kelime varsa ve o da isimse, onu da dahil et
                if w3 and w3.upper() in NAME_DICT["all_names"]:
                    trip = re.match(rf'{re.escape(w1)}[ \t]+{re.escape(w2)}[ \t]+{re.escape(w3)}', seg)
                    if trip:
                        end = m.start() + trip.end()

                text = text[:m.start()] + "[ISIM_SILINDI]" + text[end:]
                report.fields_removed.append("İsim (sözlük)")

        # Apostrof/tırnak ile yapışık isimler: 'SEDA YILMAZ
        apost_pattern = rf"""['"'`]\s*([{tr_chars}]{{2,15}})\s+([{tr_chars}]{{2,20}})"""
        for m in reversed(list(re.finditer(apost_pattern, text))):
            if self._is_in_medical_section(m.start(), medical_sections):
                continue
            w1 = m.group(1).upper()
            w2 = m.group(2).upper()
            if w1 in NAME_DICT["all_names"] or w2 in NAME_DICT["all_names"]:
                if w1 in NAME_DICT["first_names"] or w2 in NAME_DICT["surnames"]:
                    text = text[:m.start()] + "[ISIM_SILINDI]" + text[m.end():]
                    report.fields_removed.append("İsim (yapışık)")

        # "sile AEA:" gibi OCR bozuk yapılardaki isimler
        sile_pattern = rf'(?:sile\s+AEA\s*[:.]?\s*)([{tr_chars}\s]{{2,40}})'
        text = re.sub(sile_pattern, "[ISIM_SILINDI]", text, flags=re.IGNORECASE)

        # DR + sözlükteki soyadı (unvansız doktor referansı: "DR ÖZTÜRK")
        dr_pattern = rf'\bDR\s+([{tr_chars}]{{2,20}})'
        for m in reversed(list(re.finditer(dr_pattern, text))):
            if '_SILINDI]' in text[max(0, m.start()-5):m.start()]:
                continue
            surname = m.group(1).upper()
            surname_norm = _normalize_turkish(surname)
            if surname in NAME_DICT["surnames"] or surname_norm in NAME_DICT["surnames"] or surname in NAME_DICT["all_names"] or surname_norm in NAME_DICT["all_names"]:
                text = text[:m.start()] + "[DOKTOR_SILINDI]" + text[m.end():]
                report.fields_removed.append("Doktor (DR+sözlük)")

        # Etiket sonrası kalıntı: "[DOKTOR_SILINDI] ERDEM" — aynı satırda kalan büyük harfli kelimeler
        residual = r'(\[(?:DOKTOR|HASTA_ADI|ISIM|BABA_ADI)_SILINDI\])((?:[ \t]+[A-ZÇĞİÖŞÜ]{2,}(?:[ \t]*\.)?)+)(?=[ \t]*(?:\n|$|[:|]))'
        for m in reversed(list(re.finditer(residual, text))):
            text = text[:m.start()] + m.group(1) + text[m.end():]
            report.fields_removed.append("İsim (etiket sonrası kalıntı)")

        # Son geçiş: [WEB_SILINDI] veya [ILETISIM_SILINDI] sonrasında kalan sözlük isimleri
        post_tag = rf'\[(?:WEB|ILETISIM|EMAIL)_SILINDI\]\s*([{tr_chars}]{{2,15}})\s+([{tr_chars}]{{2,20}})'
        for m in reversed(list(re.finditer(post_tag, text))):
            w1 = m.group(1).upper()
            w2 = m.group(2).upper()
            if w1 in NAME_DICT["all_names"] or w2 in NAME_DICT["all_names"]:
                # Sadece etiket sonrasındaki ismi sil, etiketi koru
                tag_end = m.start() + text[m.start():].index(']') + 1
                text = text[:tag_end] + " [ISIM_SILINDI]" + text[m.end():]
                report.fields_removed.append("İsim (etiket sonrası)")

        return text

    # ════════════════════════════════════════════
    # Katman 12: Sayısal tanımlayıcılar
    # ════════════════════════════════════════════

    def _remove_numeric_identifiers(self, text: str, report: AnonymizationReport) -> str:
        # İşlem No: 10+ haneli
        pattern = r'(?:[İI][şs]lem\s*(?:No|©)?[\s:.]*)\d{5,15}'
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, "[ISLEM_NO_SILINDI]", text, flags=re.IGNORECASE)
            report.fields_removed.append("İşlem No")

        # Protokol numaraları: 51441/23 formatı
        proto_pattern = r'\b\d{4,6}/\d{2,4}\b'
        proto_matches = re.findall(proto_pattern, text)
        if proto_matches:
            text = re.sub(proto_pattern, "[PROTOKOL_SILINDI]", text)
            report.fields_removed.append(f"Protokol no ({len(proto_matches)} adet)")

        return text

    # ════════════════════════════════════════════
    # Temizlik
    # ════════════════════════════════════════════

    def _cleanup(self, text: str) -> str:
        # Ardışık etiketleri birleştir
        text = re.sub(r'(\[[\w_]+\]\s*){4,}', '[COKLU_ALAN_SILINDI]\n', text)
        # Fazla boş satırlar
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        # Satır sonu boşlukları
        lines = [line.rstrip() for line in text.split('\n')]
        return '\n'.join(lines).strip()


# ════════════════════════════════════════════
# Görüntü anonimleştirme (mevcut)
# ════════════════════════════════════════════

class ImageAnonymizer:
    """PDF ve görüntü dosyalarından PII alanlarını maskeleme."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def anonymize_pdf(self, pdf_bytes: bytes) -> bytes:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        new_doc = fitz.open()
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            text_content = page.get_text("text")
            medical_start_y = self._find_medical_content_start(blocks, text_content)
            if medical_start_y is not None:
                rect = fitz.Rect(0, 0, page.rect.width, medical_start_y - 10)
                page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                text_point = fitz.Point(50, medical_start_y - 30)
                page.insert_text(text_point, "[KISISEL BILGILER MASKELENDI]", fontsize=10, color=(0.5, 0.5, 0.5))
            sig_y = self._find_signature_block(blocks, text_content)
            if sig_y is not None:
                rect = fitz.Rect(0, sig_y, page.rect.width, page.rect.height)
                page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
            new_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)
        result = new_doc.tobytes()
        doc.close()
        new_doc.close()
        return result

    def _find_medical_content_start(self, blocks: list, text: str) -> Optional[float]:
        keywords = ["MAKROSKOBİK", "MAKROSKOPİ", "MAKROSKOPI", "MİKROSKOPİ", "MIKROSKOPI", "TANI", "KLİNİK"]
        for block in blocks:
            if block["type"] == 0:
                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if any(kw in line_text.upper() for kw in keywords):
                        return line["bbox"][1]
        return None

    def _find_signature_block(self, blocks: list, text: str) -> Optional[float]:
        keywords = ["imzalanmıştır", "Dipl Tescil", "GİZLİLİĞİ", "KOPYALANAMAZ"]
        for block in blocks:
            if block["type"] == 0:
                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if any(kw.lower() in line_text.lower() for kw in keywords):
                        return line["bbox"][1] - 5
        return None
