"""Anonymizer regresyon testleri — bilinen sızıntılar ve sentetik senaryolar.

Kural: Her test 'beklenen sızıntı: sıfır' ilkesiyle yazılır. Gerçek dosya
çıktıları için output/originals/ altındaki OCR metinleri kullanılır.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.anonymizer import ReportAnonymizer

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def anon():
    return ReportAnonymizer({"generalize_age": True, "remove_dates": True, "remove_institutions": True})


def _leaks(text: str, tokens: list[str]) -> list[str]:
    up = text.upper()
    return [t for t in tokens if t.upper() in up]


# ── Gerçek dosyalarda bilinen PII (yalnızca lokal; repoya girmez) ─────────
# tests/private/known_pii.txt: her satırda bir PII parçası (isim, TC, no...). output/originals/*.txt:
# gerçek OCR çıktıları. İkisi de .gitignore'da; yoksa test atlanır.
PRIVATE_PII = ROOT / "tests" / "private" / "known_pii.txt"
KNOWN_PII = [l.strip() for l in PRIVATE_PII.read_text(encoding="utf-8").splitlines() if l.strip()] if PRIVATE_PII.exists() else []
REAL_FILES = sorted((ROOT / "output" / "originals").glob("*_original.txt"))


@pytest.mark.skipif(not (KNOWN_PII and REAL_FILES), reason="gerçek veri regresyonu: tests/private/known_pii.txt ve output/originals yok")
@pytest.mark.parametrize("orig", REAL_FILES, ids=lambda p: p.stem)
def test_real_files_no_known_pii(anon, orig):
    out, _ = anon.anonymize(orig.read_text(encoding="utf-8"))
    assert _leaks(out, KNOWN_PII) == []
    assert not re.search(r"\b[1-9]\d{10}\b", out), "11 haneli TC kaldı"
    assert not re.search(r"\b\d{1,2}[./]\d{1,2}[./]\d{4}\b", out), "tarih kaldı"


# ── Sentetik senaryolar ──────────────────────────────────────────────────
def test_diploma_number_removed(anon):
    out, _ = anon.anonymize("Uzm. Dr. HASAN DEMİR Dipl-Tescil No: 123456\nTANI: karsinom")
    assert "123456" not in out and "HASAN" not in out


def test_three_word_doctor_after_label(anon):
    out, _ = anon.anonymize("Gönd. Dokt. ; EĞTİM GÖREVLİSİ CANER UĞUR AKSU\nTC Kimlik No :12345678901")
    assert _leaks(out, ["CANER", "UĞUR", "AKSU", "12345678"]) == []


def test_islem_no_with_double_punct(anon):
    out, _ = anon.anonymize("islem No. : 5010226416 | Rapor No : 51440 / 2023")
    assert "5010226416" not in out


def test_hasta_label_variant(anon):
    out, _ = anon.anonymize("Hasta: BERKANT TOSUNOĞULLARI\nSorumlu Hekim: Dr. ZEYNEP ARSLANTÜRK\nMide biyopsisi: adenokarsinom")
    assert _leaks(out, ["BERKANT", "TOSUNOĞULLARI", "ZEYNEP", "ARSLANTÜRK"]) == []
    assert "adenokarsinom" in out


def test_signature_names_and_mobile(anon):
    out, _ = anon.anonymize("Raporlayan: Elif Şahinoğlu\nOnaylayan Patolog  : SERKAN YÜCEBAŞ\nHasta yakını: Ali Yücel 05551234567")
    assert _leaks(out, ["Elif", "Şahinoğlu", "SERKAN", "YÜCEBAŞ", "Ali Yücel", "05551234567"]) == []


def test_anne_adi(anon):
    out, _ = anon.anonymize("Anne Adı: NATALIA\nTiroid: papiller karsinom")
    assert "NATALIA" not in out and "papiller karsinom" in out


def test_medical_phrases_not_deleted(anon):
    src = "Kemik iliği: Blast %30, AML M4. NPM1 pozitif. Umut verici yanıt. Deniz seviyesi hemoglobin 9.2."
    out, _ = anon.anonymize(src)
    assert "Umut verici" in out and "Deniz seviyesi" in out and "NPM1 pozitif" in out


def test_headers_not_swallowed_across_lines(anon):
    src = "Konya Şehir Hastanesi\nTIBBI PATOLOJİ TETKİK SONUÇ RAPORU\nKabul Tarihi\nNUMUNE BİLGİLERİ\nAlındığı Yer : Akciğer"
    out, _ = anon.anonymize(src)
    assert "TIBBI PATOLOJİ TETKİK SONUÇ RAPORU" in out
    assert "NUMUNE BİLGİLERİ" in out
    assert "Konya" not in out


def test_province_names_removed_with_suffix(anon):
    out, _ = anon.anonymize("AKSARAYDAN POLİSTEMİ SEBEBİYLE YÖNLENDİRİLEN HASTA. TAKİPLERİNE Aksaray'da DEVAM EDECEK. Konya'dan geldi.")
    assert "AKSARAY" not in out.upper() and "KONYA" not in out.upper()
    assert "POLİSTEMİ" in out and "DEVAM EDECEK" in out


def test_province_no_false_positive_inside_words(anon):
    out, _ = anon.anonymize("Advanced karsinom, ordusal değil; bolus verildi. Vanadyum yok.")
    assert out == "Advanced karsinom, ordusal değil; bolus verildi. Vanadyum yok."


def test_institution_with_conjunction_and_lowercase_province(anon):
    """Batch 5'te kapı yakaladı: 'Başaksehir Çam ve' kalıntısı ve küçük harfli 'istanbul'."""
    out, _ = anon.anonymize("istanbul İl Sağlık Müdürlüğü\nBaşaksehir Çam ve Sakura Şehir Hastanesi\nPATOLOJİ TALEP FORMU")
    assert "Başaksehir" not in out and "Çam" not in out and "istanbul" not in out.lower()
    assert "PATOLOJİ TALEP FORMU" in out


def test_short_ambiguous_provinces_not_matched_lowercase(anon):
    src = "batman filmi, van gölü, ordu birliği, bolu dağı"
    assert anon.anonymize(src)[0] == src


def test_label_with_double_punctuation_and_unknown_surname(anon):
    """Sözlükte OLMAYAN soyadı, yalnızca etiketle yakalanmalı (OCR '=:' / '= :')."""
    out, _ = anon.anonymize("Hasta Adi Soyadi =: KEMAL KARAGÖZOĞLU Isteyen Servis » AMELİYATHANE\nHasta Adi Soyadi = : VELİ TOSUNOĞULLARI isteyen Servis : BEYİN")
    assert "KARAGÖZOĞLU" not in out and "TOSUNOĞULLARI" not in out and "KEMAL" not in out
    assert "AMELİYATHANE" in out and "BEYİN" in out
