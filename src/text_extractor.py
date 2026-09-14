"""Metin Çıkarma Modülü — PDF ve görüntülerden metin çıkarır (lokal).

Tüm işlem lokal yapılır, hiçbir veri dışarı gönderilmez.
PyMuPDF ile PDF metin katmanını okur, Tesseract OCR ile görüntülerden metin çıkarır.
"""

import io
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def extract_from_pdf(file_bytes: bytes, config: Optional[dict] = None) -> str:
    """PDF'ten metin çıkar.

    Önce metin katmanını dener, yetersizse sayfaları görüntüye çevirip OCR yapar.

    Args:
        file_bytes: PDF dosyasının byte içeriği
        config: PDF ayarları (dpi, max_pages, extract_text_first)

    Returns:
        Çıkarılan metin
    """
    config = config or {}
    dpi = config.get("dpi", 300)
    max_pages = config.get("max_pages", 10)
    extract_text_first = config.get("extract_text_first", True)

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    all_text = []

    for i, page in enumerate(doc):
        if i >= max_pages:
            logger.warning(f"Maksimum sayfa sayısına ulaşıldı ({max_pages}), kalan sayfalar atlanıyor")
            break

        if extract_text_first:
            text = page.get_text("text").strip()
            if len(text) > 50:
                all_text.append(text)
                continue

        # Metin katmanı yetersiz — sayfayı görüntüye çevirip OCR yap
        try:
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            ocr_text = _ocr_image_bytes(img_bytes, config)
            if ocr_text.strip():
                all_text.append(ocr_text.strip())
        except Exception as e:
            logger.warning(f"Sayfa {i+1} OCR hatası: {e}")

    page_count = len(doc)
    doc.close()

    result = "\n\n".join(all_text)
    logger.info(f"PDF'ten {len(result)} karakter çıkarıldı ({min(page_count, max_pages)} sayfa)")
    return result


def extract_from_image(file_bytes: bytes, config: Optional[dict] = None) -> str:
    """Görüntüden metin çıkar (Tesseract OCR).

    Args:
        file_bytes: Görüntü dosyasının byte içeriği
        config: OCR ayarları (lang)

    Returns:
        Çıkarılan metin
    """
    return _ocr_image_bytes(file_bytes, config)


def extract_from_file(file_path: str, config: Optional[dict] = None) -> str:
    """Dosya yolundan metin çıkar (otomatik format tespiti).

    Args:
        file_path: Dosya yolu
        config: İşleme ayarları

    Returns:
        Çıkarılan metin
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    file_bytes = path.read_bytes()

    if suffix == ".pdf":
        return extract_from_pdf(file_bytes, config)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
        return extract_from_image(file_bytes, config)
    elif suffix in {".txt", ".text"}:
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {suffix}")


def _ocr_image_bytes(image_bytes: bytes, config: Optional[dict] = None) -> str:
    """Görüntü byte'larından OCR ile metin çıkar."""
    config = config or {}
    lang = config.get("ocr_lang", "tur+eng")

    try:
        import pytesseract
    except ImportError:
        logger.error("pytesseract yüklü değil. pip install pytesseract")
        return ""

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, lang=lang)
        return text
    except Exception as e:
        logger.error(f"OCR hatası: {e}")
        return ""


def check_tesseract() -> dict:
    """Tesseract OCR kurulumunu kontrol et."""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        langs = pytesseract.get_languages()
        return {
            "installed": True,
            "version": str(version),
            "languages": langs,
            "has_turkish": "tur" in langs,
        }
    except Exception as e:
        return {
            "installed": False,
            "error": str(e),
        }
