"""OCR backend soyutlaması — Tesseract (temel) ve Vision-LLM (GLM-OCR, Qwen3-VL, PaddleOCR-VL).

Tüm işlem RAM'de yapılır: ham dosya diske yazılmaz, sayfa görüntüleri byte olarak
LLM backend'e iletilir ve atılır.
"""

import io
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from .llm import LLMBackend, make_llm

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
TEXT_SUFFIXES = {".txt", ".text"}

VISION_OCR_PROMPT = (
    "Bu görüntü Türkçe bir tıbbi form (anamnez veya patoloji raporu) taramasıdır. "
    "Görüntüdeki TÜM metni olduğu gibi, satır düzenini koruyarak, hiçbir şeyi atlamadan "
    "ve hiçbir şey eklemeden düz metin olarak yaz. Yorum yapma, özetleme, çeviri yapma. "
    "Okunamayan yerler için [okunamadı] yaz. Sadece metni döndür."
)


@dataclass
class OCRResult:
    text: str
    engine: str
    pages: int = 0
    page_engines: list[str] = field(default_factory=list)
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    alt_text: str = ""            # ikincil motor (tesseract) çıktısı — çapraz sızıntı kontrolü için
    alt_engine: str = ""
    completeness: float = 1.0     # len(primary)/len(alt); düşükse VLM içerik atlamış olabilir


MAX_PIXELS = 25_000_000  # sayfa başına piksel bütçesi (~A4 @ 300 dpi = 8.7 MP); devasa sayfada dpi düşürülür


def _render_pdf_pages(file_bytes: bytes, dpi: int, max_pages: int):
    """Sayfaları tek tek üretir (generator) — tüm PDF'i PNG olarak bellekte tutmaz."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            w_in, h_in = page.rect.width / 72, page.rect.height / 72
            eff_dpi = dpi
            if w_in * h_in * dpi * dpi > MAX_PIXELS:
                eff_dpi = max(72, int((MAX_PIXELS / (w_in * h_in)) ** 0.5))
                logger.warning("Sayfa %d çok büyük, dpi %d→%d", i + 1, dpi, eff_dpi)
            pix = page.get_pixmap(dpi=eff_dpi)
            yield pix.tobytes("png")
            del pix
    finally:
        doc.close()


def _pdf_text_layer(file_bytes: bytes, max_pages: int) -> list[str]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = [page.get_text("text").strip() for i, page in enumerate(doc) if i < max_pages]
    doc.close()
    return texts


def _shrink(png: bytes, max_dim: int) -> bytes:
    img = Image.open(io.BytesIO(png))
    if max(img.size) <= max_dim:
        return png
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


class TesseractOCR:
    """tesseract'ı stdin→stdout ile çağırır: pytesseract'ın aksine görüntüyü ve metni
    geçici dosyaya YAZMAZ (KVKK: ham veri diske inmez)."""

    name = "tesseract"

    def __init__(self, cfg: dict):
        self.lang = cfg.get("lang", "tur+eng")
        self.timeout = float(cfg.get("tesseract_timeout", 120))
        self.binary = cfg.get("tesseract_cmd") or shutil.which("tesseract") or "tesseract"
        self._langs: Optional[list[str]] = None

    def ocr_image(self, png: bytes) -> str:
        proc = subprocess.run(
            [self.binary, "stdin", "stdout", "-l", self.lang],
            input=png, capture_output=True, timeout=self.timeout, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"tesseract çıkış kodu {proc.returncode}")  # stderr loglanmaz
        return proc.stdout.decode("utf-8", errors="replace")

    def health(self) -> dict:
        try:
            if self._langs is None:  # başlangıçta bir kez
                out = subprocess.run([self.binary, "--list-langs"], capture_output=True, timeout=10, check=False)
                self._langs = [l.strip() for l in out.stdout.decode(errors="replace").splitlines()[1:] if l.strip()]
            return {"ok": "tur" in self._langs, "engine": self.name, "languages": self._langs}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "engine": self.name, "error": type(e).__name__}


class VisionLLMOCR:
    name = "vision_llm"

    def __init__(self, cfg: dict):
        self.llm: LLMBackend = make_llm(cfg)
        self.model = cfg["model"]
        self.max_dimension = int(cfg.get("max_dimension", 1600))

    def ocr_image(self, png: bytes) -> str:
        png = _shrink(png, self.max_dimension)
        out = self.llm.chat(
            [{"role": "user", "content": VISION_OCR_PROMPT}],
            images=[png],
            max_tokens=4096,
        )
        return out.strip()

    def health(self) -> dict:
        h = self.llm.health()
        h["engine"] = self.name
        return h


class OCRService:
    """Dosya → metin. Vision-LLM birincil, Tesseract fallback."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dpi = int(cfg.get("dpi", 200))
        self.max_pages = int(cfg.get("max_pages", 10))
        self.min_chars = int(cfg.get("min_chars", 20))
        self.backend_name = cfg.get("backend", "tesseract")
        self.fallback = bool(cfg.get("fallback_to_tesseract", True))
        self.cross_check = bool(cfg.get("cross_check", True))
        self.min_completeness = float(cfg.get("min_completeness", 0.6))
        self.tesseract = TesseractOCR(cfg)
        self.vision: Optional[VisionLLMOCR] = None
        if self.backend_name == "vision_llm":
            self.vision = VisionLLMOCR(cfg.get("vision", {}))

    def _ocr_page(self, png: bytes, result: OCRResult) -> str:
        if self.vision is not None:
            try:
                text = self.vision.ocr_image(png)
                if len(text.strip()) >= self.min_chars:
                    result.page_engines.append(self.vision.model)
                    return text
                result.warnings.append("vision OCR yetersiz metin döndürdü")
            except Exception as e:  # noqa: BLE001
                logger.warning("Vision OCR hatası: %s", e)
                result.warnings.append(f"vision OCR hatası: {type(e).__name__}")
                if not self.fallback:
                    raise
        text = self.tesseract.ocr_image(png)
        result.page_engines.append("tesseract")
        return text

    def extract(self, file_bytes: bytes, suffix: str) -> OCRResult:
        t0 = time.time()
        suffix = suffix.lower()
        result = OCRResult(text="", engine=self.backend_name)

        if suffix in TEXT_SUFFIXES:
            result.text = file_bytes.decode("utf-8", errors="replace")
            result.engine = "text"
            result.pages = 1
        elif suffix == ".pdf":
            layer = _pdf_text_layer(file_bytes, self.max_pages)
            texts, alts = [], []
            n_pages = 0
            for i, png in enumerate(_render_pdf_pages(file_bytes, self.dpi, self.max_pages)):
                n_pages += 1
                # Gerçek metin katmanı varsa OCR'a gerek yok
                if i < len(layer) and len(layer[i]) > 50:
                    texts.append(layer[i])
                    result.page_engines.append("pdf_text_layer")
                else:
                    texts.append(self._ocr_page(png, result))
                    alts.append(self._alt_page(png))
            result.text = "\n\n".join(t.strip() for t in texts if t.strip())
            result.alt_text = "\n\n".join(t.strip() for t in alts if t.strip())
            result.pages = n_pages
        elif suffix in IMAGE_SUFFIXES:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png = buf.getvalue()
            result.text = self._ocr_page(png, result).strip()
            result.alt_text = self._alt_page(png).strip()
            result.pages = 1
        else:
            raise ValueError(f"Desteklenmeyen dosya türü: {suffix}")

        if result.alt_text:
            result.alt_engine = "tesseract"
            result.completeness = round(len(result.text) / max(1, len(result.alt_text)), 2)
            if result.completeness < self.min_completeness:
                result.warnings.append(
                    f"olası içerik atlama: birincil/tesseract oranı {result.completeness}"
                )
        result.seconds = round(time.time() - t0, 2)
        engines = set(result.page_engines)
        if engines:
            result.engine = "+".join(sorted(engines))
        return result

    def _alt_page(self, png: bytes) -> str:
        """Çapraz kontrol için ikincil OCR (tesseract). Birincil zaten tesseract ise gereksiz."""
        if not self.cross_check or self.vision is None:
            return ""
        try:
            return self.tesseract.ocr_image(png)
        except Exception as e:  # noqa: BLE001
            logger.warning("Alt OCR hatası: %s", type(e).__name__)
            return ""

    def health(self) -> dict:
        out = {"tesseract": self.tesseract.health()}
        if self.vision is not None:
            out["vision"] = self.vision.health()
        return out
