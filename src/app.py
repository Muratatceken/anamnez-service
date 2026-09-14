"""Web Arayüzü — KVKK Uyumlu Anonimizasyon Sistemi.

Anamnez ve patoloji formlarından kişisel verileri temizleyerek
API LLM'lere gönderilebilir hale getirir.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_extractor import extract_from_file, extract_from_pdf, extract_from_image, check_tesseract
from src.anonymizer import ReportAnonymizer
from src.db import Database
from src.models import AnonymizationRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

config = None
anonymizer = None
db = None


def init_components():
    global config, anonymizer, db

    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    anon_config = config.get("anonymization", {})
    anonymizer = ReportAnonymizer(anon_config)
    db = Database(config["database"]["path"])


def process_anonymization(file, text_input):
    """Ana anonimizasyon fonksiyonu."""
    start_time = time.time()

    if file is None and not (text_input or "").strip():
        return "Lutfen bir dosya yukleyin veya metin girin.", "", "", "", ""

    try:
        if file is not None:
            path = Path(file.name)
            filename = path.name
            suffix = path.suffix.lower()

            # Metin cikar
            raw_text = extract_from_file(str(path), config.get("pdf", {}))
            input_mode = f"{suffix.upper().lstrip('.')} -> Metin Cikarma"
        else:
            filename = "manuel_giris.txt"
            raw_text = text_input
            input_mode = "Manuel Metin Girisi"

        if len(raw_text.strip()) < 20:
            return "Yeterli metin cikarilmadi. Dosya icerigini kontrol edin.", "", "", "", ""

        # Anonimize et
        anonymized_text, report = anonymizer.anonymize(raw_text)
        elapsed = time.time() - start_time

        # Logla
        record = AnonymizationRecord(
            filename=filename,
            file_type=Path(filename).suffix.lstrip(".") or "text",
            original_length=report.original_length,
            anonymized_length=report.anonymized_length,
            fields_removed_count=len(report.fields_removed),
            fields_removed=json.dumps(report.fields_removed, ensure_ascii=False),
            fields_generalized=json.dumps(report.fields_generalized, ensure_ascii=False),
            warnings=json.dumps(report.warnings, ensure_ascii=False),
            processing_time_seconds=round(elapsed, 2),
            method="regex",
        )
        db.log_anonymization(record)

        # Durum
        status = f"Tamamlandi ({elapsed:.1f}s) - {input_mode}"

        # Rapor
        report_lines = []
        report_lines.append(f"**Orijinal:** {report.original_length} karakter")
        report_lines.append(f"**Anonimize:** {report.anonymized_length} karakter")
        report_lines.append(f"**Silinen alan sayisi:** {len(report.fields_removed)}")

        if report.fields_removed:
            report_lines.append("\n**Silinen alanlar:**")
            for f in report.fields_removed:
                report_lines.append(f"- {f}")

        if report.fields_generalized:
            report_lines.append("\n**Genellestirilen:**")
            for g in report.fields_generalized:
                report_lines.append(f"- {g}")

        if report.warnings:
            report_lines.append("\n**Uyarilar:**")
            for w in report.warnings:
                report_lines.append(f"- {w}")

        report_text = "\n".join(report_lines)

        return status, raw_text, anonymized_text, report_text, anonymized_text

    except Exception as e:
        logger.error(f"Isleme hatasi: {e}", exc_info=True)
        return f"Hata: {e}", "", "", "", ""


def process_batch(files):
    """Toplu anonimizasyon."""
    if not files:
        return "Dosya yuklenmedi.", ""

    results = []
    total_removed = 0
    start_time = time.time()

    for file in files:
        path = Path(file.name)
        try:
            raw_text = extract_from_file(str(path), config.get("pdf", {}))
            if len(raw_text.strip()) < 20:
                results.append(f"- **{path.name}**: Yeterli metin cikarilmadi")
                continue

            anonymized_text, report = anonymizer.anonymize(raw_text)
            removed_count = len(report.fields_removed)
            total_removed += removed_count

            # Logla
            record = AnonymizationRecord(
                filename=path.name,
                file_type=path.suffix.lstrip("."),
                original_length=report.original_length,
                anonymized_length=report.anonymized_length,
                fields_removed_count=removed_count,
                fields_removed=json.dumps(report.fields_removed, ensure_ascii=False),
                fields_generalized=json.dumps(report.fields_generalized, ensure_ascii=False),
                warnings=json.dumps(report.warnings, ensure_ascii=False),
                processing_time_seconds=0,
                method="regex",
            )
            db.log_anonymization(record)

            results.append(
                f"- **{path.name}**: {report.original_length} -> {report.anonymized_length} karakter, "
                f"{removed_count} alan silindi"
            )
        except Exception as e:
            results.append(f"- **{path.name}**: HATA - {e}")

    elapsed = time.time() - start_time

    summary = f"## Toplu Anonimizasyon Sonucu\n\n"
    summary += f"**{len(files)} dosya** islendi ({elapsed:.1f}s)\n"
    summary += f"**Toplam {total_removed} kisisel veri** alani silindi\n\n"
    summary += "### Detaylar\n"
    summary += "\n".join(results)

    return summary, ""


def get_stats():
    """Anonimizasyon istatistikleri."""
    try:
        stats = db.get_anonymization_stats()

        text = f"""## Anonimizasyon Istatistikleri
- **Toplam islem:** {stats['total_anonymizations']}
- **Ort. silinen alan:** {stats['avg_fields_removed']}
- **Ort. islem suresi:** {stats['avg_processing_time']}s

## Son Islemler
"""
        for r in stats["recent"][:15]:
            ts = r["timestamp"][:16]
            text += f"- [{ts}] **{r['filename']}** — {r['fields_removed_count']} alan silindi ({r['method']})\n"

        return text
    except Exception as e:
        return f"Hata: {e}"


def check_system():
    """Sistem durumunu kontrol et."""
    lines = []

    # Tesseract
    tess = check_tesseract()
    if tess.get("installed"):
        lines.append(f"Tesseract OCR: OK (v{tess['version']})")
        if tess.get("has_turkish"):
            lines.append("  Turkce dil paketi: Yuklu")
        else:
            lines.append("  Turkce dil paketi: YUKLU DEGIL (tesseract-ocr-tur yukleyin)")
        lines.append(f"  Mevcut diller: {', '.join(tess.get('languages', []))}")
    else:
        lines.append(f"Tesseract OCR: YUKLU DEGIL ({tess.get('error', '')})")

    # NER (spaCy)
    try:
        import spacy
        lines.append(f"spaCy: Yuklu (v{spacy.__version__})")
        try:
            spacy.load("xx_ent_wiki_sm")
            lines.append("  NER model: xx_ent_wiki_sm yuklu")
        except OSError:
            lines.append("  NER model: Yuklu degil (python -m spacy download xx_ent_wiki_sm)")
    except ImportError:
        lines.append("spaCy: Yuklu degil (opsiyonel, NER modu icin)")

    # PyMuPDF
    try:
        import fitz
        lines.append(f"PyMuPDF: OK (v{fitz.version[0]})")
    except ImportError:
        lines.append("PyMuPDF: YUKLU DEGIL")

    # Database
    try:
        stats = db.get_anonymization_stats()
        lines.append(f"Veritabani: OK ({stats['total_anonymizations']} kayit)")
    except Exception as e:
        lines.append(f"Veritabani: HATA ({e})")

    return "\n".join(lines)


def build_ui():
    with gr.Blocks(title="KVKK Anonimizasyon", theme=gr.themes.Soft()) as app:
        gr.Markdown("# KVKK Uyumlu Tibbi Form Anonimizasyonu")
        gr.Markdown(
            "Anamnez ve patoloji formlarindan kisisel verileri temizler. "
            "Anonimize metin API LLM'lere guvenle gonderilebilir. Tum islem lokaldir."
        )

        with gr.Tab("Anonimizasyon"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Form Dosyasi (PDF / Goruntu / Metin)",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".txt"],
                    )
                    text_input = gr.Textbox(
                        label="veya Metin Yapistirin",
                        lines=8,
                        placeholder="Patoloji raporu veya anamnez formu metnini buraya yapistirin...",
                    )
                    anon_btn = gr.Button("Anonimize Et", variant="primary", size="lg")

                with gr.Column(scale=1):
                    status_out = gr.Textbox(label="Durum", interactive=False)

            with gr.Row():
                with gr.Column(scale=1):
                    original_out = gr.Textbox(
                        label="Orijinal Metin",
                        lines=15,
                        interactive=False,
                        show_copy_button=True,
                    )
                with gr.Column(scale=1):
                    anonymized_out = gr.Textbox(
                        label="Anonimize Metin (API'ye gonderilebilir)",
                        lines=15,
                        interactive=False,
                        show_copy_button=True,
                    )

            report_out = gr.Markdown(label="Anonimizasyon Raporu")

            # Indirme
            download_out = gr.Textbox(visible=False)
            download_btn = gr.DownloadButton(
                label="Anonimize Metni Indir (.txt)",
                visible=True,
            )

            anon_btn.click(
                fn=process_anonymization,
                inputs=[file_input, text_input],
                outputs=[status_out, original_out, anonymized_out, report_out, download_out],
            )

        with gr.Tab("Toplu Isleme"):
            gr.Markdown("Birden fazla dosyayi ayni anda anonimize edin.")
            batch_files = gr.File(
                label="Dosyalar",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".txt"],
            )
            batch_btn = gr.Button("Toplu Anonimize Et", variant="primary")
            batch_result = gr.Markdown()
            batch_download = gr.Textbox(visible=False)

            batch_btn.click(
                fn=process_batch,
                inputs=[batch_files],
                outputs=[batch_result, batch_download],
            )

        with gr.Tab("Istatistikler"):
            ref_btn = gr.Button("Yenile")
            stats_md = gr.Markdown()
            ref_btn.click(fn=get_stats, outputs=[stats_md])

        with gr.Tab("Sistem"):
            chk_btn = gr.Button("Kontrol Et")
            sys_out = gr.Textbox(label="Sistem Durumu", lines=10, interactive=False)
            chk_btn.click(fn=check_system, outputs=[sys_out])

    return app


def main():
    init_components()
    app = build_ui()
    app.launch(
        server_name=config["ui"]["host"],
        server_port=config["ui"]["port"],
        share=config["ui"]["share"],
    )


if __name__ == "__main__":
    main()
