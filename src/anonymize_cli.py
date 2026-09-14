"""CLI Arayüzü — Anamnez ve patoloji formlarını anonimize eder.

KVKK uyumlu anonimizasyon: Kişisel verileri temizleyerek
formları API LLM'lere gönderilebilir hale getirir.

Kullanım:
  python src/anonymize_cli.py -i rapor.pdf
  python src/anonymize_cli.py -d anamnes_files/ -o output/
  python src/anonymize_cli.py -i rapor.pdf --method ner
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_extractor import extract_from_file
from src.anonymizer import ReportAnonymizer
from src.db import Database
from src.models import AnonymizationRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".txt", ".text"}


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def anonymize_file(
    file_path: str,
    anonymizer: ReportAnonymizer,
    db: Database,
    config: dict,
    output_dir: Optional[Path] = None,
    method: str = "regex",
) -> dict:
    """Tek bir dosyayı anonimize et."""
    start_time = time.time()
    path = Path(file_path)
    suffix = path.suffix.lower()

    logger.info(f"İşleniyor: {path.name}")

    try:
        # 1. Metin çıkar
        raw_text = extract_from_file(str(path), config.get("pdf", {}))

        if len(raw_text.strip()) < 20:
            return {
                "filename": path.name,
                "error": "Yeterli metin çıkarılamadı",
                "original_length": len(raw_text),
            }

        # 2. Anonimize et
        if method == "ner":
            try:
                from src.ner_anonymizer import NERAnonymizer
                ner_anon = NERAnonymizer(use_ner=True, ner_backend="spacy")
                ner_result = ner_anon.anonymize(raw_text)
                anonymized_text = ner_result.anonymized_text
                fields_removed = ner_result.fields_removed
                fields_generalized = []
                warnings = []
            except ImportError:
                logger.warning("spaCy yüklü değil, regex moduna düşülüyor")
                method = "regex"

        if method == "regex":
            anonymized_text, report = anonymizer.anonymize(raw_text)
            fields_removed = report.fields_removed
            fields_generalized = report.fields_generalized
            warnings = report.warnings

        elapsed = time.time() - start_time

        # 3. Çıktı dosyası
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / f"{path.stem}_anonim.txt"
            out_file.write_text(anonymized_text, encoding="utf-8")

        # 4. Veritabanına logla
        record = AnonymizationRecord(
            filename=path.name,
            file_type=suffix.lstrip("."),
            original_length=len(raw_text),
            anonymized_length=len(anonymized_text),
            fields_removed_count=len(fields_removed),
            fields_removed=json.dumps(fields_removed, ensure_ascii=False),
            fields_generalized=json.dumps(fields_generalized, ensure_ascii=False),
            warnings=json.dumps(warnings, ensure_ascii=False),
            processing_time_seconds=round(elapsed, 2),
            method=method,
        )
        db.log_anonymization(record)

        result = {
            "filename": path.name,
            "original_length": len(raw_text),
            "anonymized_length": len(anonymized_text),
            "fields_removed": fields_removed,
            "fields_generalized": fields_generalized,
            "warnings": warnings,
            "processing_time": round(elapsed, 2),
            "method": method,
        }

        # Konsol çıktısı
        print(f"\n{'='*60}")
        print(f"  Dosya           : {path.name}")
        print(f"  Yöntem          : {method}")
        print(f"  Orijinal        : {len(raw_text)} karakter")
        print(f"  Anonimize       : {len(anonymized_text)} karakter")
        print(f"  Silinen alanlar : {len(fields_removed)} adet")
        for f in fields_removed[:8]:
            print(f"    - {f}")
        if len(fields_removed) > 8:
            print(f"    ... ve {len(fields_removed) - 8} tane daha")
        if fields_generalized:
            print(f"  Genelleştirilen : {', '.join(fields_generalized)}")
        if warnings:
            print(f"  Uyarılar        : {'; '.join(warnings)}")
        print(f"  Süre            : {elapsed:.2f}s")
        if output_dir:
            print(f"  Çıktı           : {output_dir / f'{path.stem}_anonim.txt'}")
        print(f"{'='*60}")

        return result

    except Exception as e:
        logger.error(f"İşleme hatası: {e}", exc_info=True)
        return {"filename": path.name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="KVKK Uyumlu Anonimizasyon — Anamnez ve Patoloji Formları",
        epilog="Örnekler:\n"
               "  python src/anonymize_cli.py -i rapor.pdf\n"
               "  python src/anonymize_cli.py -d anamnes_files/ -o output/\n"
               "  python src/anonymize_cli.py -d anamnes_files/ --method ner\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", help="Tek dosya yolu")
    parser.add_argument("--input-dir", "-d", help="Toplu işleme dizini")
    parser.add_argument("--output", "-o", help="Çıktı dizini (anonimize dosyalar)")
    parser.add_argument(
        "--method", "-m",
        default="regex",
        choices=["regex", "ner"],
        help="Anonimizasyon yöntemi: regex (hızlı) veya ner (daha doğru, spaCy gerektirir)",
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("--input veya --input-dir belirtilmeli")

    config = load_config()
    anon_config = config.get("anonymization", {})
    anonymizer = ReportAnonymizer(anon_config)
    db = Database(config["database"]["path"])
    output_dir = Path(args.output) if args.output else None

    print("\n" + "=" * 60)
    print("  KVKK Uyumlu Anonimizasyon Sistemi")
    print(f"  Yöntem: {args.method}")
    print("=" * 60)

    if args.input:
        anonymize_file(args.input, anonymizer, db, config, output_dir, args.method)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)

        if not files:
            print(f"Dizinde uygun dosya bulunamadı: {input_dir}")
            sys.exit(0)

        print(f"\n{len(files)} dosya bulundu. İşleniyor...\n")
        results = []
        for i, fp in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {fp.name}")
            results.append(anonymize_file(str(fp), anonymizer, db, config, output_dir, args.method))

        # Özet
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        total_removed = sum(len(r.get("fields_removed", [])) for r in successful)

        print(f"\n{'='*60}")
        print(f"  ÖZET")
        print(f"  Başarılı    : {len(successful)}/{len(results)}")
        if failed:
            print(f"  Başarısız   : {len(failed)}")
            for r in failed:
                print(f"    - {r['filename']}: {r['error']}")
        print(f"  Toplam PII  : {total_removed} alan silindi")
        if output_dir:
            print(f"  Çıktı dizini: {output_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
