"""Veritabanı Modülü - SQLite ile sınıflandırma sonuçlarını loglama."""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ProcessingRecord, AnonymizationRecord

logger = logging.getLogger(__name__)


class Database:
    """SQLite veritabanı yönetimi."""

    def __init__(self, db_path: str = "data/classification_log.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Veritabanı tablolarını oluştur."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                ocr_engine_used TEXT,
                extracted_text_length INTEGER DEFAULT 0,
                classification_category TEXT NOT NULL,
                classification_confidence REAL NOT NULL,
                validation_passed INTEGER NOT NULL,
                final_category TEXT NOT NULL,
                processing_time_seconds REAL NOT NULL,
                warnings TEXT DEFAULT '',
                llm_model TEXT DEFAULT ''
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS category_stats (
                category TEXT PRIMARY KEY,
                total_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                last_updated TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anonymization_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                original_length INTEGER DEFAULT 0,
                anonymized_length INTEGER DEFAULT 0,
                fields_removed_count INTEGER DEFAULT 0,
                fields_removed TEXT DEFAULT '',
                fields_generalized TEXT DEFAULT '',
                warnings TEXT DEFAULT '',
                processing_time_seconds REAL DEFAULT 0.0,
                method TEXT DEFAULT 'regex'
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Veritabanı hazır: {self.db_path}")

    def log_result(self, record: ProcessingRecord) -> int:
        """Sınıflandırma sonucunu kaydet."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO classification_log 
            (timestamp, filename, file_type, ocr_engine_used, extracted_text_length,
             classification_category, classification_confidence, validation_passed,
             final_category, processing_time_seconds, warnings, llm_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.timestamp.isoformat(),
            record.filename,
            record.file_type,
            record.ocr_engine_used,
            record.extracted_text_length,
            record.classification_category,
            record.classification_confidence,
            1 if record.validation_passed else 0,
            record.final_category,
            record.processing_time_seconds,
            record.warnings,
            record.llm_model,
        ))

        record_id = cursor.lastrowid

        # Kategori istatistiklerini güncelle
        cursor.execute("""
            INSERT INTO category_stats (category, total_count, avg_confidence, last_updated)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(category) DO UPDATE SET
                total_count = total_count + 1,
                avg_confidence = (avg_confidence * total_count + ?) / (total_count + 1),
                last_updated = ?
        """, (
            record.final_category,
            record.classification_confidence,
            datetime.now().isoformat(),
            record.classification_confidence,
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

        logger.info(f"Sonuç kaydedildi (ID: {record_id})")
        return record_id

    def log_anonymization(self, record: AnonymizationRecord) -> int:
        """Anonimizasyon sonucunu kaydet."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO anonymization_log
            (timestamp, filename, file_type, original_length, anonymized_length,
             fields_removed_count, fields_removed, fields_generalized,
             warnings, processing_time_seconds, method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.timestamp.isoformat(),
            record.filename,
            record.file_type,
            record.original_length,
            record.anonymized_length,
            record.fields_removed_count,
            record.fields_removed,
            record.fields_generalized,
            record.warnings,
            record.processing_time_seconds,
            record.method,
        ))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Anonimizasyon kaydedildi (ID: {record_id})")
        return record_id

    def get_anonymization_stats(self) -> dict:
        """Anonimizasyon istatistiklerini getir."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM anonymization_log")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(fields_removed_count) FROM anonymization_log")
        avg_removed = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT AVG(processing_time_seconds) FROM anonymization_log")
        avg_time = cursor.fetchone()[0] or 0.0

        cursor.execute("""
            SELECT * FROM anonymization_log
            ORDER BY timestamp DESC LIMIT 20
        """)
        columns = [desc[0] for desc in cursor.description]
        recent = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()

        return {
            "total_anonymizations": total,
            "avg_fields_removed": round(avg_removed, 1),
            "avg_processing_time": round(avg_time, 2),
            "recent": recent,
        }

    def get_stats(self) -> dict:
        """Genel istatistikleri getir."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM classification_log")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(classification_confidence) FROM classification_log")
        avg_conf = cursor.fetchone()[0] or 0.0

        cursor.execute("""
            SELECT final_category, COUNT(*) as cnt 
            FROM classification_log 
            GROUP BY final_category 
            ORDER BY cnt DESC
        """)
        category_counts = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT SUM(validation_passed) FROM classification_log")
        valid_count = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_classifications": total,
            "average_confidence": round(avg_conf, 3),
            "validation_pass_rate": round(valid_count / total, 3) if total > 0 else 0,
            "category_distribution": category_counts,
        }

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Son sınıflandırmaları getir."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM classification_log 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def export_csv(self, output_path: str):
        """Tüm sonuçları CSV olarak dışa aktar."""
        import csv

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM classification_log ORDER BY timestamp")

        rows = cursor.fetchall()
        if not rows:
            logger.warning("Dışa aktarılacak veri yok")
            return

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([dict(row) for row in rows])

        conn.close()
        logger.info(f"CSV dışa aktarıldı: {output_path}")
