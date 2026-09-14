"""Asenkron iş kuyruğu + SQLite iş deposu.

- Dosya baytları yalnızca RAM'deki kuyrukta tutulur; diske yazılmaz.
- Depoda: durum, metadata, sonuç JSON (ham metin asla yok). Dosya ADI da saklanmaz
  (hasta adı/TC içerebilir) — yalnızca uzantı ve müşterinin opsiyonel opak referansı (X-Ref).
- Tek süreç, N worker thread (GPU başına 1 önerilir). Kuyruk sınırlı (bellek DoS önlemi).
"""

import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .pipeline import Pipeline

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    status TEXT NOT NULL,          -- queued | processing | done | needs_review | failed
    filename TEXT NOT NULL,        -- yalnızca "<job_id_ilk8>.<uzantı>" (gerçek ad saklanmaz)
    ref TEXT DEFAULT '',           -- müşterinin opak referansı (X-Ref), saklanır ve aynen döner
    file_sha256 TEXT,
    size_bytes INTEGER,
    result_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            c.execute("PRAGMA journal_mode=WAL")
            # Şema göçü: eski DB'de ref sütunu yoksa ekle
            cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
            if "ref" not in cols:
                c.execute("ALTER TABLE jobs ADD COLUMN ref TEXT DEFAULT ''")
            # Önceki süreçten kalan bayat işler: dosya baytları RAM'deydi, kurtarılamaz → failed
            stale = c.execute(
                "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE status IN ('queued','processing')",
                (_now(), "servis yeniden başlatıldı; dosya yeniden gönderilmeli"),
            ).rowcount
            if stale:
                logger.warning("%d bayat iş 'failed' olarak işaretlendi", stale)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def create(self, suffix: str, size_bytes: int, ref: str = "") -> str:
        job_id = uuid.uuid4().hex
        filename = f"{job_id[:8]}{suffix}"
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO jobs (id, created_at, status, filename, size_bytes, ref) VALUES (?,?,?,?,?,?)",
                (job_id, _now(), "queued", filename, size_bytes, ref),
            )
        return job_id

    def mark_processing(self, job_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE jobs SET status='processing', started_at=? WHERE id=?", (_now(), job_id))

    def finish(self, job_id: str, status: str, result: dict, sha256: Optional[str], error: Optional[str]) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE jobs SET status=?, finished_at=?, result_json=?, file_sha256=?, error=? WHERE id=?",
                (status, _now(), json.dumps(result, ensure_ascii=False), sha256, error, job_id),
            )

    def get(self, job_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["result"] = json.loads(d.pop("result_json")) if d.get("result_json") else None
        return d

    def list(self, limit: int = 50, status: Optional[str] = None) -> list[dict]:
        q = "SELECT id, created_at, started_at, finished_at, status, filename, ref, size_bytes, error FROM jobs"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY created_at DESC LIMIT ?"
        with self._conn() as c:
            rows = c.execute(q, args + (limit,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}


class QueueFull(Exception):
    pass


class JobRunner:
    def __init__(self, pipeline: Pipeline, store: JobStore, workers: int = 1, max_queue: int = 20):
        self.pipeline = pipeline
        self.store = store
        if workers > 1:
            logger.warning("workers=%d: aynı LLM örneğinde çekişme/timeout riski; GPU başına 1 önerilir", workers)
        self.queue: "queue.Queue[tuple[str, bytes, str]]" = queue.Queue(maxsize=max_queue)
        self._threads = [threading.Thread(target=self._worker, name=f"worker-{i}", daemon=True) for i in range(workers)]
        self._stop = threading.Event()
        self._accepting = True

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def stop(self, grace_seconds: float = 30.0) -> None:
        """Yeni iş kabulünü kapat, işlenmekte olanın bitmesini (en fazla grace) bekle."""
        self._accepting = False
        self._stop.set()
        for t in self._threads:
            t.join(timeout=grace_seconds)

    def alive(self) -> int:
        return sum(1 for t in self._threads if t.is_alive())

    def submit(self, suffix: str, data: bytes, ref: str = "") -> str:
        if not self._accepting or self.queue.full():
            raise QueueFull()
        job_id = self.store.create(suffix, len(data), ref)
        try:
            self.queue.put_nowait((job_id, data, f"{job_id[:8]}{suffix}"))
        except queue.Full:
            self.store.finish(job_id, "failed", {}, None, "kuyruk dolu")
            raise QueueFull()
        return job_id

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job_id, data, filename = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run_one(job_id, data, filename)
            except BaseException as e:  # noqa: BLE001 — worker asla ölmesin
                logger.error("Worker beklenmedik hata (%s): %s", job_id, type(e).__name__)
            finally:
                del data  # ham baytları hemen bırak
                self.queue.task_done()

    def _run_one(self, job_id: str, data: bytes, filename: str) -> None:
        try:
            self.store.mark_processing(job_id)
        except Exception as e:  # noqa: BLE001
            logger.error("mark_processing hatası (%s): %s", job_id, type(e).__name__)
        try:
            res = self.pipeline.run(data, filename)
            status, result, sha, err = res.status, res.as_dict(), res.file_sha256, res.error
        except Exception as e:  # noqa: BLE001
            logger.error("İş %s pipeline hatası: %s", job_id, type(e).__name__)
            status, result, sha, err = "failed", {}, None, type(e).__name__
        for attempt in range(3):  # DB geçici kilitliyse yeniden dene
            try:
                self.store.finish(job_id, status, result, sha, err)
                return
            except Exception as e:  # noqa: BLE001
                logger.error("finish hatası (%s, deneme %d): %s", job_id, attempt + 1, type(e).__name__)
                time.sleep(0.5)
