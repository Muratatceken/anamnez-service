"""HTTP API — kapalı devre anonimizasyon + sınıflandırma servisi.

  POST /jobs            gövde = dosyanın ham baytları (application/octet-stream)
                        başlıklar: X-Filename (uzantı için; saklanmaz), X-Ref (opsiyonel müşteri referansı)
                        → 202 {job_id}
  GET  /jobs/{id}       durum + sonuç
  GET  /jobs            liste (status filtresi)
  GET  /health          bileşen durumu
  GET  /stats           iş sayaçları

Neden multipart değil: Starlette multipart ayrıştırıcısı 1 MB üstü parçaları geçici DOSYAYA spool eder
(ham hasta verisi diske iner). Ham gövde `request.stream()` ile yalnızca RAM'e okunur, boyut aşımında kesilir.

Kimlik doğrulama: X-API-Key başlığı (config server.api_key). Boşsa kapalı (yalnızca dev).
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import load_config
from .jobs import JobRunner, JobStore, QueueFull
from .pipeline import Pipeline
from .schemas import HealthResponse, JobCreated, JobListItem, JobResponse

logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".txt"}
HEALTH_CACHE_SECONDS = 5.0


class State:
    cfg: dict
    pipeline: Pipeline
    store: JobStore
    runner: JobRunner
    health_cache: tuple[float, dict] = (0.0, {})
    health_lock = threading.Lock()


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # her LLM isteğini loglama
    srv = state.cfg["server"]
    state.pipeline = Pipeline(state.cfg)
    state.store = JobStore(state.cfg["storage"]["db_path"])
    state.runner = JobRunner(
        state.pipeline, state.store,
        workers=int(srv.get("workers", 1)),
        max_queue=int(srv.get("max_queue", 20)),
    )
    state.runner.start()
    logger.info("Servis hazır — OCR: %s, LLM: %s", state.cfg["ocr"].get("backend"), state.cfg["llm"].get("model"))
    yield
    await run_in_threadpool(state.runner.stop, float(srv.get("shutdown_grace_seconds", 30)))


app = FastAPI(
    title="Anamnez Anonimizasyon & Sınıflandırma Servisi",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = (state.cfg.get("server", {}).get("api_key") or "").strip()
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Geçersiz API anahtarı")


async def _read_body_capped(request: Request, max_bytes: int) -> bytes:
    """Gövdeyi RAM'e oku; limit aşılırsa hemen 413 (diske spool YOK)."""
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > max_bytes:
                raise HTTPException(status_code=413, detail="Dosya çok büyük")
        except ValueError:
            raise HTTPException(status_code=400, detail="Geçersiz Content-Length")
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="Dosya çok büyük")
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/jobs", response_model=JobCreated, status_code=202, dependencies=[Depends(require_api_key)])
async def create_job(
    request: Request,
    x_filename: Optional[str] = Header(default=None),
    x_ref: Optional[str] = Header(default=None),
):
    suffix = Path(x_filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=415, detail=f"Desteklenmeyen dosya türü: {suffix or '(X-Filename başlığı gerekli)'}")
    max_bytes = int(state.cfg["server"].get("max_file_size_mb", 50)) * 1024 * 1024
    data = await _read_body_capped(request, max_bytes)
    if not data:
        raise HTTPException(status_code=400, detail="Boş dosya")
    ref = (x_ref or "")[:64]
    try:
        # SQLite yazımı + lock: event loop'u bloklamasın
        job_id = await run_in_threadpool(state.runner.submit, suffix, data, ref)
    except QueueFull:
        raise HTTPException(status_code=503, detail="Kuyruk dolu, daha sonra tekrar deneyin",
                            headers={"Retry-After": "30"})
    finally:
        del data
    return JobCreated(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
def get_job(job_id: str):
    job = state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="İş bulunamadı")
    return JobResponse(**job)


@app.get("/jobs", response_model=list[JobListItem], dependencies=[Depends(require_api_key)])
def list_jobs(limit: int = Query(50, ge=1, le=500), status: Optional[str] = None):
    return [JobListItem(**j) for j in state.store.list(limit=limit, status=status)]


@app.get("/stats", dependencies=[Depends(require_api_key)])
def stats():
    return {"jobs": state.store.stats(), "queue_depth": state.runner.queue.qsize(), "workers_alive": state.runner.alive()}


@app.get("/health", response_model=HealthResponse)
def health():
    now = time.monotonic()
    with state.health_lock:
        ts, cached = state.health_cache
        if cached and now - ts < HEALTH_CACHE_SECONDS:
            body = cached
        else:
            h = state.pipeline.health()
            ocr_ok = all(v.get("ok", False) for v in h["ocr"].values())
            workers_ok = state.runner.alive() > 0
            ok = ocr_ok and h["llm"].get("ok", False) and workers_ok
            body = {"ok": ok, "components": {**h, "workers_alive": state.runner.alive()}}
            state.health_cache = (now, body)
    return JSONResponse(status_code=200 if body["ok"] else 503, content=body)
