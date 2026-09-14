# API konteyneri — kapalı devre. Build aşaması internet ister (pip); çalışma aşaması istemez.
# tesseract stdin/stdout ile çağrılır (pytesseract yok → geçici dosya yok).
# Build'i hazırlık makinesinde yapıp `docker save` ile taşıyın (bkz. scripts/prepare_models.sh).
FROM python:3.12-slim AS base

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src/ ./src/
COPY service/ ./service/
COPY config/ ./config/
COPY templates/ ./templates/

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ANAMNEZ_CONFIG=/app/config/service.yaml

RUN useradd -r -u 10001 svc && mkdir -p /data && chown svc /data
USER svc
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -sf http://localhost:8080/health || exit 1
CMD ["python", "-m", "service"]
