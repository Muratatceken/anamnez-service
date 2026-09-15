# API konteyneri — hibrit/CPU. tesseract (stdin/stdout) + GLiNER (torch CPU) + Anthropic SDK.
# Build aşaması internet ister; çalışma aşaması yalnızca egress-proxy üzerinden api.anthropic.com'a çıkar.
FROM python:3.12-slim AS base

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-service.txt .
# torch CPU tekerleği (CUDA'sız, ~200 MB) — GLiNER için
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<3" \
 && pip install --no-cache-dir -r requirements-service.txt

COPY src/ ./src/
COPY service/ ./service/
COPY config/ ./config/
COPY templates/ ./templates/

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    ANAMNEZ_CONFIG=/app/config/service.yaml

RUN useradd -r -u 10001 svc && mkdir -p /data /models && chown svc /data
USER svc
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -sf http://localhost:8080/health || exit 1
CMD ["python", "-m", "service"]
