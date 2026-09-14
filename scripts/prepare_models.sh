#!/usr/bin/env bash
# Hazırlık makinesinde (İNTERNETLİ, GPU'lu tercihen) çalıştırılır. Çıktı: ./bundle/
# → USB / imzalı arşiv ile kapalı devre sunucuya taşınır. Sunucuda hiçbir indirme yapılmaz.
#
#   scripts/prepare_models.sh bundle/            # varsayılan: GLM-OCR + Qwen3-8B-AWQ
#   LLM_REPO=Qwen/Qwen3-32B-AWQ scripts/prepare_models.sh bundle/   # 48 GB+ GPU için
#   OCR_REPO=PaddlePaddle/PaddleOCR-VL-1.6 ...                        # bake-off alternatifi
set -euo pipefail
OUT=${1:-bundle}
VLLM_TAG=${VLLM_TAG:-v0.19.1}
OCR_REPO=${OCR_REPO:-zai-org/GLM-OCR}
LLM_REPO=${LLM_REPO:-Qwen/Qwen3-8B-AWQ}
mkdir -p "$OUT/models" "$OUT/images"

echo "==> Hugging Face CLI"
python3 -m pip install -q -U "huggingface_hub[cli]>=0.34"
HF=$(command -v hf || command -v huggingface-cli)

echo "==> Model ağırlıkları"
"$HF" download "$OCR_REPO" --local-dir "$OUT/models/glm-ocr"
"$HF" download "$LLM_REPO" --local-dir "$OUT/models/qwen3-8b-awq"
# Opsiyonel GLiNER katmanı (gate.gliner: true):
# "$HF" download neondijital/neonredact-tr-model --local-dir "$OUT/models/neonredact-tr"

echo "==> Docker imajları"
docker pull "vllm/vllm-openai:$VLLM_TAG"
docker build -f deploy/Dockerfile -t anamnez-service:1.1.0 .

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> Smoke test: vLLM bu ağırlıkları yükleyebiliyor mu? (GPU var)"
  docker run --rm --gpus all -v "$PWD/$OUT/models:/models:ro" "vllm/vllm-openai:$VLLM_TAG" \
    --model /models/glm-ocr --max-model-len 4096 --gpu-memory-utilization 0.3 --port 8000 &
  PID=$!; sleep 120; kill $PID || true
  echo "   (yukarıda 'Application startup complete' görülmeliydi)"
else
  echo "!! GPU yok: smoke test atlandı — sunucuda ilk açılışta 'docker compose logs vllm-ocr' izleyin"
fi

docker save "vllm/vllm-openai:$VLLM_TAG" anamnez-service:1.1.0 | gzip > "$OUT/images/images.tar.gz"

echo "==> Kaynak kod kopyası (config/, deploy/, scripts/ — sunucuda compose için)"
mkdir -p "$OUT/src"; cp -r deploy scripts config docs "$OUT/src/"

echo "==> Bütünlük"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | xargs -0 sha256sum > SHA256SUMS )
echo "Hazır: $OUT"
echo "Sunucuda: sha256sum -c SHA256SUMS && docker load < images/images.tar.gz && cp -r models src/deploy/"
