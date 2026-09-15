#!/usr/bin/env bash
# Hazırlık makinesinde (internetli) çalıştırılır → ./bundle/  → sunucuya taşınır. Sunucuda indirme yapılmaz.
#
#   scripts/prepare_models.sh bundle/               # HİBRİT (varsayılan): GLiNER-tr + CPU API imajı + squid
#   MODE=gpu scripts/prepare_models.sh bundle/      # Kapalı devre GPU: + GLM-OCR + Qwen3-8B-AWQ + vLLM imajı
set -euo pipefail
OUT=${1:-bundle}
MODE=${MODE:-hybrid}
VLLM_TAG=${VLLM_TAG:-v0.19.1}
NER_REPO=${NER_REPO:-neondijital/neonredact-tr-model}
OCR_REPO=${OCR_REPO:-zai-org/GLM-OCR}
LLM_REPO=${LLM_REPO:-Qwen/Qwen3-8B-AWQ}
mkdir -p "$OUT/models" "$OUT/images"

python3 -m pip install -q -U "huggingface_hub[cli]>=0.34"
HF=$(command -v hf || command -v huggingface-cli)

echo "==> GLiNER Türkçe PII modeli"
"$HF" download "$NER_REPO" --local-dir "$OUT/models/neonredact-tr"
# GLiNER tokenizer'ı (mDeBERTa) da cache'lenmeli: model klasörü kendi tokenizer dosyalarını içerir; doğrula
python3 - <<PY
import json,sys,pathlib
p=pathlib.Path("$OUT/models/neonredact-tr"); cfg=json.loads((p/"gliner_config.json").read_text())
print("   base:", cfg.get("model_name")); print("   dosyalar:", sorted(f.name for f in p.iterdir())[:12])
PY

echo "==> Docker imajları"
docker build -f deploy/Dockerfile -t anamnez-service:1.2.0 .
docker pull ubuntu/squid:6.6-24.04_beta
IMAGES="anamnez-service:1.2.0 ubuntu/squid:6.6-24.04_beta"

if [ "$MODE" = "gpu" ]; then
  echo "==> GPU modelleri"
  "$HF" download "$OCR_REPO" --local-dir "$OUT/models/glm-ocr"
  "$HF" download "$LLM_REPO" --local-dir "$OUT/models/qwen3-8b-awq"
  docker pull "vllm/vllm-openai:$VLLM_TAG"
  IMAGES="$IMAGES vllm/vllm-openai:$VLLM_TAG"
fi

docker save $IMAGES | gzip > "$OUT/images/images.tar.gz"

echo "==> Kaynak kopyası (config/, deploy/, scripts/, docs/, client/)"
mkdir -p "$OUT/src"; cp -r deploy scripts config docs client "$OUT/src/"
rm -f "$OUT/src/deploy/api_key.txt" "$OUT/src/deploy/anthropic_key.txt"

( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | xargs -0 sha256sum > SHA256SUMS )
echo "Hazır: $OUT   (sunucuda: sha256sum -c SHA256SUMS && docker load < images/images.tar.gz)"
