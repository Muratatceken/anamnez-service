# Anamnez — Kapalı Devre Tıbbi Form Anonimizasyon & Sınıflandırma Servisi

KVKK uyumlu, **tamamen kapalı devre** (air-gapped) çalışan servis. Taranmış anamnez/patoloji formlarını
alır → lokal OCR → kişisel verileri maskeler → bağımsız doğrulama kapısından geçirir → lokal LLM ile
kanser kategorisini belirler → JSON rapor döner. **Hiçbir veri sunucu dışına çıkmaz; ham veri diske yazılmaz.**

```
POST /jobs ──► OCR (Vision-LLM + Tesseract çapraz) ──► Regex anonimizasyon ──► KAPI ──► LLM sınıflandırma ──► JSON
                                                                          │ (deterministik + çapraz-OCR + LLM yargıç)
                                                                          └─ bulgu varsa: needs_review (fail-closed)
```

## Özellikler
- **Çift OCR:** Vision-LLM (GLM-OCR / PaddleOCR-VL) birincil, Tesseract paralel; Tesseract'ın gördüğü PII nihai
  metinde aranır (VLM'in atladığı satırlar sızamaz), içerik atlama oranı raporlanır.
- **12 katmanlı regex anonimizasyon:** TC (maskeli dahil), ad-soyad (etiketli + Türk isim sözlüğü), doktor/unvan,
  tarih (7 format), yaş (aralığa genelleme), kurum/hastane, 81 il, telefon/e-posta/adres, diploma/tescil, protokol no, imza.
- **Fail-closed doğrulama kapısı:** 3 bağımsız katman; herhangi biri bulgu üretirse sınıflandırma yapılmaz,
  sonuç `needs_review` olur ve kalıntı da maskelenir.
- **Lokal sınıflandırma:** 33 kanser kategorisi, keyword doğrulama, Türkçe kategori adı eşleme.
- **KVKK tasarımı:** ham dosya/metin hiçbir yerde loglanmaz/saklanmaz; multipart spool yok; tesseract stdin/stdout;
  dosya adı saklanmaz; hata mesajları yalnızca istisna tipi taşır; telemetri env'leri zorla kapalı.
- **Kapalı devre dağıtım:** Docker `internal` ağ + host nftables (çift egress engeli), offline model paketi,
  imzalı bundle, `verify_no_egress.sh` kanıt scripti.
- **Pluggable backend:** `ollama` (geliştirme, Mac) ↔ `openai` (vLLM, üretim) tek config değişikliği.

## Hızlı başlangıç (geliştirme, Mac/Linux + Ollama)
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-service.txt pytest
brew install tesseract tesseract-lang          # Linux: apt install tesseract-ocr tesseract-ocr-tur
ollama serve & ollama pull glm-ocr && ollama pull qwen3:8b
make test                                       # 69 test, Ollama gerekmez
make run                                        # http://127.0.0.1:8080  (API key: devkey)
python client/anamnez_client.py --url http://127.0.0.1:8080 --key devkey rapor.pdf
```
Mac M1 notu: Ollama **ARM64** olmalı (`/opt/homebrew/bin/ollama`); Intel binary Metal GPU kullanamaz.

## Üretim (kapalı devre, Linux + NVIDIA)
Adım adım: **[docs/KURULUM_REHBERI.md](docs/KURULUM_REHBERI.md)**. Özet:
1. İnternetli hazırlık makinesinde `scripts/prepare_models.sh bundle/` → modeller + imajlar + SHA256SUMS
2. Bundle'ı sunucuya taşı, `docker load`, `deploy/api_key.txt` üret
3. `nft -f deploy/nftables.conf` (egress kapalı) → `docker compose -f deploy/docker-compose.yml up -d`
4. `scripts/verify_no_egress.sh 300` ile "0 dış paket" kanıtı

## API
| | |
|---|---|
| `POST /jobs` | gövde = ham dosya, `X-Filename`, `X-Ref` (opsiyonel), `X-API-Key` → `202 {job_id}` |
| `GET /jobs/{id}` | `queued / processing / done / needs_review / failed` + sonuç |
| `GET /jobs?status=` · `GET /stats` · `GET /health` · `/docs` | |

Detay ve örnek yanıt: [docs/SERVIS.md](docs/SERVIS.md). Python istemci: [client/anamnez_client.py](client/anamnez_client.py).

## Depo yapısı
```
service/            FastAPI servis: api, jobs (kuyruk+SQLite), pipeline, backends/{llm,ocr}, anonymization/gate
src/                anonymizer (regex), validator, models, ner_anonymizer (ops.), CLI & Gradio aracı (eski)
config/             service.yaml (servis), turkish_names.json, categories.json
deploy/             docker-compose.yml, Dockerfile, nftables.conf, systemd/, .env.example
scripts/            prepare_models.sh (offline bundle), verify_no_egress.sh
client/             Python istemci + CLI
tests/              69 test: anonimizasyon regresyonu, kapı, API, sınıflandırma (Ollama gerekmez)
docs/               KURULUM_REHBERI.md, SERVIS.md
```

## Bilinen sınırlar (dürüst tablo)
- Küçük yargıç modeli (qwen3:8b) yanlış pozitif üretir → gereksiz `needs_review`. Üretimde Qwen3-32B-AWQ önerilir.
- Vision-LLM OCR bazen satır/sütun atlar; sızıntı çapraz-OCR ile engellenir ama **veri kaybı** olabilir
  (`ocr.completeness` < 0.6 uyarısı). vLLM'de GLM-OCR / PaddleOCR-VL bake-off yapılmalı.
- %100 garanti yoktur; sistem "emin değilse durdur" ilkesiyle çalışır. `needs_review` kuyruğu insan tarafından
  görülmelidir.

## Lisans
Tüm hakları saklıdır. Kullanılan açık modeller: GLM-OCR (MIT), Qwen3 (Apache-2.0), Tesseract (Apache-2.0).
