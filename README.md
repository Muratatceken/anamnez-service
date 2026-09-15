# Anamnez — KVKK Uyumlu Tıbbi Form Anonimizasyon, Rapor & Sınıflandırma Servisi

Taranmış anamnez/patoloji formlarını alır → **Türkiye'deki sunucuda** OCR + anonimizasyon (kişisel veri hiç
dışarı çıkmaz) → fail-closed doğrulama kapısı → yalnızca **anonim** metin, tek bir pinhole üzerinden Claude'a
gider → doktor için **kısa yapılandırılmış rapor + 33 kategoride kanser sınıflandırması** döner.

```
POST /jobs ─► OCR (Tesseract; GPU varsa Vision-LLM) ─► GLiNER-tr NER + regex taban ─► KAPI ─► EGRESS GATEWAY ─► Claude
                                                                                     │                │ (yalnızca api.anthropic.com:443)
                                                                                     └ bulgu → needs_review (fail-closed)
```

İki dağıtım modu, aynı kod:
| Mod | OCR | Anonimizasyon | Rapor/sınıflandırma | Dışarı çıkan |
|---|---|---|---|---|
| **Hibrit (varsayılan, GPU gerekmez)** | Tesseract (CPU) | GLiNER-tr + regex (CPU) | Claude API | yalnızca anonim metin (hash'li denetim kaydı) |
| Kapalı devre (GPU) | GLM-OCR / PaddleOCR-VL (vLLM) | GLiNER-tr + regex + LLM yargıç | lokal Qwen3 | hiçbir şey |

## Özellikler
- **NER + regex anonimizasyon:** GLiNER Türkçe PII modeli (etiketsiz/bozuk etiketli isimler, şehirler, kurumlar;
  CPU'da ~1 sn) + deterministik regex tabanı (TC, tarih, telefon, tescil, unvan, 81 il). Sentetik sette
  140/140 PII, 0 tıbbi terim kaybı. Tıbbi sözlük koruması (tanı satırları silinmez).
- **Fail-closed doğrulama kapısı:** deterministik kalıntı avcısı + çapraz-OCR + NER adayları (+ GPU'da LLM yargıç);
  bulgu varsa sonuç `needs_review`, kalıntı da maskelenir. Sentetik el yazısı benchmark'ı: **kapı PASS + sızıntı = 0** hedefi CI'da.
- **Egress gateway:** buluta çıkmadan önce son kontrol (kapı geçti mi, bilinen PII dizesi var mı, TC/tarih/telefon
  kalıntısı var mı); her çıkış hash+boyut+hedef ile denetim tablosuna yazılır (metin yazılmaz). Ağda squid allowlist +
  nftables: yalnızca proxy konteyneri `api.anthropic.com:443`'e çıkabilir.
- **Doktor raporu (Claude, yapılandırılmış JSON):** kategori + güven + gerekçe, histolojik tip, primer bölge,
  evre/derece, belirteçler, önemli bulgular, tedavi/plan, ≤3 cümle özet, belirsizlikler; keyword doğrulama.
- **KVKK tasarımı:** ham dosya/metin hiçbir yerde loglanmaz/saklanmaz; multipart spool yok; tesseract stdin/stdout;
  dosya adı saklanmaz; hata mesajları yalnızca istisna tipi taşır; telemetri env'leri zorla kapalı.
- **Kapalı devre dağıtım:** Docker `internal` ağ + host nftables (çift egress engeli), offline model paketi,
  imzalı bundle, `verify_no_egress.sh` kanıt scripti.
- **Pluggable backend:** `ollama` (geliştirme, Mac) ↔ `openai` (vLLM, üretim) tek config değişikliği.

## Hızlı başlangıç (geliştirme)
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-service.txt torch pytest
brew install tesseract tesseract-lang          # Linux: apt install tesseract-ocr tesseract-ocr-tur
make test                                       # ~80 test, ağ/GPU gerekmez
export ANTHROPIC_API_KEY=sk-ant-...             # bulut raporu için
ANAMNEZ_CLOUD__ENABLED=true ANAMNEZ_LLM__ENABLED=false make run     # http://127.0.0.1:8080 (key: devkey)
python client/anamnez_client.py --url http://127.0.0.1:8080 --key devkey rapor.pdf
```
GLiNER modeli ilk çalıştırmada HF'den iner (`neondijital/neonredact-tr-model`, ~1 GB); üretimde bundle ile offline.

## Üretim — hibrit (GPU'suz Linux sunucu + Claude API)
Adım adım: **[docs/KURULUM_REHBERI.md](docs/KURULUM_REHBERI.md)**. Özet:
1. Hazırlık makinesinde `scripts/prepare_models.sh bundle/` → GLiNER + imajlar + SHA256SUMS
2. Sunucuda `docker load`; `deploy/api_key.txt` ve `deploy/anthropic_key.txt` üret
3. `nft -f deploy/nftables.cpu.conf` → `docker compose -f deploy/docker-compose.cpu.yml up -d`
4. `scripts/verify_no_egress.sh 300`: yalnızca proxy→api.anthropic.com:443 görülmeli

Kapalı devre GPU modu: `MODE=gpu scripts/prepare_models.sh`, `deploy/docker-compose.yml`, `deploy/nftables.conf`.

## KVKK notu
Kişisel veri (ham tarama, OCR metni, NER adayları) yalnızca Türkiye'deki sunucuda işlenir ve diske yazılmaz.
Claude'a giden metin anonimleştirilmiştir; kimliklendirme riski kapı + egress gateway ile fail-closed sınırlanır ve
her çıkış hash'iyle denetlenebilir. Anonimleştirmenin etkinliği `bench/` ile ölçülür (DPIA kanıtı).
`review_text_mode: masked` ile `needs_review` sonuçlarında kalıntılar da maskelenir.

## API
| | |
|---|---|
| `POST /jobs` | gövde = ham dosya, `X-Filename`, `X-Ref` (opsiyonel), `X-API-Key` → `202 {job_id}` |
| `GET /jobs/{id}` | `queued / processing / done / needs_review / failed` + sonuç |
| `GET /jobs?status=` · `GET /stats` · `GET /health` · `/docs` | |

Detay ve örnek yanıt: [docs/SERVIS.md](docs/SERVIS.md). Python istemci: [client/anamnez_client.py](client/anamnez_client.py).

## Depo yapısı
```
service/            FastAPI servis: api, jobs, pipeline, egress (gateway), report (Claude), backends/{llm,ocr},
                    anonymization/{ner (GLiNER), gate}
src/                anonymizer (regex), validator, models, ner_anonymizer (ops.), CLI & Gradio aracı (eski)
config/             service.yaml (servis), turkish_names.json, categories.json
deploy/             docker-compose.cpu.yml (hibrit) + docker-compose.yml (GPU), Dockerfile, nftables.{cpu,}.conf, squid/, systemd/
scripts/            prepare_models.sh (offline bundle), verify_no_egress.sh
client/             Python istemci + CLI
tests/              69 test: anonimizasyon regresyonu, kapı, API, sınıflandırma (Ollama gerekmez)
docs/               KURULUM_REHBERI.md, SERVIS.md
```

## Bilinen sınırlar (dürüst tablo)
- **El yazısında Tesseract zayıf** (sentetik sette CER ~0.16, PII okuma ~%54): okunamayan PII sızmaz ama rapor
  eksik kalabilir. GPU-VLM OCR (GLM-OCR) aynı sette CER ~0.02 — bütçe çıkınca yurt içi GPU'ya taşınır (kod hazır).
- NER modeli sentetik veriyle eğitilmiş; gerçek form şablonlarında yeni varyantlar çıkabilir → `bench/` ile ölçüp
  `tests/` ile kilitle. %100 garanti yoktur; sistem "emin değilse durdur" ilkesiyle çalışır.
- `needs_review` kuyruğu insan tarafından görülmelidir.

## Lisans
Tüm hakları saklıdır. Kullanılan açık modeller: GLM-OCR (MIT), Qwen3 (Apache-2.0), Tesseract (Apache-2.0).
