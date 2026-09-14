# Kapalı Devre Anonimizasyon & Sınıflandırma Servisi

Tek kiracılı (tek müşteri şirketi) servis. Müşteri dosya gönderir → servis lokal OCR,
anonimizasyon, doğrulama kapısı ve lokal LLM sınıflandırma yapar → JSON rapor döner.
**Hiçbir veri sunucu dışına çıkmaz.**

## Akış

```
POST /jobs (PDF/PNG/JPG/TXT)
  │
  ├─ 1. OCR         birincil: Vision-LLM (GLM-OCR / PaddleOCR-VL)   ikincil: Tesseract (paralel)
  │                 completeness = len(birincil)/len(tesseract) → <0.6 ise "olası atlama" uyarısı
  ├─ 2. Anonimize   regex + isim sözlüğü (src/anonymizer.py, 12 katman)
  ├─ 3. KAPI        a) residual_heuristics  — deterministik (etiket sonrası kalıntı, TC, tarih, tel, e-posta)
  │                 b) cross_ocr            — Tesseract'ın gördüğü PII nihai metinde var mı?
  │                 c) llm_judge            — Qwen3 "kalan PII var mı?" (JSON)
  │                 d) gliner (opsiyonel)   — neondijital/neonredact-tr-model
  │                 herhangi biri bulgu üretirse → status: needs_review, metin/sınıf DÖNMEZ
  ├─ 4. Sınıflandır lokal LLM (templates/classification_prompt.txt) + keyword validator
  └─ 5. Rapor       JSON: anonim metin, kategori, güven, gerekçe, kapı denetimi, zamanlamalar
```

## API

| Uç nokta | Açıklama |
|---|---|
| `POST /jobs` (gövde = ham dosya baytları, `X-Filename: rapor.pdf`, opsiyonel `X-Ref: <müşteri-ref>`) | İş oluştur → `202 {job_id}`; kuyruk doluysa `503` + `Retry-After` |
| `GET /jobs/{id}` | Durum + sonuç (`queued/processing/done/needs_review/failed`) |
| `GET /jobs?status=needs_review` | Liste |
| `GET /stats` | Sayaçlar + kuyruk derinliği |
| `GET /health` | Bileşen durumu (OCR, LLM) — 503 ise hazır değil |

Kimlik doğrulama: `X-API-Key` başlığı (`server.api_key` veya Docker secret `server.api_key_file`). Swagger: `/docs`.

Neden multipart değil: Starlette multipart ayrıştırıcısı 1 MB üstü parçaları geçici **dosyaya** spool eder —
ham hasta verisi diske iner. Ham gövde `request.stream()` ile yalnızca RAM'e okunur; `Content-Length`
limit üstündeyse gövde okunmadan `413`.

Gerçek dosya adı **saklanmaz** (hasta adı/TC içerebilir): `filename` alanı `<job_id_ilk8>.<uzantı>` olur;
korelasyon için `X-Ref` başlığıyla kendi opak referansınızı verin, aynen döner.

```bash
curl -H "X-API-Key: $KEY" -H "X-Filename: rapor.pdf" -H "X-Ref: HBYS-123456" \
     --data-binary @rapor.pdf http://10.0.0.5:8080/jobs
```

Örnek sonuç (`done`):
```json
{
  "status": "done",
  "ocr": {"engine": "glm-ocr", "alt_engine": "tesseract", "completeness": 0.86, "warnings": []},
  "anonymization": {"fields_removed": ["TC Kimlik No", "Hasta adı", "..."], "cross_ocr_candidates": 12},
  "gate": {"passed": true, "layers_run": ["cross_ocr", "residual_heuristics", "llm_judge"], "findings": []},
  "classification": {"validated_category": "Brain", "adjusted_confidence": 1.0, "histological_type": "Glioblastom", "reasoning": "..."},
  "anonymized_text": "...",
  "timings": {"ocr": 69.5, "anonymize": 0.1, "gate": 73.2, "classify": 73.4}
}
```

`needs_review`: sınıflandırma yapılmaz. `storage.review_text_mode`:
- `masked` (varsayılan): kapının yakaladığı kalıntılar da `[KAPI_BULGUSU_n:tür]` ile maskelenir; `gate.findings`
  yalnızca tür/kaynak/uzunluk taşır (bulgu METNİ dönmez).
- `full`: kalıntı içeren metin ve bulgu metni döner — yalnızca kapalı devre inceleme arayüzü için.
- `none`: metin dönmez.
`anonymization.fields_removed` yalnızca kategori adı içerir (silinen değerin kendisi asla yazılmaz).
Ham metin **hiçbir durumda** dönmez/loglanmaz; hata mesajları yalnızca istisna tipi taşır.

## Çalıştırma

### Geliştirme (Mac, Ollama)
```bash
/opt/homebrew/bin/ollama serve          # ARM64 Ollama (Intel binary Metal kullanamaz!)
ollama pull glm-ocr && ollama pull qwen3:8b
ANAMNEZ_SERVER__API_KEY=devkey .venv/bin/python -m service
curl -H "X-API-Key: devkey" -H "X-Filename: rapor.pdf" --data-binary @rapor.pdf http://127.0.0.1:8080/jobs
```
Konfig: `config/service.yaml`; her anahtar `ANAMNEZ_<BÖLÜM>__<ANAHTAR>` ile ezilebilir.

### Üretim (Linux + NVIDIA, kapalı devre)
1. **Hazırlık makinesi (internetli):** `scripts/prepare_models.sh bundle/` → model ağırlıkları + Docker
   imajları + SHA256SUMS. USB/imzalı arşivle sunucuya taşı.
2. **Sunucu:** `sha256sum -c SHA256SUMS && docker load < images/images.tar.gz`
3. `openssl rand -hex 32 > deploy/api_key.txt` (Docker secret; repo'ya girmez)
4. `sudo nft -f deploy/nftables.conf` (LAN/MGMT CIDR'larını düzenle) — Docker köprüleri hariç **outbound deny-all**;
   Docker'ın kendi nft tablolarına dokunmaz (`flush ruleset` yok), systemd'de `After=docker.service` ile sırala
5. `VLLM_TAG=<model kartındaki min sürüm> API_BIND=10.0.0.5 docker compose -f deploy/docker-compose.yml up -d`
6. Kanıt: `EDGE_IF=eth0 scripts/verify_no_egress.sh 300` → 0 dış paket; raporu müşteriye ver.

Ağ: vllm-* yalnızca `internal` (egress yok) ağda; `api` ek olarak `edge` ağında (Docker, yalnızca internal ağdaki
konteynerde port publish'i desteklemez). Sabit alt ağlar 10.200.1/2.0/24 → nftables/verify bunları "iç" sayar.

Docker ağı `internal: true` → konteynerler internete çıkamaz (docker seviyesi), nftables → host
seviyesi. İki bağımsız engel.

## Güvenlik ilkeleri (kodda uygulanmış)
- Ham dosya diske yazılmaz: multipart spool yok (raw body), tesseract `stdin→stdout` (pytesseract'ın
  geçici dosyaları yok), iş kuyruğu RAM'de (sınırlı: `server.max_queue`), DB'de yalnızca metadata + sonuç.
- Servis çökerse kuyruktaki işler yeniden başlatmada `failed` olur (baytlar RAM'deydi) — istemci yeniden gönderir.
- Kapı: katman yoksa/hata verirse fail-closed; LLM bulgusu deterministik filtrelerden geçer (SUT/ICD kodları elenir,
  maske etiketi içeren bulgular sıyrılıp değerlendirilir, Türkçe İ/ı casefold).
- `logger.error(..., exc_info=True)` **kullanılmaz** — traceback ham metin taşıyabilir.
- Uvicorn access log kapalı (dosya adları loglanmaz).
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK`, `GRADIO_ANALYTICS_ENABLED=False`
  servis başlarken zorlanır (`service/config.py`).
- httpx istemcileri `trust_env=False` — sistem proxy'si üzerinden sızma yok.
- API konteyneri `read_only`, `cap_drop: ALL`, root değil.

## Bilinen sınırlar / yapılacaklar
- **VLM-OCR içerik atlayabiliyor** (GLM-OCR Ollama GGUF, iki sütunlu başlıkta sol sütunu düşürdü).
  Çift OCR + cross_ocr katmanı bunu *sızıntı* açısından kapatır ama *veri kaybı* (yaş, cinsiyet, doku
  yeri) devam edebilir. Üretimde vLLM (tam hassasiyet) ile bake-off şart; alternatif: PaddleOCR-VL-1.6.
- LLM yargıç (qwen3:8b) tek başına güvenilmez — "[DOKTOR_SILINDI] ERDEM" kalıntısını kaçırdı;
  deterministik katmanlar bu yüzden var. Üretimde Qwen3-32B/AWQ değerlendirilebilir.
- M1'de iş başına ~3.5 dk; L4/4090'da vLLM ile ~10-20 sn beklenir.
- GLiNER katmanı (`gate.gliner: true`) için `pip install gliner` + ağırlıklar offline sağlanmalı.
- Regresyon seti: `tests/test_anonymizer.py` (gerçek 8 dosya + sentetik), `tests/test_service.py`.
  Yeni her müşteri dosyası (anonim OCR çıktısı) `output/originals`'a eklenip test kapsamına alınmalı.
