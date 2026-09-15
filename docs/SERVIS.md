# Anonimizasyon, Doktor Raporu & Sınıflandırma Servisi

Tek kiracılı servis. Müşteri dosya gönderir → sunucuda (Türkiye) OCR + anonimizasyon + doğrulama kapısı →
yalnızca anonim metin egress gateway üzerinden Claude'a → doktor raporu + sınıflandırma JSON döner.
**Kişisel veri sunucu dışına çıkmaz; ham veri diske yazılmaz.**

## Akış

```
POST /jobs (PDF/PNG/JPG/TXT)
  │
  ├─ 1. OCR         Tesseract (CPU). GPU modunda: Vision-LLM birincil + Tesseract çapraz (completeness uyarısı)
  ├─ 2. Anonimize   a) GLiNER-tr NER (service/anonymization/ner.py): isim/şehir/kurum/TC/tel/tarih — etiketsiz de
  │                    tam metin + BÜYÜK HARF için harf-duyarlı geçiş + başlık bölgesi dar-etiket geçişi; tıbbi sözlük koruması
  │                 b) regex + isim sözlüğü (src/anonymizer.py): TC/tarih/tescil/unvan/81 il (OCR-bulanık dahil), kalıntı temizliği
  ├─ 3. KAPI        residual_heuristics (deterministik) + cross_ocr (NER adayları + ikincil OCR'ın gördüğü PII nihai metinde var mı?)
  │                 [+ llm_judge: yalnızca lokal LLM varsa]   → bulgu/hata varsa: needs_review (fail-closed)
  ├─ 4. EGRESS      service/egress.py: kapı geçti mi + hiçbir PII adayı çıktıda yok + TC/tarih/tel/e-posta/URL yok
  │                 → red: needs_review; kabul: hash+boyut+hedef+model denetim tablosuna (metin yazılmaz)
  ├─ 5. Claude      claude-opus-5, yapılandırılmış JSON (DoctorReport): kategori, güven, gerekçe, histolojik tip,
  │                 primer bölge, evre/derece, belirteçler, önemli bulgular, tedavi/plan, ≤3 cümle özet, belirsizlikler
  │                 + keyword doğrulama (src/validator.py). Ağ: squid allowlist → yalnızca api.anthropic.com:443
  └─ 6. Sonuç       JSON: anonim metin, report (yapılandırılmış + markdown), classification, gate, egress, timings
```

`cloud.enabled: false` iken (kapalı devre GPU modu) 4-5 yerine lokal LLM sınıflandırması çalışır.

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
  "egress": {"allowed": true, "text_sha256": "…", "chars": 812, "host": "api.anthropic.com", "model": "claude-opus-5"},
  "report": {"rapor": {"kategori": "Brain", "guven": 0.92, "histolojik_tip": "Glioblastom", "belirtecler": [{"ad": "IDH", "deger": "wild tip"}],
             "onemli_bulgular": ["…"], "ozet": "…", "belirsizlikler": []}, "markdown": "**Kategori:** Brain …", "usage": {"input_tokens": 1200}},
  "classification": {"validated_category": "Brain", "adjusted_confidence": 1.0, "histological_type": "Glioblastom", "reasoning": "..."},
  "anonymized_text": "...",
  "timings": {"ocr": 3.1, "anonymize": 1.8, "gate": 0.2, "report": 9.4}
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

### Geliştirme (hibrit)
```bash
export ANTHROPIC_API_KEY=sk-ant-...      # veya cloud.api_key_file
ANAMNEZ_SERVER__API_KEY=devkey ANAMNEZ_CLOUD__ENABLED=true ANAMNEZ_LLM__ENABLED=false .venv/bin/python -m service
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
