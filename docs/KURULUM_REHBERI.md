# Kurulum Rehberi

İki mod aynı kodla çalışır. **Varsayılan: Hibrit** (GPU gerekmez; kişisel veri sunucuda kalır, yalnızca anonim
metin Claude'a gider). GPU'lu tam kapalı devre için sonda "Mod B".

---

## Mod A — Hibrit: Türkiye'de CPU sunucu + Claude API

### 0. Donanım / yazılım
| | Öneri |
|---|---|
| CPU | 4+ çekirdek (GLiNER + Tesseract) |
| RAM | 8 GB (api konteyneri ~2-3 GB) |
| Disk | 20 GB (imajlar ~2 GB, model ~1 GB, veri) |
| OS | Ubuntu 22.04/24.04, Docker + compose v2, nftables |
| Ağ | İç ağdan 8080; dışarı **yalnızca** squid konteyneri → `api.anthropic.com:443` |

Beklenen hız: sayfa başına OCR 2-5 sn + NER 1-2 sn + Claude 5-15 sn.

### 1. Hazırlık makinesi (internetli) — bundle
```bash
git clone https://github.com/Muratatceken/anamnez-service.git && cd anamnez-service
scripts/prepare_models.sh bundle/          # GLiNER-tr modeli + anamnez-service:1.2.0 + squid imajı + SHA256SUMS
tar czf anamnez-bundle-$(date +%Y%m%d).tgz bundle/ && gpg --detach-sign anamnez-bundle-*.tgz
```
Sunucunun interneti varsa bu adım sunucuda da yapılabilir; yine de tekrar üretilebilirlik için bundle önerilir.

### 2. Sunucu — kur
```bash
gpg --verify anamnez-bundle-*.tgz.sig && tar xzf anamnez-bundle-*.tgz && cd bundle && sha256sum -c SHA256SUMS
sudo mkdir -p /opt/anamnez && sudo cp -r src/* /opt/anamnez/ && sudo cp -r models /opt/anamnez/deploy/models
docker load < images/images.tar.gz
cd /opt/anamnez
openssl rand -hex 32 > deploy/api_key.txt && chmod 600 deploy/api_key.txt          # müşteri API anahtarı
echo "sk-ant-..."     > deploy/anthropic_key.txt && chmod 600 deploy/anthropic_key.txt   # Anthropic anahtarı (Docker secret)
cp deploy/.env.example .env && nano .env    # API_BIND=<iç ağ IP>
```

### 3. Güvenlik duvarı (tek pinhole)
`deploy/nftables.cpu.conf` içinde `LAN`, `MGMT`, `DNS` tanımlarını düzenleyin, sonra:
```bash
sudo nft -f deploy/nftables.cpu.conf && sudo nft list table inet closedloop | grep PINHOLE
```
Kural özeti: host ve `api` konteyneri dışarı **hiçbir** paket atamaz; yalnızca squid (`10.200.2.20`) → 443 ve DNS.
Squid ise yalnızca `CONNECT api.anthropic.com:443` kabul eder (`deploy/squid/squid.conf`).

### 4. Başlat
```bash
docker compose -f deploy/docker-compose.cpu.yml up -d
docker compose -f deploy/docker-compose.cpu.yml logs -f api      # "NER ısınma: {'ok': True ...}" ve "Servis hazır"
curl -s http://<API_BIND>:8080/health | python3 -m json.tool      # ok: true; cloud.enabled: true
```
Açılışta otomatik: `deploy/systemd/anamnez-compose.service` (içindeki compose dosya adını `docker-compose.cpu.yml`,
nft dosyasını `nftables.cpu.conf` yapın), `systemctl enable --now anamnez-compose`.

### 5. Kabul testi
```bash
# a) Egress kanıtı: 5 dk boyunca dış hedefe giden paketler yalnızca 10.200.2.20 → 443 olmalı
sudo EDGE_IF=eth0 LAN=10.0.0.0/24 MGMT=10.0.99.0/24 scripts/verify_no_egress.sh 300
# b) Sentetik uçtan uca (gerçek hasta verisi kullanmayın)
python3 bench/render_handwriting.py --out /tmp/synth --n 3
python3 client/anamnez_client.py --url http://<API_BIND>:8080 --key "$(cat deploy/api_key.txt)" --out /tmp/kabul /tmp/synth/*.png
#    → status done, report.markdown dolu, egress.allowed true, anonymized_text içinde sentetik isim yok
# c) Denetim kaydı: her Claude çağrısı için hash/boyut/model satırı, metin YOK
docker compose -f deploy/docker-compose.cpu.yml exec api python -c "import sqlite3;print(sqlite3.connect('/data/egress_audit.db').execute('select ts,allowed,chars,model from egress_audit order by id desc limit 5').fetchall())"
```

### 6. İşletim
| Konu | Nasıl |
|---|---|
| Anthropic anahtarı döndürme | `deploy/anthropic_key.txt` güncelle → `docker compose ... up -d api` |
| Maliyet | `report.usage` (input/output/cache_read token); sistem prompt'u cache'lenir |
| İnceleme kuyruğu | `GET /jobs?status=needs_review` (maskeli metin + bulgu türleri); insan onayı sonrası dosya yeniden gönderilebilir |
| Saklama süresi | `sqlite3 /data/service.db "DELETE FROM jobs WHERE created_at < date('now','-90 day')"` (cron) |
| Loglar | Ham metin/dosya adı asla yok; `docker compose logs api` |
| Güncelleme | Yeni bundle → `sha256sum -c` → `docker load` → `up -d`. Otomatik güncelleme yok |
| Model değişikliği | `ANAMNEZ_CLOUD__MODEL` (compose env) — yalnızca Anthropic modelleri; sağlayıcı değiştirilemez (kodda kilit) |

### 7. Sorun giderme
| Belirti | Çözüm |
|---|---|
| `/health` → `ner.ok: false` | `deploy/models/neonredact-tr` yok/eksik; bundle'ı kontrol et (gliner_config.json, pytorch_model.bin, tokenizer.json) |
| İş `failed`, error `APIConnectionError` | squid çalışmıyor / nftables pinhole kapalı → `docker compose logs egress-proxy`, `nft list table inet closedloop` |
| İş `failed`, error `AuthenticationError` | `deploy/anthropic_key.txt` yanlış |
| İş `needs_review`, `egress.allowed: false` | Kapı bulgusu — `review_text_mode: full` ile bak; yanlış pozitifse `gate` filtrelerine test ekle |
| Çok yavaş | `OMP_NUM_THREADS` artır; `server.workers` 1 kalsın (CPU'da paralel NER bellek yer) |
| Tesseract el yazısını okuyamıyor | Beklenen sınır (CER ~0.15-0.25). GPU-VLM OCR için Mod B veya yurt içi GPU |

---

## Mod B — Kapalı devre (Linux + NVIDIA GPU, hiçbir veri dışarı çıkmaz)
`MODE=gpu scripts/prepare_models.sh bundle/` (GLM-OCR + Qwen3-8B-AWQ + vLLM imajı), `deploy/docker-compose.yml`,
`deploy/nftables.conf` (outbound tamamen kapalı). Donanım: 1× L4/RTX 4090 (24 GB). `cloud.enabled: false`,
`llm.enabled: true`. Ayrıntılar önceki rehber bölümleriyle aynıdır; egress kanıtında **0 dış paket** beklenir.

---

## Geliştirme ortamı
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-service.txt torch pytest
brew install tesseract tesseract-lang            # Linux: apt install tesseract-ocr tesseract-ocr-tur
make test                                         # ~100 test (GLiNER modeli ilk seferde HF'den iner)
export ANTHROPIC_API_KEY=sk-ant-...
ANAMNEZ_CLOUD__ENABLED=true ANAMNEZ_LLM__ENABLED=false make run
python bench/run_bench.py --ocr tesseract --engine ner+regex --no-classify   # sentetik el yazısı ölçümü
```
Gerçek hasta dosyalarıyla lokal regresyon: `output/originals/*.txt` + `tests/private/known_pii.txt` (ikisi de
`.gitignore`'da) varsa otomatik çalışır.
