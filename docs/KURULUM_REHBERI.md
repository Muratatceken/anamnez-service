# Kurulum Rehberi — Kapalı Devre Üretim

Bu rehber, servisi müşteri şirketin **internetsiz** sunucusuna kurar. İki makine gerekir:

| Makine | İnternet | Görev |
|---|---|---|
| **Hazırlık** (herhangi bir Linux/Mac, tercihen NVIDIA GPU'lu) | Var | Modelleri ve Docker imajlarını indirip **bundle** üretmek |
| **Sunucu** (Linux + NVIDIA GPU) | **Yok** (yalnızca iç ağ) | Servisi çalıştırmak |

---

## 0. Donanım

| Kurulum | GPU | RAM | Disk | Not |
|---|---|---|---|---|
| Pilot | 1× RTX 4090 / L4 (24 GB) | 64 GB | 500 GB NVMe | GLM-OCR (~3 GB VRAM) + Qwen3-8B-AWQ (~6 GB) |
| Daha güçlü yargıç | 1× L40S / RTX 6000 Ada (48 GB) | 64 GB | 500 GB | + Qwen3-32B-AWQ (~20 GB) → daha az yanlış pozitif |

İşletim sistemi: Ubuntu 22.04/24.04 LTS. Beklenen hız: iş başına 10–30 sn (L4), M1 Mac'te 3–5 dk.

---

## 1. Hazırlık makinesi — bundle üretimi

```bash
# Gereksinimler: python3, docker, git (GPU varsa nvidia-container-toolkit → smoke test çalışır)
git clone <bu repo> anamnez && cd anamnez

# Varsayılan: GLM-OCR + Qwen3-8B-AWQ, vLLM v0.19.1
scripts/prepare_models.sh bundle/

# 48 GB+ GPU'lu sunucu için daha güçlü yargıç:
# LLM_REPO=Qwen/Qwen3-32B-AWQ scripts/prepare_models.sh bundle/
```

Çıktı (`bundle/`):
```
models/glm-ocr/            ~2 GB   OCR modeli (MIT)
models/qwen3-8b-awq/       ~6 GB   yargıç + sınıflandırıcı (Apache-2.0)
images/images.tar.gz       ~12 GB  vllm/vllm-openai:v0.19.1 + anamnez-service:1.1.0
src/{deploy,scripts,config,docs}/
SHA256SUMS
```

Bundle'ı **imzalayıp** taşıyın (USB / iç ağ SFTP):
```bash
tar czf anamnez-bundle-$(date +%Y%m%d).tgz bundle/
gpg --detach-sign anamnez-bundle-*.tgz        # kurum anahtarıyla
```

> GPU'lu hazırlık makinesinde script otomatik **smoke test** yapar (vLLM ağırlıkları yükleyebiliyor mu?).
> GPU yoksa bu adım atlanır; sunucuda ilk açılışta `docker compose logs -f vllm-ocr` izleyin.

---

## 2. Sunucu — işletim sistemi

```bash
# Ubuntu 24.04 örneği
sudo apt-get install -y nftables docker.io docker-compose-v2
# NVIDIA sürücü + container toolkit (offline .deb paketlerini bundle'a ekleyebilirsiniz)
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
nvidia-smi                                      # GPU görünmeli
docker run --rm --gpus all ubuntu nvidia-smi    # konteynerden GPU görünmeli (imaj bundle'da yoksa atlayın)
```

Disk şifreleme (LUKS) kurulumda seçilmiş olmalı; `/opt/anamnez` ve Docker veri dizini şifreli diskte olsun.

---

## 3. Sunucu — bundle'ı aç

```bash
gpg --verify anamnez-bundle-*.tgz.sig anamnez-bundle-*.tgz
tar xzf anamnez-bundle-*.tgz && cd bundle
sha256sum -c SHA256SUMS                          # hepsi OK olmalı

sudo mkdir -p /opt/anamnez && sudo cp -r src/* /opt/anamnez/
sudo cp -r models /opt/anamnez/deploy/models     # compose ./models'ı buradan mount eder
docker load < images/images.tar.gz
docker images | grep -E "vllm-openai|anamnez-service"
```

---

## 4. Sunucu — yapılandırma

```bash
cd /opt/anamnez
openssl rand -hex 32 > deploy/api_key.txt && chmod 600 deploy/api_key.txt   # Docker secret
cp deploy/.env.example .env && nano .env
#   API_BIND=10.0.0.5     ← host'un müşteri iç ağına bakan IP'si
#   VLLM_TAG=v0.19.1      ← bundle'daki etiket
```

`deploy/nftables.conf` içinde ağları düzenleyin:
```
define LAN   = 10.0.0.0/24     # API'yi çağıracak istemcilerin ağı
define MGMT  = 10.0.99.0/24    # SSH yönetim ağı
```

`config/service.yaml` — üretimde compose zaten env ile eziyor (`ANAMNEZ_*`). Değiştirmek isteyebilecekleriniz:
| Anahtar | Varsayılan | Açıklama |
|---|---|---|
| `server.max_queue` | 20 | Bekleyen iş sınırı (dolunca 503) |
| `storage.review_text_mode` | `masked` | `needs_review` sonucunda metin: masked / full / none |
| `anonymization.gate.fail_closed` | true | Katman hatasında da durdur (ÖNERİLİR) |
| `ocr.min_completeness` | 0.6 | VLM/Tesseract uzunluk oranı altında uyarı |
| `classification.confidence_threshold` | 0.6 | Altında uyarı |

---

## 5. Sunucu — güvenlik duvarı (egress kapalı)

```bash
sudo nft -f deploy/nftables.conf
sudo nft list table inet closedloop | head        # tablo görünmeli
# Kalıcı: systemd birimi Docker'dan SONRA uygular (bkz. adım 6)
```

Kural özeti: dışarı **hiçbir** paket çıkmaz (DNS dahil); içeri yalnızca LAN→8080 ve MGMT→22; Docker
köprüleri arası trafik ve LAN→API forward'ı açık. Docker'ın kendi nft tabloları korunur.

---

## 6. Sunucu — servisi başlat

```bash
cd /opt/anamnez
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml logs -f vllm-ocr vllm-llm   # "Application startup complete" bekleyin (1-3 dk)
curl -s http://10.0.0.5:8080/health | python3 -m json.tool               # "ok": true
```

Açılışta otomatik başlatma:
```bash
sudo cp deploy/systemd/anamnez-compose.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now anamnez-compose
```

---

## 7. Kabul testi (müşteriye teslim)

```bash
# 1) Egress kanıtı — 5 dakika boyunca dış hedefe 0 paket
sudo EDGE_IF=eth0 LAN=10.0.0.0/24 MGMT=10.0.99.0/24 scripts/verify_no_egress.sh 300

# 2) Uçtan uca (sentetik/izinli test dosyasıyla)
python3 client/anamnez_client.py --url http://10.0.0.5:8080 --key "$(cat deploy/api_key.txt)" \
        --ref KABUL-001 --out /tmp/kabul test_rapor.pdf
#    → status done, kategori, kapı PASS; /tmp/kabul/<job>.json içinde anonim metin

# 3) Negatif test: kapı çalışıyor mu?
#    İçine bilinçli olarak "Dr. Ayşe Kaya" yazılmış bir .txt gönderin → needs_review beklenir (regex yakalarsa done)
```

Teslim raporuna ekleyin: `verify_no_egress` çıktısı, `SHA256SUMS`, kullanılan model/imaj etiketleri, `/health` çıktısı.

---

## 8. İşletim

| Konu | Nasıl |
|---|---|
| Loglar | `docker compose logs api` — ham metin/dosya adı **loglanmaz** |
| İnceleme kuyruğu | `GET /jobs?status=needs_review` → inceleyici anonim (maskeli) metni ve bulgu türlerini görür |
| Veri saklama | `service-data` volume'ünde SQLite (metadata + anonim sonuç). Saklama süresi politikası için `sqlite3 service.db "DELETE FROM jobs WHERE created_at < date('now','-90 day')"` cron'a alınabilir |
| Yedek | Yalnızca `deploy/api_key.txt`, `.env`, `service-data` volume (anonim veri) |
| Güncelleme | Yeni bundle → `sha256sum -c` → `docker load` → `docker compose up -d`. Otomatik güncelleme **yok** |
| Model değişikliği | `deploy/models/` altına yeni klasör + compose `--model` yolu; ilk açılışta `logs -f` |

---

## 9. Sorun giderme

| Belirti | Neden / Çözüm |
|---|---|
| `/health` 503, `llm.ok=false` | vLLM daha yüklenmedi (start_period 120 s) veya VRAM yetmedi → `logs vllm-llm`; `--gpu-memory-utilization` düşürün |
| `vllm-ocr` "unrecognized model" | vLLM etiketi GLM-OCR için eski (≥0.12 gerek) → `VLLM_TAG` |
| İşler `failed`, error `ReadTimeout` | LLM yanıtı `llm.timeout` içinde gelmedi → GPU yavaş/paylaşımlı; timeout artırın |
| Hep `needs_review` | Yargıç yanlış pozitif → `review_text_mode: full` ile bulguya bakın; 32B model; regex'e yeni etiket ekleyin ve **test yazın** |
| `ocr.completeness` < 0.6 uyarısı | VLM sayfa/sütun atladı → `ocr.dpi`/`max_dimension`; alternatif OCR modeli |
| Port publish çalışmıyor | `api` hem `internal` hem `edge` ağında olmalı (compose'da öyle); `API_BIND` doğru IP mi? |
| Konteynerler birbirini görmüyor | nftables forward zinciri: `br-*` ↔ `br-*` accept var mı? `nft list table inet closedloop` |

---

## 10. Geliştirme ortamı (Mac M1 / Linux, Ollama)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-service.txt pytest
brew install tesseract tesseract-lang                     # Linux: apt install tesseract-ocr tesseract-ocr-tur
/opt/homebrew/bin/ollama serve &                          # M1: ARM64 Ollama (Intel binary Metal kullanamaz!)
ollama pull glm-ocr && ollama pull qwen3:8b
make test                                                  # 69 test
make run                                                   # config/service.yaml: backend ollama
```
Gerçek hasta dosyalarıyla regresyon: `output/originals/*.txt` + `tests/private/known_pii.txt` (ikisi de
`.gitignore`'da) mevcutsa `test_real_files_no_known_pii` otomatik çalışır; yoksa atlanır.
