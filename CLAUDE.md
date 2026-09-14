# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KVKK Uyumlu Tıbbi Form Anonimizasyon Sistemi. Anamnez ve patoloji formlarından kişisel verileri (PII) temizleyerek API LLM'lere güvenle gönderilebilir hale getirir. Tüm işlem lokal yapılır — hiçbir veri dışarı gönderilmez.

## Commands

```bash
# CLI - tek dosya anonimize
python src/anonymize_cli.py -i rapor.pdf

# CLI - toplu anonimizasyon (dizin)
python src/anonymize_cli.py -d anamnes_files/ -o output/

# CLI - NER yöntemiyle (spaCy gerektirir)
python src/anonymize_cli.py -d anamnes_files/ -o output/ --method ner

# Web UI (Gradio, 127.0.0.1:7860)
python src/app.py

# Testler (anonymizer regresyon + servis + validator; Ollama gerekmez)
.venv/bin/python -m pytest tests/ -q

# Kapalı devre servis (asenkron job API) — bkz. docs/SERVIS.md
/opt/homebrew/bin/ollama serve            # ARM64 Ollama şart (Intel binary Metal kullanamaz)
ANAMNEZ_SERVER__API_KEY=devkey .venv/bin/python -m service
curl -H "X-API-Key: devkey" -H "X-Filename: rapor.pdf" --data-binary @rapor.pdf http://127.0.0.1:8080/jobs

# İstemci
python client/anamnez_client.py --url http://127.0.0.1:8080 --key devkey rapor.pdf
```

## Architecture

**Ana pipeline:** PDF/Görüntü → Metin Çıkarma (lokal OCR) → Anonimizasyon (regex/NER) → Temiz metin çıktısı

**Servis (`service/`, kapalı devre, tek kiracı):** Dosya → çift OCR (Vision-LLM + Tesseract) → regex anonimizasyon → doğrulama kapısı (deterministik + çapraz OCR + LLM yargıç; fail-closed) → lokal LLM sınıflandırma → JSON rapor. Backend'ler pluggable: `ollama` (dev) / `openai` (vLLM, prod). Dağıtım: `deploy/` (docker-compose internal network, nftables egress deny, offline model bundle). Detay: `docs/SERVIS.md`.

Key modules in `src/`:

- **`text_extractor.py`** — Bağımsız metin çıkarma modülü. PyMuPDF ile PDF metin katmanını okur, Tesseract OCR ile görüntülerden metin çıkarır. `extract_from_file()` otomatik format tespiti yapar.
- **`anonymizer.py`** — Regex tabanlı 11 adımlı anonimizasyon pipeline'ı. TC kimlik, hasta/doktor adları, tarihler, kurum adları, iletişim bilgileri, imza blokları temizler. Tıbbi terimleri korur. `ReportAnonymizer.anonymize()` → (text, report).
- **`ner_anonymizer.py`** — NER + regex hibrit anonimizasyon (opsiyonel, spaCy gerektirir). Yapısal alan etiketleri olmadan da isim tespiti yapabilir. 69+ tıbbi terim whitelist'i ile false positive'leri engeller.
- **`anonymize_cli.py`** — CLI arayüzü. Tek dosya veya toplu anonimizasyon, çıktı dosyaları oluşturma, veritabanına loglama.
- **`app.py`** — Gradio web UI. Anonimizasyon (orijinal vs temiz metin yan yana), toplu işleme, istatistikler, sistem durumu.
- **`db.py`** — SQLite loglama. `anonymization_log` ve `classification_log` tabloları.
- **`models.py`** — Pydantic v2 modelleri: `AnonymizationRecord`, `CancerCategory` (33 tip), `ClassificationResult`, `ProcessingRecord`.

- **`validator.py`** — Keyword tabanlı sınıflandırma doğrulama (servis tarafından kullanılır)

Servis modülleri (`service/`): `api.py` (FastAPI, raw-body upload, API key), `jobs.py` (sınırlı kuyruk + SQLite), `pipeline.py`, `backends/llm.py` (Ollama/OpenAI-uyumlu), `backends/ocr.py` (Vision-LLM + Tesseract stdin/stdout), `anonymization/gate.py` (deterministik + çapraz-OCR + LLM yargıç), `classification.py`.

**Kapalı devre kuralı:** Bulut API'ye veri gönderen hiçbir modül yoktur ve eklenmemelidir. Ham metin/dosya adı loglanmaz; `exc_info=True` ve `str(e)` loglama yasak (ham metin taşıyabilir).

## Configuration

- **`config/service.yaml`** — Servis ayarları (OCR/LLM backend, kapı katmanları, API). `ANAMNEZ_<BÖLÜM>__<ANAHTAR>` env ile ezilir.
- **`config/settings.yaml`** — PDF/OCR ayarları, anonimizasyon seçenekleri (yaş genelleştirme, tarih/kurum silme), veritabanı yolu, UI ayarları, opsiyonel LLM ayarları.
- **`config/categories.json`** — 33 kanser kategorisi + Türkçe/İngilizce keyword hints (validator için).
- **`templates/`** — LLM prompt şablonları (sınıflandırma için).

## Key Design Decisions

- **OCR-first:** PDF'lerin çoğu taranmış görüntü, metin katmanı yok. Tesseract OCR ile metin çıkarma zorunlu. `extract_text_first: true` ayarı ile önce metin katmanı denenir.
- **Regex > NER varsayılan:** Regex yöntemi bağımlılık gerektirmez ve hızlı. NER (spaCy) opsiyonel ama yapısal alan etiketleri olmadan da isim tespiti yapabilir.
- **Tıbbi terim koruması:** Hem regex hem NER anonimizasyonda kapsamlı whitelist var (kanser tipleri, organlar, markerlar, patoloji terimleri). False positive'leri engeller.
- **KVKK uyumu:** Tam metin loglama kapalı, tüm işlem lokal, public Gradio link kapalı. Gerçek hasta verisi (`anamnes_files/`, `output/`, `tests/private/`) `.gitignore`'dadır; testlere/yorumlara/sözlüğe gerçek isim yazılmaz (sentetik isim kullan).
- **Sınıflandırma opsiyonel:** Ana odak anonimizasyon. Sınıflandırma modülleri hala mevcut ama Ollama veya API key gerektirir.

## System Requirements

- Python 3.9+
- Tesseract OCR (`brew install tesseract tesseract-lang` on macOS)
- Opsiyonel: spaCy + model (`pip install spacy && python -m spacy download xx_ent_wiki_sm`)
