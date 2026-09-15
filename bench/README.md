# bench/ — el yazısı OCR & anonimizasyon benchmark'ı

Gerçek hasta verisi olmadan, **sentetik el yazısı formlarıyla** sistemi ölçer.

| Dosya | Amaç |
|---|---|
| `synth_cases.py` | 20 sentetik vaka (anamnez/patoloji, 20 kanser kategorisi, uydurma PII, korunacak tıbbi terimler) |
| `render_handwriting.py` | Vakaları 9 el yazısı fontu + form şablonu + tarama/fotoğraf bozulmalarıyla PNG'ye çevirir (ground truth ile) |
| `run_bench.py` | OCR backend'leri (tesseract / ollama / vLLM) × görüntüler → CER, PII okuma, sızıntı, aşırı silme, kapı, sınıflandırma |
| `colab_bench.py` | Ücretsiz Colab T4'te her şeyi tek komutla koşar (vLLM: GLM-OCR, PaddleOCR-VL, Qwen3-8B-AWQ) |
| `anamnez_bench.ipynb` | Colab notebook'u (Runtime → T4 → hücreyi çalıştır) |
| `fonts/` | Google Fonts (OFL), Türkçe glif destekli el yazısı fontları |

Kritik metrik: **`kapı PASS + sızıntı`** = kapıdan geçtiği halde PII içeren görüntü sayısı → **0 olmalı**.
Diğerleri kalite: CER (düşük iyi), PII okuma (OCR PII'yi görüyor mu — görmezse maskeleyemez ama sızdırmaz da),
aşırı silme (tıbbi terim kaybı), sınıflandırma doğruluğu.

```bash
# Lokal (Mac, Ollama):
python bench/render_handwriting.py --out bench/data --n 20
python bench/run_bench.py --ocr tesseract --ocr ollama:glm-ocr --llm ollama:qwen3:8b --out bench/results/local
# Colab: bench/anamnez_bench.ipynb
```
