"""Colab / Kaggle (ücretsiz T4) üzerinde uçtan uca benchmark.

Notebook'ta tek hücre:
    !git clone https://github.com/Muratatceken/anamnez-service.git && cd anamnez-service && python bench/colab_bench.py

Ne yapar:
  1. Sistem paketleri (tesseract tur) + vLLM kurar
  2. Modelleri indirir: GLM-OCR, PaddleOCR-VL-1.6, Qwen3-8B-AWQ  (hepsi T4 16 GB'a sığar; sırayla yüklenir)
  3. Sentetik el yazısı setini üretir (bench/render_handwriting.py) — GERÇEK HASTA VERİSİ YOK
  4. Her OCR modeli için vLLM sunucusunu başlatır → benchmark → kapatır
  5. Yargıç + sınıflandırma için Qwen3-8B-AWQ ile son turu koşar
  6. bench/results/colab/summary.md yazar ve ekrana basar

Ortam değişkenleri: N_CASES (20), FONTS_PER_CASE (1), STYLES ("scan,phone,bad"), SKIP_PADDLE (0/1), VLLM_VERSION
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
N_CASES = int(os.environ.get("N_CASES", "20"))
FONTS_PER_CASE = int(os.environ.get("FONTS_PER_CASE", "1"))
STYLES = os.environ.get("STYLES", "scan,phone,bad")
SKIP_PADDLE = os.environ.get("SKIP_PADDLE", "0") == "1"
VLLM_VERSION = os.environ.get("VLLM_VERSION", "")   # boş = en son stabil
MODELS = ROOT / "models"
LOGS = ROOT / "bench" / "results" / "colab"
LOGS.mkdir(parents=True, exist_ok=True)


def sh(cmd: str, check: bool = True, quiet: bool = False) -> int:
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.STDOUT if quiet else None)
    if check and r.returncode != 0:
        raise SystemExit(f"komut başarısız ({r.returncode}): {cmd}")
    return r.returncode


def wait_http(url: str, timeout: float = 900) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if urllib.request.urlopen(url, timeout=3).status == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def start_vllm(model_dir: Path, served: str, port: int, extra: str = "") -> subprocess.Popen:
    log = open(LOGS / f"vllm_{served}.log", "w")
    cmd = (f"python -m vllm.entrypoints.openai.api_server --model {model_dir} --served-model-name {served} "
           f"--port {port} --dtype half --gpu-memory-utilization 0.85 --max-model-len 8192 "
           f"--enforce-eager --disable-log-requests {extra}")
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    ok = wait_http(f"http://127.0.0.1:{port}/health")
    if not ok:
        p.kill()
        print(open(LOGS / f"vllm_{served}.log").read()[-3000:])
        raise SystemExit(f"vLLM {served} başlamadı")
    print(f"vLLM {served} hazır (port {port})", flush=True)
    return p


def stop(p: subprocess.Popen) -> None:
    p.terminate()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    time.sleep(5)


def main() -> None:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    # 1) paketler
    sh("apt-get -qq update && apt-get -qq install -y tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng > /dev/null", check=False)
    sh(f"pip install -q 'vllm{('==' + VLLM_VERSION) if VLLM_VERSION else ''}' 'huggingface_hub[cli]' pymupdf pillow pyyaml pydantic httpx", quiet=False)

    # 2) modeller
    hf = "hf" if subprocess.run("command -v hf", shell=True, capture_output=True).returncode == 0 else "huggingface-cli"
    sh(f"{hf} download zai-org/GLM-OCR --local-dir {MODELS}/glm-ocr", quiet=True)
    if not SKIP_PADDLE:
        sh(f"{hf} download PaddlePaddle/PaddleOCR-VL-1.6 --local-dir {MODELS}/paddleocr-vl", quiet=True, check=False)
    sh(f"{hf} download Qwen/Qwen3-8B-AWQ --local-dir {MODELS}/qwen3-8b-awq", quiet=True)

    # 3) sentetik veri
    sh(f"python bench/render_handwriting.py --out bench/data --n {N_CASES} --fonts-per-case {FONTS_PER_CASE} --styles {STYLES}")

    # 4) OCR turları (her model ayrı vLLM süreci; T4'e tek tek sığar)
    ocr_specs = []
    p = start_vllm(MODELS / "glm-ocr", "glm-ocr", 8000, "--limit-mm-per-prompt '{\"image\": 1}'")
    sh("python bench/run_bench.py --ocr tesseract --ocr openai:http://127.0.0.1:8000/v1:glm-ocr --no-classify "
       "--out bench/results/colab/ocr_glm --timeout 300")
    stop(p)
    if not SKIP_PADDLE and (MODELS / "paddleocr-vl").exists():
        try:
            p = start_vllm(MODELS / "paddleocr-vl", "paddleocr-vl", 8000, "--trust-remote-code --limit-mm-per-prompt '{\"image\": 1}'")
            sh("python bench/run_bench.py --ocr openai:http://127.0.0.1:8000/v1:paddleocr-vl --no-classify "
               "--out bench/results/colab/ocr_paddle --timeout 300", check=False)
            stop(p)
        except SystemExit as e:
            print("PaddleOCR-VL atlandı:", e)

    # 5) yargıç + sınıflandırma turu: GLM-OCR çıktıları üzerinde Qwen3-8B-AWQ
    #    (OCR'ı tekrar koşmamak için: GLM-OCR + Qwen3 aynı anda T4'e sığmaz → OCR'ı tesseract+glm ile yeniden;
    #     GLM-OCR 0.9B + Qwen3-8B-AWQ ~6 GB → 16 GB'a sığar, iki sunucu)
    p1 = start_vllm(MODELS / "glm-ocr", "glm-ocr", 8000, "--gpu-memory-utilization 0.25 --limit-mm-per-prompt '{\"image\": 1}'")
    p2 = start_vllm(MODELS / "qwen3-8b-awq", "qwen3-8b", 8001, "--quantization awq --gpu-memory-utilization 0.55 --max-model-len 8192")
    sh("python bench/run_bench.py --ocr openai:http://127.0.0.1:8000/v1:glm-ocr "
       "--llm openai:http://127.0.0.1:8001/v1:qwen3-8b --out bench/results/colab/full --timeout 300")
    stop(p1); stop(p2)

    print("\n\n================ ÖZET ================")
    for d in ("ocr_glm", "ocr_paddle", "full"):
        f = LOGS / d / "summary.md"
        if f.exists():
            print(f"\n## {d}\n" + f.read_text())


if __name__ == "__main__":
    main()
