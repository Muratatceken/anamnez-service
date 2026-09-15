"""Servis konfigürasyonu — YAML + ortam değişkeni ezme.

Kapalı devre ilkesi: konfigürasyon yüklenirken tüm bilinen telemetri/çevrimiçi
davranışlar ortam değişkenleriyle kapatılır (savunma derinliği; asıl güvence
ağ seviyesindeki egress engelidir).
"""

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "service.yaml"
ENV_PREFIX = "ANAMNEZ_"

# Kütüphanelerin dışarı konuşmasını engelleyen ortam değişkenleri
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "OLLAMA_NOPRUNE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


# Bulut istemcisinin hedefini/loglamasını değiştirebilecek env'ler — servis bunları YOK SAYAR
UNSAFE_ENV = ("ANTHROPIC_BASE_URL", "ANTHROPIC_LOG", "ANTHROPIC_CUSTOM_HEADERS",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")


def enforce_offline_env() -> None:
    for k, v in OFFLINE_ENV.items():
        os.environ.setdefault(k, v)
    for k in UNSAFE_ENV:
        if k in os.environ:
            os.environ.pop(k, None)


_BOOL_WORDS = {"true": True, "false": False, "yes": True, "no": False, "on": True, "off": False, "1": True, "0": False}


def _coerce(value: str, like: Any) -> Any:
    """Ortam değişkeni string'ini mevcut YAML değerinin tipine çevir (YAML'da yoksa bool kelimeleri yine bool)."""
    if isinstance(like, bool):
        if value.lower() not in _BOOL_WORDS:
            raise ValueError(f"boolean bekleniyor: {value!r}")
        return _BOOL_WORDS[value.lower()]
    if like is None and value.lower() in _BOOL_WORDS:
        return _BOOL_WORDS[value.lower()]
    if isinstance(like, int):
        return int(value)
    if isinstance(like, float):
        return float(value)
    return value


def _apply_env_overrides(cfg: dict, prefix: str = ENV_PREFIX) -> dict:
    """ANAMNEZ_LLM__MODEL=... → cfg['llm']['model']"""
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        node = cfg
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        leaf = path[-1]
        node[leaf] = _coerce(value, node.get(leaf))
    return cfg


def load_config(path: Path | str | None = None) -> dict:
    enforce_offline_env()
    cfg_path = Path(path or os.environ.get(f"{ENV_PREFIX}CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _apply_env_overrides(cfg)
    # Docker secret: api_key dosyadan (env'de anahtar görünmez)
    key_file = cfg.get("server", {}).get("api_key_file")
    if key_file and Path(key_file).exists():
        cfg["server"]["api_key"] = Path(key_file).read_text(encoding="utf-8").strip()
    return cfg
