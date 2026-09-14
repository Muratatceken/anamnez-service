"""LLM backend soyutlaması — Ollama (geliştirme) ve OpenAI-uyumlu (vLLM, üretim).

Tek arayüz: `chat(messages, json_mode, images)` → str
Her iki backend de yalnızca konfigürasyondaki base_url'e konuşur; kapalı devrede
bu adres iç ağdaki bir konteynerdir.
"""

import base64
import json
import logging
import re
from typing import Optional, Protocol

import httpx

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        images: Optional[list[bytes]] = None,
        max_tokens: Optional[int] = None,
    ) -> str: ...

    def health(self) -> dict: ...


def extract_json(text: str) -> dict:
    """LLM yanıtından ilk JSON nesnesini çıkar (```json ... ``` ve <think> toleranslı)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = m.group(0) if m else None
    if candidate is None:
        raise ValueError(f"Yanıtta JSON bulunamadı (uzunluk={len(text)})")  # ham yanıt mesaja GİRMEZ
    return json.loads(candidate)


class OllamaLLM:
    name = "ollama"

    def __init__(self, cfg: dict):
        self.base_url = cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")
        self.model = cfg["model"]
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 2048))
        self.timeout = float(cfg.get("timeout", 300))
        self.think = cfg.get("think", None)
        self.num_ctx = int(cfg.get("num_ctx", 0)) or None   # Ollama varsayılanı 4096; görüntü için artır
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, trust_env=False)

    def chat(self, messages, *, json_mode=False, images=None, max_tokens=None) -> str:
        msgs = [dict(m) for m in messages]
        if images:
            msgs[-1]["images"] = [base64.b64encode(b).decode() for b in images]
        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }
        if self.num_ctx:
            payload["options"]["num_ctx"] = self.num_ctx
        if json_mode:
            payload["format"] = "json"
        if self.think is not None:
            payload["think"] = bool(self.think)
        r = self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def health(self) -> dict:
        try:
            r = self._client.get("/api/tags", timeout=5.0)  # health uzun LLM timeout'una bağlı kalmasın
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            base = self.model.split(":")[0]
            return {
                "ok": any(m == self.model or m.split(":")[0] == base for m in models),
                "backend": self.name,
                "model": self.model,
                "available": models,
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "backend": self.name, "model": self.model, "error": type(e).__name__}


class OpenAICompatLLM:
    """vLLM / SGLang / herhangi bir OpenAI-uyumlu /v1 uç noktası."""

    name = "openai"

    def __init__(self, cfg: dict):
        self.base_url = cfg.get("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = cfg["model"]
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 2048))
        self.timeout = float(cfg.get("timeout", 300))
        headers = {}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, headers=headers, trust_env=False)

    def chat(self, messages, *, json_mode=False, images=None, max_tokens=None) -> str:
        msgs = [dict(m) for m in messages]
        if images:
            parts = [{"type": "text", "text": msgs[-1]["content"]}]
            for b in images:
                b64 = base64.b64encode(b).decode()
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            msgs[-1]["content"] = parts
        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def health(self) -> dict:
        try:
            r = self._client.get("/models", timeout=5.0)
            r.raise_for_status()
            models = [m["id"] for m in r.json().get("data", [])]
            return {"ok": self.model in models or not models, "backend": self.name, "model": self.model, "available": models}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "backend": self.name, "model": self.model, "error": type(e).__name__}


def make_llm(cfg: dict) -> LLMBackend:
    backend = cfg.get("backend", "ollama")
    if backend == "ollama":
        return OllamaLLM(cfg)
    if backend == "openai":
        return OpenAICompatLLM(cfg)
    raise ValueError(f"Bilinmeyen LLM backend: {backend}")
