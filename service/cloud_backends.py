"""Bulut LLM sağlayıcıları — tek arayüz: complete(system, user, schema) → CloudResponse.

İzinli sağlayıcılar (service/egress.py ALLOWED_PROVIDERS ile aynı): anthropic, gemini.
Her ikisi de yalnızca kapıdan/egress'ten geçmiş ANONİM metni alır; base_url sabittir, proxy zorunlu kılınabilir.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HOSTS = {"anthropic": "api.anthropic.com", "gemini": "generativelanguage.googleapis.com"}


class CloudError(Exception):
    """Kalıcı bulut hatası (kimlik, istek biçimi, şema)."""


class TransientCloudError(CloudError):
    """Yeniden denenebilir: 429, 5xx, bağlantı, zaman aşımı."""


@dataclass
class CloudResponse:
    text: str
    stop_reason: str            # end_turn | max_tokens | refusal | other
    refusal_category: Optional[str] = None
    usage: dict = field(default_factory=dict)
    request_id: Optional[str] = None


def _read_key(cfg: dict, env_name: str) -> Optional[str]:
    import os

    key_file = cfg.get("api_key_file")
    if key_file and Path(key_file).exists():
        return Path(key_file).read_text(encoding="utf-8").strip()
    return os.environ.get(env_name) or None


def inline_refs(schema: dict) -> dict:
    """pydantic'in $defs/$ref şemasını düzleştir (Gemini responseJsonSchema $ref desteklemeyebilir)."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"].split("/")[-1]
                return walk(copy.deepcopy(defs[ref]))
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


# ── Anthropic ───────────────────────────────────────────────────────────
class AnthropicBackend:
    name = "anthropic"

    def __init__(self, cfg: dict):
        self.model = cfg.get("model", "claude-opus-5")
        self.effort = cfg.get("effort", "high")
        self.max_tokens = int(cfg.get("max_tokens", 16000))
        self.timeout = float(cfg.get("timeout", 300))
        self.proxy = cfg.get("proxy")
        self._cfg = cfg
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            kwargs = {"api_key": _read_key(self._cfg, "ANTHROPIC_API_KEY"), "timeout": self.timeout,
                      "max_retries": 0, "base_url": f"https://{HOSTS['anthropic']}"}
            if self.proxy:
                kwargs["http_client"] = anthropic.DefaultHttpxClient(proxy=self.proxy)
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(self, system: str, user: str, schema: dict) -> CloudResponse:
        import anthropic

        client = self._get_client()
        try:
            r = client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError as e:
            raise TransientCloudError("RateLimitError") from e
        except anthropic.APITimeoutError as e:
            raise TransientCloudError("APITimeoutError") from e
        except anthropic.APIConnectionError as e:
            raise TransientCloudError("APIConnectionError") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 or e.status_code in (408, 409, 529):
                raise TransientCloudError(f"APIStatusError{e.status_code}") from e
            raise CloudError(type(e).__name__) from e
        text = next((b.text for b in r.content if getattr(b, "type", "") == "text"), "")
        usage = getattr(r, "usage", None)
        stop = {"end_turn": "end_turn", "max_tokens": "max_tokens", "refusal": "refusal"}.get(r.stop_reason, "other")
        details = getattr(r, "stop_details", None)
        return CloudResponse(
            text=text, stop_reason=stop, refusal_category=getattr(details, "category", None) if stop == "refusal" else None,
            usage={"input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None),
                   "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None)},
            request_id=getattr(r, "_request_id", None),
        )


# ── Gemini (REST, httpx) ────────────────────────────────────────────────
class GeminiBackend:
    name = "gemini"
    _EFFORT_TO_LEVEL = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}

    def __init__(self, cfg: dict):
        import httpx

        self.model = cfg.get("model", "gemini-3.8-flash")
        self.effort = cfg.get("effort", "high")
        self.max_tokens = int(cfg.get("max_tokens", 16000))
        self.timeout = float(cfg.get("timeout", 300))
        self.api_key = _read_key(cfg, "GEMINI_API_KEY")
        self._client = httpx.Client(
            base_url=f"https://{HOSTS['gemini']}/v1beta", timeout=self.timeout, trust_env=False,
            proxy=cfg.get("proxy") or None,
            headers={"x-goog-api-key": self.api_key or "", "Content-Type": "application/json"},
        )
        self._thinking_ok = True

    def _post(self, body: dict):
        import httpx

        try:
            r = self._client.post(f"/models/{self.model}:generateContent", json=body)
        except httpx.TimeoutException as e:
            raise TransientCloudError("Timeout") from e
        except httpx.HTTPError as e:
            raise TransientCloudError("ConnectionError") from e
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientCloudError(f"HTTP{r.status_code}")
        if r.status_code >= 400:
            # mesaj gövdesi loglanmaz (istek içeriğini yansıtabilir)
            raise CloudError(f"HTTP{r.status_code}")
        return r.json()

    def complete(self, system: str, user: str, schema: dict) -> CloudResponse:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": inline_refs(schema),
                "maxOutputTokens": self.max_tokens,
                "temperature": 0.0,
            },
        }
        if self._thinking_ok and self.model.startswith("gemini-3"):
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self._EFFORT_TO_LEVEL.get(self.effort, "high")}
        try:
            data = self._post(body)
        except CloudError as e:
            # thinkingConfig desteklenmeyen model → onsuz tekrar (bir kez)
            if "HTTP400" in str(e) and "thinkingConfig" in body["generationConfig"]:
                self._thinking_ok = False
                body["generationConfig"].pop("thinkingConfig", None)
                data = self._post(body)
            else:
                raise
        pf = data.get("promptFeedback", {})
        if pf.get("blockReason"):
            return CloudResponse(text="", stop_reason="refusal", refusal_category=pf["blockReason"], usage={})
        cands = data.get("candidates") or []
        if not cands:
            raise CloudError("no_candidates")
        c = cands[0]
        fr = c.get("finishReason", "STOP")
        stop = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens"}.get(fr)
        if stop is None:
            stop = "refusal" if fr in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII") else "other"
        parts = c.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        um = data.get("usageMetadata", {})
        return CloudResponse(
            text=text, stop_reason=stop, refusal_category=fr if stop == "refusal" else None,
            usage={"input_tokens": um.get("promptTokenCount"), "output_tokens": um.get("candidatesTokenCount"),
                   "thinking_tokens": um.get("thoughtsTokenCount"), "cache_read_input_tokens": um.get("cachedContentTokenCount")},
            request_id=data.get("responseId"),
        )


def make_backend(cfg: dict):
    provider = cfg.get("provider", "anthropic")
    if provider == "anthropic":
        return AnthropicBackend(cfg)
    if provider == "gemini":
        return GeminiBackend(cfg)
    raise ValueError(f"izin verilmeyen sağlayıcı: {provider}")
