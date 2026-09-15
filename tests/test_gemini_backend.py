"""GeminiBackend — ağ yok; httpx transport taklidi."""

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from service.cloud_backends import CloudError, GeminiBackend, TransientCloudError, inline_refs, make_backend
from service.report import DoctorReport, ReportGenerator


def _backend(handler):
    b = GeminiBackend({"model": "gemini-3.8-flash", "effort": "high", "api_key_file": "/nonexistent"})
    b._client = httpx.Client(base_url="https://generativelanguage.googleapis.com/v1beta", transport=httpx.MockTransport(handler))
    return b


def _ok_body(text, finish="STOP"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "thoughtsTokenCount": 30}, "responseId": "r1"}


def test_inline_refs_flattens_pydantic_schema():
    sch = inline_refs(DoctorReport.model_json_schema())
    assert "$defs" not in json.dumps(sch) and "$ref" not in json.dumps(sch)
    assert sch["properties"]["belirtecler"]["items"]["properties"]["ad"]["type"] == "string"


def test_gemini_request_shape_and_success():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url); seen["body"] = json.loads(req.content); seen["key"] = req.headers.get("x-goog-api-key")
        return httpx.Response(200, json=_ok_body('{"a": 1}'))

    r = _backend(handler).complete("SYS", "USER", {"type": "object", "properties": {"a": {"type": "integer"}}})
    assert r.stop_reason == "end_turn" and r.text == '{"a": 1}' and r.usage["thinking_tokens"] == 30
    assert seen["url"].endswith("/models/gemini-3.8-flash:generateContent")
    gc = seen["body"]["generationConfig"]
    assert gc["responseMimeType"] == "application/json" and gc["thinkingConfig"] == {"thinkingLevel": "high"}
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"


@pytest.mark.parametrize("finish,expected", [("MAX_TOKENS", "max_tokens"), ("SAFETY", "refusal"), ("RECITATION", "refusal")])
def test_gemini_finish_reason_mapping(finish, expected):
    r = _backend(lambda req: httpx.Response(200, json=_ok_body("", finish))).complete("s", "u", {})
    assert r.stop_reason == expected


def test_gemini_prompt_block_is_refusal():
    body = {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}
    r = _backend(lambda req: httpx.Response(200, json=body)).complete("s", "u", {})
    assert r.stop_reason == "refusal" and r.refusal_category == "PROHIBITED_CONTENT"


@pytest.mark.parametrize("status,exc", [(429, TransientCloudError), (503, TransientCloudError), (400, CloudError), (403, CloudError)])
def test_gemini_http_errors(status, exc):
    with pytest.raises(exc):
        _backend(lambda req: httpx.Response(status, json={"error": {"message": "GİZLİ"}})).complete("s", "u", {})


def test_gemini_thinking_config_fallback_on_400():
    calls = []

    def handler(req):
        body = json.loads(req.content); calls.append("thinkingConfig" in body["generationConfig"])
        if "thinkingConfig" in body["generationConfig"]:
            return httpx.Response(400, json={"error": {"message": "thinkingConfig unsupported"}})
        return httpx.Response(200, json=_ok_body('{"a": 1}'))

    b = _backend(handler)
    assert b.complete("s", "u", {}).text == '{"a": 1}' and calls == [True, False]
    assert b.complete("s", "u", {}).text == '{"a": 1}' and calls == [True, False, False]   # bir daha denemez


def test_report_generator_with_gemini_backend_end_to_end():
    from tests.test_cloud import sample_report

    class FakeGemini:
        def complete(self, system, user, schema):
            from service.cloud_backends import CloudResponse
            assert "Kategori listesi" in system and "RAPOR METNİ" in user
            return CloudResponse(text=sample_report().model_dump_json(), stop_reason="end_turn", usage={"input_tokens": 1})

    gen = ReportGenerator({"provider": "gemini", "model": "gemini-pro-latest"}, backend=FakeGemini())
    r = gen.generate("TANI\nSol akciğer, skuamöz hücreli karsinom. TTF-1 negatif.")
    assert r["validated_category"] == "Lung" and r["provider"] == "gemini"


def test_make_backend_rejects_unknown_provider():
    with pytest.raises(ValueError):
        make_backend({"provider": "openai"})
