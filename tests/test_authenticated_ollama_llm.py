# tests/test_authenticated_ollama_llm.py
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Hilfen: Dummy HTTP-Client und Response-Fabriken
# ──────────────────────────────────────────────────────────────────────────────

class DummyResponse:
    """
    Minimaler Ersatz für httpx.Response, der genau das kann,
    was der LLM-Code benutzt: status_code, json(), text, raise_for_status().
    """
    def __init__(self, url: str, status_code: int, payload: Dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or (json.dumps(self._payload) if self._payload else "")
        self._url = url

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self):
        if 400 <= self.status_code:
            # Baue echte httpx.HTTPStatusError mit httpx.Response/httpx.Request
            request = httpx.Request("POST", self._url)
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("boom", request=request, response=response)


class DummyHttpClient:
    """
    Simpler Client, dessen post(url, json=...) je nach URL eine vorab konfigurierte Antwort liefert.
    - mapping: Dict[url_prefix, List[DummyResponse]] (FIFO pro URL)
    Wenn keine Antwort konfiguriert ist, 500.
    """
    def __init__(self, mapping: Dict[str, List[DummyResponse]]):
        self.mapping = mapping

    def post(self, url: str, json: Dict[str, Any] | None = None):
        # finde das erste Mapping, das zum url prefix passt
        for prefix, queue in self.mapping.items():
            if url.startswith(prefix):
                if queue:
                    return queue.pop(0)
                return DummyResponse(url, 500, {"error": "no more queued responses"})
        return DummyResponse(url, 500, {"error": "unmapped url"})


# ──────────────────────────────────────────────────────────────────────────────
# Imports unter Test
# ──────────────────────────────────────────────────────────────────────────────

from graphrag.authenticated_ollama_llm import AuthenticatedOllamaLLM


BASE = "http://fake.ollama.local"
CHAT = f"{BASE}/api/chat"
COMP = f"{BASE}/v1/completions"
GEN  = f"{BASE}/api/generate"


def make_llm(monkeypatch, client: DummyHttpClient) -> AuthenticatedOllamaLLM:
    """Erzeugt den LLM und injiziert den Dummy-HTTP-Client."""
    llm = AuthenticatedOllamaLLM(
        model_name="llama3.2:latest",
        base_url=BASE,
        api_key="test-key",
        request_timeout=5.0,
        max_tokens=64,
        temperature=0.0,
    )
    # echten httpx.Client ersetzen
    monkeypatch.setattr(llm, "_http_client", client, raising=True)
    return llm


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_chat_endpoint_success_content_key(monkeypatch):
    """
    Priorität 1: /api/chat liefert message.content -> Erfolg.
    """
    mapping = {
        CHAT: [DummyResponse(CHAT, 200, {"message": {"role": "assistant", "content": "Hello from chat"}})]
    }
    llm = make_llm(monkeypatch, DummyHttpClient(mapping))

    out = llm.complete("Hi there")
    assert out.text == "Hello from chat"


def test_chat_endpoint_success_text_key(monkeypatch):
    """
    /api/chat liefert message.text -> Erfolg.
    """
    mapping = {
        CHAT: [DummyResponse(CHAT, 200, {"message": {"role": "assistant", "text": "Hi via text"}})]
    }
    llm = make_llm(monkeypatch, DummyHttpClient(mapping))

    out = llm.complete("Hi there")
    assert out.text == "Hi via text"


def test_chat_empty_then_openai_completions_success(monkeypatch):
    """
    /api/chat antwortet 200 aber leerer message.content -> Extraction-Error -> Fallback auf /v1/completions.
    """
    mapping = {
        CHAT: [DummyResponse(CHAT, 200, {"message": {"role": "assistant", "content": ""}})],
        COMP: [DummyResponse(COMP, 200, {"choices": [{"text": "From completions"}]})],
    }
    llm = make_llm(monkeypatch, DummyHttpClient(mapping))

    out = llm.complete("Hi there")
    assert out.text == "From completions"


def test_404_fallback_to_generate(monkeypatch):
    """
    /api/chat 404 -> weiter; /v1/completions 404 -> weiter; /api/generate liefert response.
    """
    mapping = {
        CHAT: [DummyResponse(CHAT, 404)],
        COMP: [DummyResponse(COMP, 404)],
        GEN:  [DummyResponse(GEN, 200, {"response": "Gen says hello"})],
    }
    llm = make_llm(monkeypatch, DummyHttpClient(mapping))

    out = llm.complete("Hi")
    assert out.text == "Gen says hello"


def test_422_skips_and_tries_next(monkeypatch):
    """
    /api/chat -> 422 Validation Error (HTTPStatusError) -> Code soll 'continue' machen und nächste Route probieren.
    """
    # Für 422 muss raise_for_status() auslösen -> DummyResponse mit 422
    mapping = {
        CHAT: [DummyResponse(CHAT, 422, {"error": "bad payload"})],
        COMP: [DummyResponse(COMP, 200, {"choices": [{"text": "Works after 422"}]})],
    }
    llm = make_llm(monkeypatch, DummyHttpClient(mapping))

    out = llm.complete("Payload maybe odd")
    assert out.text == "Works after 422"


def test_403_auth_fails_with_clear_message(monkeypatch):
    """
    /api/chat -> 403 -> Code hebt ValueError mit Hilfetext.
    """
    mapping = {
        CHAT: [DummyResponse(CHAT, 403, {"error": "forbidden"})],
    }
    llm = make_llm(monkeypatch, DummyHttpClient(mapping))

    with pytest.raises(ValueError) as exc:
        llm.complete("Hi")
    assert "Authentication failed (403)" in str(exc.value)


def test_unexpected_format_tries_next_and_then_fails(monkeypatch):
    """
    /api/chat liefert 200 aber kein extrahierbarer String in message (nur verschachtelte Dicts),
    /v1/completions liefert 500, /api/generate liefert 500 -> am Ende ValueError.
    """
    mapping = {
        CHAT: [DummyResponse(
            CHAT,
            200,
            {
                "model": "ok",
                "message": {
                    "role": "assistant",
                    # Wichtig: KEIN direktes String-Feld auf Top-Level!
                    "payload": {"foo": 1, "bar": {"baz": 2}}
                }
            }
        )],
        COMP: [DummyResponse(COMP, 500, {"error": "server err"})],
        GEN:  [DummyResponse(GEN, 500, {"error": "server err2"})],
    }
    llm = make_llm(monkeypatch, DummyHttpClient(mapping))

    with pytest.raises(ValueError) as exc:
        llm.complete("Hi")
    msg = str(exc.value)
    assert "All endpoints failed" in msg



def test_metadata_properties(monkeypatch):
    """
    Metadaten sind gesetzt (context_window, num_output, model_name).
    """
    mapping = {
        CHAT: [DummyResponse(CHAT, 200, {"message": {"role": "assistant", "content": "ok"}})],
    }
    llm = make_llm(monkeypatch, DummyHttpClient(mapping))

    meta = llm.metadata
    assert meta.context_window == 4096
    assert meta.num_output == 64
    assert meta.model_name == "llama3.2:latest"


def test_stream_complete_yields_completion(monkeypatch):
    """
    stream_complete delegiert aktuell auf complete() und yieldet genau eine Antwort.
    """
    mapping = {
        CHAT: [DummyResponse(CHAT, 200, {"message": {"role": "assistant", "content": "stream-ok"}})],
    }
    llm = make_llm(monkeypatch, DummyHttpClient(mapping))

    gen = llm.stream_complete("Hi stream")
    # sollte genau einen Wert liefern
    out = next(gen)
    with pytest.raises(StopIteration):
        next(gen)

    assert out.text == "stream-ok"
