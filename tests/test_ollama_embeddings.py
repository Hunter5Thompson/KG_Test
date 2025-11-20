# tests/test_ollama_embeddings.py
from __future__ import annotations
import json
import types
import builtins
from typing import Any, Dict, List, Optional
import pytest
import asyncio
from pathlib import Path
import sys

# --- Optional: Projektwurzel auf sys.path setzen, falls nötig ---
# Passe den relativen Pfad an, falls deine Struktur anders ist.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- Import zu testenem Modul ---
from embeddings.ollama_embeddings import OllamaEmbedding


# ─────────────────────────────────────────────────────────────────
# Test-Hilfen: Fake HTTP Response & Fake Client
# ─────────────────────────────────────────────────────────────────
class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeHttpClient:
    """Minimaler Fake für httpx.Client, um .post() und .close() zu kontrollieren."""
    def __init__(self, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0):
        self.headers = headers or {}
        self.timeout = timeout
        self.closed = False
        self._next_response: Optional[FakeResponse] = None
        self._raise: Optional[BaseException] = None
        self.post_calls: List[Dict[str, Any]] = []

    # Ermöglicht flexibel das nächste Verhalten zu setzen
    def queue_response(self, resp: FakeResponse) -> None:
        self._next_response = resp
        self._raise = None

    def queue_exception(self, exc: BaseException) -> None:
        self._next_response = None
        self._raise = exc

    def post(self, endpoint: str, json: Dict[str, Any]) -> FakeResponse:
        self.post_calls.append({"endpoint": endpoint, "json": json})
        if self._raise:
            raise self._raise
        return self._next_response or FakeResponse(500, {})

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def patch_httpx_client(monkeypatch):
    """
    Fixture, um httpx.Client durch unseren Fake zu ersetzen und
    pro Test den Fake zurückzugeben.
    """
    fake_client = FakeHttpClient()

    def fake_client_ctor(*args, **kwargs):
        # Übernehme Header/Timeout in Fake
        return FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))

    # Wir ersetzen den Konstruktor, merken uns aber die Instanz,
    # indem wir sie nach Erstellung aus dem OllamaEmbedding ziehen.
    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", fake_client_ctor)
    return fake_client


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def make_instance(monkeypatch, nested: bool = True, ok: bool = True) -> OllamaEmbedding:
    """
    Hilfsfunktion: erzeugt eine OllamaEmbedding-Instanz mit gefakter
    httpx.Client.post-Response für den 'test'-Probeaufruf (Dimensionserkennung).
    """
    # Wir wollen die frisch erzeugte httpx.Client Instanz bekommen:
    created_clients: List[FakeHttpClient] = []

    def intercept_ctor(*args, **kwargs):
        inst = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created_clients.append(inst)
        return inst

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", intercept_ctor)

    # Konstruiere Rückgabe für den Erstdurchlauf (_get_embedding_dim)
    if ok:
        payload = {"embeddings": [[0.1, 0.2, 0.3]]} if nested else {"embeddings": [0.1, 0.2, 0.3]}
        first_resp = FakeResponse(200, payload)
    else:
        first_resp = FakeResponse(500, {})  # Fehler => Dimension-Fallback

    # Instanz bauen
    inst = OllamaEmbedding(
        model_name="qwen3-embedding:4b-q8_0",
        base_url="http://host",
        api_key="secret",
    )

    # Die bei __init__ erzeugte Client-Instanz herausholen und Response für die Dim-Erkennung setzen
    assert created_clients, "httpx.Client should have been constructed"
    client = created_clients[0]
    client.queue_response(first_resp)

    return inst


def test_init_sets_headers_and_timeout(monkeypatch):
    created: List[FakeHttpClient] = []

    def ctor(*args, **kwargs):
        c = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created.append(c)
        return c

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", ctor)

    inst = OllamaEmbedding(model_name="m", base_url="http://a", api_key="KEY", embed_batch_size=7)
    assert created, "Client was not constructed"
    c = created[0]
    # Auth Header korrekt?
    assert c.headers.get("Authorization") == "Bearer KEY"
    assert c.headers.get("Content-Type") == "application/json"
    # Timeout gesetzt?
    assert c.timeout == 30.0
    # Pydantic-Felder vom Elternteil
    assert inst.model_name == "m"


def test_dim_detection_nested_list(monkeypatch):
    inst = make_instance(monkeypatch, nested=True, ok=True)
    # Bei verschachteltem Embeddings-Array ist dim=len(inner)
    dim = inst._get_embedding_dim()
    assert dim == 3


def test_dim_detection_flat_list(monkeypatch):
    inst = make_instance(monkeypatch, nested=False, ok=True)
    dim = inst._get_embedding_dim()
    assert dim == 3


def test_dim_detection_fallback_on_error(monkeypatch):
    inst = make_instance(monkeypatch, ok=False)
    dim = inst._get_embedding_dim()
    # Fallback laut Code: 2560
    assert dim == 2560


def test_get_text_embedding_success_nested(monkeypatch):
    # Arrange
    created: List[FakeHttpClient] = []

    def ctor(*args, **kwargs):
        c = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created.append(c)
        return c

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", ctor)
    inst = OllamaEmbedding("m", "http://a", "KEY")

    # Erfolgsantwort mit verschachtelter Liste
    created[0].queue_response(FakeResponse(200, {"embeddings": [[0.5, 0.6]]}))

    emb = inst.get_text_embedding("hello")
    assert emb == [0.5, 0.6]
    # Endpoint/Body geprüft?
    call = created[0].post_calls[-1]
    assert call["endpoint"].endswith("/api/embed")
    assert call["json"]["model"] == "m"
    assert call["json"]["input"] == "hello"


def test_get_text_embedding_success_flat(monkeypatch):
    created: List[FakeHttpClient] = []

    def ctor(*args, **kwargs):
        c = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created.append(c)
        return c

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", ctor)
    inst = OllamaEmbedding("m", "http://a", "KEY")

    created[0].queue_response(FakeResponse(200, {"embeddings": [0.7, 0.8]}))
    emb = inst.get_text_embedding("x")
    assert emb == [0.7, 0.8]


def test_get_text_embedding_http_500_returns_zeros(monkeypatch):
    created: List[FakeHttpClient] = []

    def ctor(*args, **kwargs):
        c = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created.append(c)
        return c

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", ctor)
    inst = OllamaEmbedding("m", "http://a", "KEY")

    # Keine embeddings -> None -> Fallback Nullvektor
    created[0].queue_response(FakeResponse(500, {}))
    emb = inst.get_text_embedding("y")
    assert isinstance(emb, list) and len(emb) == inst._get_embedding_dim()
    assert all(v == 0.0 for v in emb)


def test_get_text_embedding_exception_returns_zeros(monkeypatch):
    created: List[FakeHttpClient] = []

    def ctor(*args, **kwargs):
        c = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created.append(c)
        return c

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", ctor)
    inst = OllamaEmbedding("m", "http://a", "KEY")

    created[0].queue_exception(RuntimeError("boom"))
    emb = inst.get_text_embedding("z")
    assert isinstance(emb, list) and len(emb) == inst._get_embedding_dim()
    assert all(v == 0.0 for v in emb)


def test_get_query_embedding_identical_path(monkeypatch):
    # Sicherstellen, dass Query denselben Codepfad nutzt
    created: List[FakeHttpClient] = []

    def ctor(*args, **kwargs):
        c = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created.append(c)
        return c

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", ctor)
    inst = OllamaEmbedding("m", "http://a", "KEY")

    created[0].queue_response(FakeResponse(200, {"embeddings": [[1.1, 1.2, 1.3]]}))
    emb = inst.get_query_embedding("what?")
    assert emb == [1.1, 1.2, 1.3]


@pytest.mark.asyncio
async def test_async_methods_call_sync(monkeypatch):
    created: List[FakeHttpClient] = []

    def ctor(*args, **kwargs):
        c = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created.append(c)
        return c

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", ctor)
    inst = OllamaEmbedding("m", "http://a", "KEY")

    created[0].queue_response(FakeResponse(200, {"embeddings": [[9.9, 8.8]]}))
    q = await inst._aget_query_embedding("q")
    t = await inst._aget_text_embedding("t")
    assert q == [9.9, 8.8]
    assert t == [9.9, 8.8]


def test_batch_text_embeddings(monkeypatch):
    created: List[FakeHttpClient] = []

    def ctor(*args, **kwargs):
        c = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created.append(c)
        return c

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", ctor)
    inst = OllamaEmbedding("m", "http://a", "KEY")

    # Wir liefern für jeden Aufruf dieselbe Response zurück – reicht für Funktionspfad
    created[0].queue_response(FakeResponse(200, {"embeddings": [[0.0, 1.0]]}))
    texts = ["a", "b", "c"]
    embs = inst._get_text_embeddings(texts)
    assert embs == [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]


def test_client_closed_on_del(monkeypatch):
    created: List[FakeHttpClient] = []

    def ctor(*args, **kwargs):
        c = FakeHttpClient(headers=kwargs.get("headers"), timeout=kwargs.get("timeout", 30.0))
        created.append(c)
        return c

    monkeypatch.setattr("embeddings.ollama_embeddings.httpx.Client", ctor)
    inst = OllamaEmbedding("m", "http://a", "KEY")
    client = created[0]
    assert client.closed is False

    # __del__ ist nicht deterministisch – wir rufen explizit auf
    inst.__del__()
    assert client.closed is True
