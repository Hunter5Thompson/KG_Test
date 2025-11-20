# tests/test_kg_extractor.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys
import pytest

# --- Projekt-Src auf sys.path setzen (src-Layout) ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Modul importieren
from extractors.kg_extractor import KnowledgeGraphExtractor
from models.triplets import Triplet


# ──────────────────────────────────────────────────────────────
# Fakes / Test Doubles
# ──────────────────────────────────────────────────────────────
class FakeLLMResponse:
    def __init__(self, text: str):
        self.text = text


class FakeLLM:
    """Minimaler LLM-Ersatz für .complete(prompt) -> object with .text"""
    def __init__(self, canned_texts: Optional[List[str]] = None):
        self.canned = canned_texts or []
        self.calls: List[Dict[str, Any]] = []

    def complete(self, prompt: str) -> FakeLLMResponse:
        self.calls.append({"prompt": prompt})
        if self.canned:
            return FakeLLMResponse(self.canned.pop(0))
        return FakeLLMResponse("")


class FakeEmbeddingModel:
    """Statt echter OllamaEmbedding – stellt nur get_text_embedding bereit."""
    def __init__(self, dim: int = 4, model_name: str = "fake-embed"):
        self.dim = dim
        self.model_name = model_name
        self.calls: List[str] = []

    def get_text_embedding(self, text: str) -> List[float]:
        self.calls.append(text)
        # deterministisches „Embedding“
        return [float(len(text) % 10)] + [0.0] * (self.dim - 1)


class FakeStore:
    """Ersetzt Neo4jStore, protokolliert Aufrufe."""
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def write_triplets(self, triplets: List[Triplet], entity_embeddings: Optional[Dict[str, List[float]]]):
        self.calls.append({"triplets": list(triplets), "embeddings": entity_embeddings})


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

def test_build_extraction_prompt_contains_text_and_rules():
    llm = FakeLLM(["(Alice, works_at, Acme)\n"])
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=None, store=None)

    prompt = extractor._build_extraction_prompt("Hello World.")
    # Sanity: Prompt enthält Regeln, Beispiel-Header und den Text
    assert "RULES:" in prompt
    assert "EXAMPLES:" in prompt
    assert "TEXT: Hello World." in prompt
    assert prompt.strip().endswith("TRIPLETS:")


def test_parse_triplets_basic_and_validation():
    llm = FakeLLM()
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=None, store=None)

    resp = """
    (Alice, works_at, Acme Corp)
    (Acme Corp, located_in, Berlin)
    (   Bob   ,  is_a ,  engineer  )
    (TooLongSubject.............................................................................................................................., rel, obj)
    (Ok, , MissingRelation)  # invalid
    """
    triplets = extractor._parse_triplets(resp)
    # Erwartet: die ersten 3 sind valide; 4. scheitert an Längen-Check (subject > 200?),
    # 5. scheitert an Validation (leeres Feld)
    assert Triplet("Alice", "works_at", "Acme Corp") in triplets
    assert Triplet("Acme Corp", "located_in", "Berlin") in triplets
    assert Triplet("Bob", "is_a", "engineer") in triplets
    # Sicherstellen, dass keine ungültigen Triplets auftauchen
    for t in triplets:
        assert all(len(x) < 200 for x in (t.subject, t.predicate, t.object))


def test_extract_triplets_from_text_calls_llm_and_parses():
    llm_text = "(Alice, works_at, Acme)\n(Acme, located_in, Berlin)"
    llm = FakeLLM([llm_text])
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=None, store=None)

    result = extractor.extract_triplets_from_text("Alice works at Acme in Berlin.", verbose=False)
    assert result == [
        Triplet("Alice", "works_at", "Acme"),
        Triplet("Acme", "located_in", "Berlin"),
    ]
    # LLM wurde aufgerufen und Prompt enthält den Text
    assert llm.calls and "TEXT: Alice works at Acme in Berlin." in llm.calls[0]["prompt"]


def test_extract_triplets_empty_llm_response_returns_empty_list():
    llm = FakeLLM([""])  # leere Antwort – simuliert UI-Fehlerfall (content='')
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=None, store=None)

    result = extractor.extract_triplets_from_text("foo bar", verbose=False)
    assert result == []


def test_compute_embeddings_deduplicates_and_calls_embedder_once_per_entity():
    llm = FakeLLM()
    embed = FakeEmbeddingModel(dim=3)
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=embed, store=None)

    triplets = [
        Triplet("Alice", "works_at", "Acme"),
        Triplet("Bob", "works_at", "Acme"),      # "Acme" taucht doppelt auf
        Triplet("Acme", "located_in", "Berlin"),
    ]
    embs = extractor._compute_embeddings(triplets, verbose=False)

    # Drei Entities -> drei Embeddings
    assert set(embs.keys()) == {"Alice", "Bob", "Acme", "Berlin"} - {"Berlin"} | {"Berlin"}  # formell
    assert set(embs.keys()) == {"Alice", "Bob", "Acme", "Berlin"}
    assert len(embed.calls) == 4  # jede Entity genau einmal
    # Dimension passt
    for v in embs.values():
        assert isinstance(v, list) and len(v) == 3


def test_extract_from_documents_without_store_or_embeddings():
    llm = FakeLLM([
        "(Alice, works_at, Acme)",     # Doc1
        "(Acme, located_in, Berlin)",  # Doc2
    ])
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=None, store=None)

    from llama_index.core import Document
    docs = [Document(text="...1..."), Document(text="...2...")]

    stats = extractor.extract_from_documents(docs, store_embeddings=False, verbose=False)
    assert stats["documents_processed"] == 2
    assert stats["triplets_extracted"] == 2
    assert stats["embeddings_computed"] is False


def test_extract_from_documents_with_embeddings_and_store_called():
    llm = FakeLLM([
        "(Alice, works_at, Acme)\n(Bob, works_at, Acme)",  # Doc1
        "(Acme, located_in, Berlin)",                      # Doc2
    ])
    embed = FakeEmbeddingModel(dim=2)
    store = FakeStore()
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=embed, store=store)

    from llama_index.core import Document
    docs = [Document(text="doc1"), Document(text="doc2")]

    stats = extractor.extract_from_documents(docs, store_embeddings=True, verbose=False)

    # Stats korrekt
    assert stats["documents_processed"] == 2
    assert stats["triplets_extracted"] == 3
    assert stats["embeddings_computed"] is True

    # Store wurde genau einmal geschrieben
    assert len(store.calls) == 1
    call = store.calls[0]
    written_triplets = call["triplets"]
    written_embs = call["embeddings"]

    assert len(written_triplets) == 3
    assert isinstance(written_embs, dict) and set(written_embs.keys()) == {"Alice", "Bob", "Acme", "Berlin"}

    # Embedding-Model wurde für alle Entities aufgerufen
    assert set(embed.calls) == {"Alice", "Bob", "Acme", "Berlin"}


def test_extract_from_documents_embeddings_but_no_model_does_not_crash():
    """Wenn store_embeddings=True aber kein embed_model gesetzt ist, darf es nicht crashen."""
    llm = FakeLLM(["(A, r, B)"])
    store = FakeStore()
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=None, store=store)

    from llama_index.core import Document
    docs = [Document(text="doc")]

    stats = extractor.extract_from_documents(docs, store_embeddings=True, verbose=False)
    assert stats["triplets_extracted"] == 1
    assert stats["embeddings_computed"] is False
    # Store sieht embeddings=None
    assert store.calls and store.calls[0]["embeddings"] is None


def test_parse_triplets_ignores_malformed_lines_and_trims_spaces():
    llm = FakeLLM()
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=None, store=None)

    resp = """
    (  Alice   ,   works_at ,   Acme  )
    (bad line no commas)
    (Charlie, is_a, )
    ( , relates , Something)
    (Delta, located_in, Munich)
    """
    triplets = extractor._parse_triplets(resp)
    assert Triplet("Alice", "works_at", "Acme") in triplets
    assert Triplet("Delta", "located_in", "Munich") in triplets
    # Ungültige Zeilen sind rausgefiltert
    assert all(all(x for x in (t.subject, t.predicate, t.object)) for t in triplets)


def test_prompt_used_for_every_document():
    texts = [
        "(A, r, B)",  # doc1
        "(C, r, D)",  # doc2
        "(E, r, F)",  # doc3
    ]
    llm = FakeLLM(texts.copy())
    extractor = KnowledgeGraphExtractor(llm=llm, embed_model=None, store=None)

    from llama_index.core import Document
    docs = [Document(text=f"doc{i}") for i in range(3)]
    extractor.extract_from_documents(docs, store_embeddings=False, verbose=False)

    # Für jedes Dokument ein LLM-Aufruf, Prompt enthält jeweiligen Text
    assert len(llm.calls) == 3
    for i, c in enumerate(llm.calls):
        assert f"TEXT: doc{i}" in c["prompt"]
