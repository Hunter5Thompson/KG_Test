# tests/test_agent.py
from __future__ import annotations

import sys
import types
from typing import Any, Dict, List

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Fakes (Retriever, Neo4j, LLM)
# ──────────────────────────────────────────────────────────────────────────────

class FakeRetrievalResult:
    """Minimaler Ersatz für graphrag.hybrid_retriever.RetrievalResult"""
    def __init__(self, entity_id: str, score: float, source: str = "vector", metadata: Dict | None = None):
        self.entity_id = entity_id
        self.score = score
        self.source = source
        self.metadata = metadata or {}


class FakeHybridGraphRetriever:
    """Fake Retriever: deterministische Ergebnisse + konstruierter Kontext."""
    def __init__(self, driver, embed_fn):
        self.driver = driver
        self.embed_fn = embed_fn
        self.calls: List[Dict[str, Any]] = []

    def retrieve(self, query: str, strategy: str = "hybrid", top_k: int = 5, **kwargs):
        self.calls.append({"query": query, "strategy": strategy, "top_k": top_k})
        if strategy == "vector":
            return [
                FakeRetrievalResult("Entity_A", 0.9, "vector"),
                FakeRetrievalResult("Entity_B", 0.85, "vector"),
            ]
        return [
            FakeRetrievalResult("Entity_X", 0.95, "vector", {"sources": ["vector", "graph"]}),
            FakeRetrievalResult("Entity_Y", 0.88, "graph", {"sources": ["graph"]}),
        ]

    def get_context_for_entities(self, results, include_neighbors: bool = True) -> str:
        return "CONTEXT: " + ", ".join(r.entity_id for r in results)


class FakeNeo4jRecord:
    def __init__(self, d: Dict[str, Any]):
        self._d = d

    def data(self) -> Dict[str, Any]:
        return dict(self._d)


class FakeNeo4jSession:
    """Fake Neo4j Session: nur READ-Queries; liefert 2 Dummy-Records."""
    def __init__(self):
        self.queries: List[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query: str):
        self.queries.append(query)
        # READ-Ergebnis
        return [
            FakeNeo4jRecord({"n.id": "A"}),
            FakeNeo4jRecord({"n.id": "B"}),
        ]


class FakeNeo4jDriver:
    def __init__(self):
        self.sessions: List[FakeNeo4jSession] = []

    def session(self):
        s = FakeNeo4jSession()
        self.sessions.append(s)
        return s


class FakeLLM:
    """
    Fake LangChain-kompatibles LLM:
    - bind_tools(tools) -> speichert Tools und returns self
    - invoke(messages):
        * 1. Aufruf: gibt tool_calls (hybrid_retrieve) aus (mit ID!)
        * 2. Aufruf: nach ToolMessage finale Antwort
    """
    def __init__(self):
        self.bound_tools = None
        self.invocations: List[List[Any]] = []
        self._saw_tool_message = False
        self._counter = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    @property
    def saw_tool_message(self) -> bool:
        return self._saw_tool_message

    def invoke(self, messages):
        from langchain_core.messages import AIMessage, ToolMessage

        self.invocations.append(list(messages))
        self._saw_tool_message = any(isinstance(m, ToolMessage) for m in messages)
        self._counter += 1

        if not self._saw_tool_message:
            # Erste Runde: Tool-Call erzwingen — jetzt mit 'id'
            return AIMessage(
                content="(calling tool)",
                tool_calls=[{
                    "id": f"call_{self._counter}",
                    "name": "hybrid_retrieve",
                    "args": {"query": "test query", "top_k": 3},
                    # 'type' ist optional bei LangChain, aber schadet nicht:
                    "type": "tool_call",
                }],
            )

        # Zweite Runde: finale Antwort nach Toolausführung
        return AIMessage(content="FINAL ANSWER: synthesized from tool results")


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: graphrag.hybrid_retriever patchen (nur Submodul, nicht das Paket)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_hybrid_retriever(monkeypatch):
    """
    Stellt sicher, dass 'from graphrag.hybrid_retriever import HybridGraphRetriever'
    unseren Fake verwendet. Greift *erst* nachdem das Paket importierbar ist.
    """
    # Wenn das Submodul schon existiert, ersetzen wir nur die Klasse
    try:
        import graphrag.hybrid_retriever as real_mod  # noqa
        monkeypatch.setattr(real_mod, "HybridGraphRetriever", FakeHybridGraphRetriever, raising=True)
        yield
        return
    except Exception:
        pass

    # Falls das Submodul nicht existiert: sauberes Submodul erzeugen
    fake_sub = types.ModuleType("graphrag.hybrid_retriever")
    setattr(fake_sub, "HybridGraphRetriever", FakeHybridGraphRetriever)
    sys.modules["graphrag.hybrid_retriever"] = fake_sub
    yield


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_agent_end_to_end_hybrid_call_and_final_answer():
    """Agent-Flow: agent -> tools(hybrid_retrieve) -> agent -> final message"""
    from graphrag.agent import create_graphrag_agent, run_agent

    driver = FakeNeo4jDriver()

    def embed_fn(q: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    llm = FakeLLM()
    app = create_graphrag_agent(llm, driver, embed_fn)

    answer = run_agent(app, "What is wargaming?", verbose=False)

    # LLM sollte ToolMessage gesehen haben (ToolNode wurde ausgeführt)
    assert llm.saw_tool_message is True
    assert isinstance(answer, str) and "FINAL ANSWER" in answer


def test_semantic_search_tool_direct_invoke():
    """Direkter Tool-Test: semantic_search liefert formatierte Vector-Result-Strings."""
    from graphrag.agent import GraphRAGTools

    driver = FakeNeo4jDriver()
    embed_fn = lambda q: [0.0, 0.0, 0.0]  # noqa: E731

    tools = GraphRAGTools(driver, embed_fn).create_tools()
    sem_tool = next(t for t in tools if getattr(t, "name", "") == "semantic_search")

    out = sem_tool.invoke({"query": "alpha", "top_k": 2})
    assert isinstance(out, str)
    assert "Vector Search Results" in out
    # kommt aus FakeHybridGraphRetriever
    assert "Entity_A" in out
    assert "Entity_B" in out


def test_cypher_query_blocks_write_and_handles_read():
    """cypher_query blockiert WRITE-Statements und formatiert READ-Ergebnisse."""
    from graphrag.agent import GraphRAGTools

    driver = FakeNeo4jDriver()
    tools = GraphRAGTools(driver, lambda q: [0.0]).create_tools()
    cy_tool = next(t for t in tools if getattr(t, "name", "") == "cypher_query")

    # WRITE blockieren
    out_bad = cy_tool.invoke({"query": "MATCH (n) CREATE (m:Test) RETURN m"})
    assert "ERROR: WRITE operations are not allowed" in out_bad

    # READ ok
    out_ok = cy_tool.invoke({"query": "MATCH (n) RETURN n.id LIMIT 2"})
    assert "Query returned" in out_ok
    assert "n.id" in out_ok
