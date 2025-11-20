from typing import List

import math
import sys
import types

import pytest

try:  # neo4j may be unavailable in CI; fall back to a stub
    from neo4j.exceptions import Neo4jError
except Exception:  # pragma: no cover - fallback for environments without neo4j installed
    class Neo4jError(Exception):
        pass

    class ClientError(Exception):
        pass

    class CypherSyntaxError(Exception):
        pass

    fake_neo4j = types.SimpleNamespace(
        Driver=object,
        GraphDatabase=types.SimpleNamespace(driver=lambda *args, **kwargs: None),
    )
    sys.modules["neo4j"] = fake_neo4j
    sys.modules["neo4j.exceptions"] = types.SimpleNamespace(
        Neo4jError=Neo4jError, ClientError=ClientError, CypherSyntaxError=CypherSyntaxError
    )

# numpy is required by HybridGraphRetriever; provide a lightweight fallback
try:  # pragma: no cover - prefer real numpy if available
    import numpy  # type: ignore # noqa: F401
except Exception:  # pragma: no cover - shim minimal numpy API
    def _norm(vec):
        try:
            return math.sqrt(sum(float(x) * float(x) for x in vec))
        except Exception:
            return float("nan")

    def _dot(a, b):
        return float(sum(float(x) * float(y) for x, y in zip(a, b)))

    fake_np = types.SimpleNamespace(
        array=lambda x: x,
        ndarray=tuple,
        linalg=types.SimpleNamespace(norm=_norm),
        dot=_dot,
        log1p=math.log1p,
    )
    sys.modules["numpy"] = fake_np

if "dotenv" not in sys.modules:  # pragma: no cover - prevent import errors
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

if "tqdm" not in sys.modules:  # pragma: no cover - lightweight shim
    sys.modules["tqdm"] = types.SimpleNamespace(tqdm=lambda x, **kwargs: x)

# Minimal langchain/langgraph shims so we can import the executor without heavy deps
if "langchain_core.messages" not in sys.modules:  # pragma: no cover - test shim
    class _Msg:
        def __init__(self, content: str = ""):
            self.content = content

    sys.modules["langchain_core.messages"] = types.SimpleNamespace(
        AIMessage=_Msg, BaseMessage=_Msg, HumanMessage=_Msg, SystemMessage=_Msg
    )

if "langchain_core.language_models.llms" not in sys.modules:  # pragma: no cover
    class _LLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, messages, **kwargs):
            return messages[-1] if messages else _Msg("")

    sys.modules["langchain_core.language_models.llms"] = types.SimpleNamespace(LLM=_LLM)

if "langgraph.graph" not in sys.modules:  # pragma: no cover
    class _StateGraph:
        def __init__(self, *args, **kwargs):
            pass

        def add_node(self, *args, **kwargs):
            return None

        def set_entry_point(self, *args, **kwargs):
            return None

        def add_conditional_edges(self, *args, **kwargs):
            return None

        def add_edge(self, *args, **kwargs):
            return None

        def compile(self):
            return types.SimpleNamespace(stream=lambda state: iter([]))

    sys.modules["langgraph.graph"] = types.SimpleNamespace(StateGraph=_StateGraph, END="END")

if "langgraph.graph.message" not in sys.modules:  # pragma: no cover
    sys.modules["langgraph.graph.message"] = types.SimpleNamespace(add_messages=lambda x, y: x)

from src.graphrag.agent import GraphRAGToolExecutor


class FakeResult:
    def __init__(self, data=None, values=None, single=None):
        self._data = data or []
        self._values = values
        self._single = single

    def data(self):
        return self._data

    def value(self):
        return self._values

    def single(self):
        return self._single


class FakeSession:
    def __init__(self, results: List[FakeResult], raise_error: Exception | None = None):
        self.results = list(results)
        self.raise_error = raise_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, *args, **kwargs):
        if self.raise_error:
            raise self.raise_error
        if not self.results:
            return FakeResult()
        return self.results.pop(0)


class FakeDriver:
    def __init__(self, results: List[FakeResult], raise_error: Exception | None = None):
        self.results = results
        self.raise_error = raise_error

    def session(self, *_, **__):
        return FakeSession(list(self.results), raise_error=self.raise_error)


@pytest.fixture
def executor():
    driver = FakeDriver([])
    return GraphRAGToolExecutor(driver, embed_fn=lambda x: [0.0, 0.0])


def test_cypher_guardrail_blocks_write(executor):
    response = executor.execute(
        "cypher_query",
        {"description": "danger", "cypher": "CREATE (n:Test {id:1}) RETURN n"},
    )
    assert GraphRAGToolExecutor.ERR_CYPHER_GUARDRAIL in response


def test_schema_overview_combines_metadata():
    driver = FakeDriver(
        [
            FakeResult(values=["Entity"]),
            FakeResult(values=["RELATED_TO"]),
            FakeResult(data=[{"nodeLabels": ["Entity"], "propertyName": "id"}]),
        ]
    )
    executor = GraphRAGToolExecutor(driver, embed_fn=lambda x: [0.0, 0.0])

    response = executor.execute("schema_overview", {})
    assert "Entity" in response
    assert "RELATED_TO" in response
    assert "id" in response


def test_telemetry_records_latency(executor):
    executor.execute("semantic_search", {"query": "test"})
    # semantic search will return unknown tool message if retriever fails, but latency should be recorded
    assert executor.telemetry
    assert any(lat > 0 for lat in executor.telemetry.get("semantic_search", []))


def test_runtime_error_is_structured():
    driver = FakeDriver([], raise_error=Neo4jError("boom"))
    executor = GraphRAGToolExecutor(driver, embed_fn=lambda x: [0.0, 0.0])

    response = executor.execute("schema_overview", {})
    assert GraphRAGToolExecutor.ERR_CYPHER_RUNTIME in response
