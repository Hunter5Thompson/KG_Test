# GraphRAG Agent Documentation

## 🎯 Überblick

Der **GraphRAG Agent** ist ein intelligenter Multi-Tool Agent der automatisch zwischen verschiedenen Retrieval-Strategien wählt um Fragen über deinen Knowledge Graph zu beantworten.

### Architecture

```
User Query
    ↓
Agent Reasoning (LLM)
    ↓
Tool Selection
    ├─→ Semantic Search (Vector Similarity)
    ├─→ Cypher Query (Graph Traversal)
    └─→ Hybrid Retrieve (Combined Strategy)
    ↓
Answer Synthesis
    ↓
Final Response
```

---

## 🛠️ Features

### 1. **Drei Retrieval-Modi**

| Tool | Wann nutzen? | Beispiel |
|------|-------------|----------|
| **Semantic Search** | Konzeptuelle Ähnlichkeit | "What are wargaming methodologies?" |
| **Cypher Query** | Strukturierte Beziehungen | "Who authored what papers?" |
| **Hybrid Retrieve** | Komplexe Multi-Aspekt Fragen | "Tell me everything about NATO" |

### 2. **Safety Features**

- ✅ **READ-only Cypher**: Keine WRITE/DELETE/MERGE Operations
- ✅ **Query Validation**: CypherGuard blockiert gefährliche Queries
- ✅ **Timeout Protection**: 10s Limit pro Query
- ✅ **Iteration Limit**: Max 5 Tool-Calls pro Query (konfigurierbar)

### 3. **ReAct Pattern**

Der Agent nutzt **Reasoning + Acting**:
1. **Reason**: LLM analysiert die Frage
2. **Act**: Wählt passendes Tool
3. **Observe**: Erhält Tool-Output
4. **Repeat**: Falls nötig, weitere Tools
5. **Synthesize**: Generiert finale Antwort

---

## 🚀 Quick Start

### Installation

```bash
# Dependencies installieren
uv sync --extra graphrag

# oder mit pip
pip install langchain langchain-community langgraph langchain-neo4j
```

### Basic Usage

```python
from config.settings import AppConfig
from neo4j import GraphDatabase
from src.graphrag.agent import GraphRAGAgent

# Config laden
config = AppConfig.from_env()

# Neo4j Connection
driver = GraphDatabase.driver(
    config.neo4j.uri,
    auth=(config.neo4j.user, config.neo4j.password)
)

# Agent initialisieren
agent = GraphRAGAgent(config, driver)

# Query ausführen
result = agent.query("What are wargaming methodologies?")

print(result['answer'])
print(f"Used {result['tool_calls']} tool(s)")

# Cleanup
agent.close()
driver.close()
```

---

## 📊 CLI Commands

```bash
# Agent testen (verschiedene Modi)
uv run python tests/test_agent.py --mode basic
uv run python tests/test_agent.py --mode tools
uv run python tests/test_agent.py --mode conversation
uv run python tests/test_agent.py --mode all

# UI mit Agent starten
uv run streamlit run src/ui/app.py --server.port 8502

# Agent standalone (main test in agent.py)
uv run python src/graphrag/agent.py
```

---

## 🎭 Tool Details

### 1. Semantic Search
- Vector-basierte Ähnlichkeitssuche
- Nutzt Ollama Embeddings (2560-dim)
- Best für: Konzepte, Themen, explorative Suche

### 2. Cypher Query
- Strukturierte Graph-Queries
- READ-only mit Safety Guard
- Best für: Relationships, Pfade, Aggregationen

### 3. Hybrid Retrieve
- Kombiniert Vector + Graph + Keyword
- Score Fusion & Re-Ranking
- Best für: Komplexe Multi-Aspekt Fragen

---

## 🐛 Troubleshooting

**Problem: Cypher-Queries schlagen fehl**
```python
from src.graphrag.agent import CypherGuard
safe, error = CypherGuard.is_safe(query)
print(f"Safe: {safe}, Error: {error}")
```

**Problem: Agent macht zu viele Tool-Calls**
```python
agent = GraphRAGAgent(config, driver, max_iterations=3)
```

**Problem: Semantic Search findet nichts**
```cypher
MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN count(n)
```

---

## 📚 Full Documentation

Siehe `AGENT_README.md` für:
- Detaillierte API Reference
- Code Examples
- Performance Tips
- Roadmap

---

**Built with ❤️ for GISA GmbH Bachelor-Thesis**
