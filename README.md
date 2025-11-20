# Knowledge Graph Extractor

Production-ready knowledge graph extraction from text using LLMs and semantic embeddings.

## Features

- 🧠 **LLM-based extraction** - Uses local Ollama models
- 🔍 **Semantic search** - Vector embeddings for entity search
- 💾 **Neo4j storage** - Scalable graph database
- 🏗️ **Modular architecture** - Clean, testable, maintainable

## Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your credentials
```

## Quick Start
```python
from examples.basic_extraction import main
main()
```

## Project Structure
```
kg_test/
├── src/
│   ├── embeddings/      # Embedding models
│   ├── extractors/      # KG extraction logic
│   ├── storage/         # Database backends
│   └── models/          # Data models
├── config/              # Configuration
├── examples/            # Usage examples
└── tests/               # Unit tests
```

## Usage

### Basic Extraction
```python
from src.extractors.kg_extractor import KnowledgeGraphExtractor
from llama_index.core import Document

extractor = KnowledgeGraphExtractor(llm, embed_model, store)
docs = [Document(text="Alice works at Acme Corp.")]
stats = extractor.extract_from_documents(docs)
```

### Semantic Search
```python
from src.storage.neo4j_store import Neo4jStore

store = Neo4jStore(uri, user, password)
query_embedding = embed_model.get_query_embedding("CEO")
results = store.semantic_search(query_embedding, limit=5)
```

## Testing
```bash
pytest tests/
```

## License

MIT