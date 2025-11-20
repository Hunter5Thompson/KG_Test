"""
Basic Knowledge Graph Extraction Example
"""
from llama_index.core import Document
from llama_index.llms.ollama import Ollama
from ollama import Client

from src.embeddings.ollama_embeddings import OllamaEmbedding
from src.storage.neo4j_store import Neo4jStore
from src.extractors.kg_extractor import KnowledgeGraphExtractor
from config.settings import AppConfig


def main():
    """Run basic extraction example"""
    print("=" * 60)
    print("🚀 Knowledge Graph Extraction - Basic Example")
    print("=" * 60)
    
    # Load configuration
    config = AppConfig.from_env()
    
    # Initialize components
    ollama_client = Client(
        host=config.ollama.host,
        headers={"Authorization": f"Bearer {config.ollama.api_key}"}
    )
    
    llm = Ollama(
        model=config.ollama.llm_model,
        base_url=config.ollama.host,
        request_timeout=120,
        client=ollama_client,
    )
    
    embed_model = OllamaEmbedding(
        model_name=config.ollama.embedding_model,
        base_url=config.ollama.host,
        api_key=config.ollama.api_key,
    )
    
    store = Neo4jStore(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password,
    )
    
    # Clear existing data
    store.clear()
    
    # Create extractor
    extractor = KnowledgeGraphExtractor(
        llm=llm,
        embed_model=embed_model,
        store=store
    )
    
    # Sample documents
    documents = [
        Document(text="Alice is the CEO of Acme Corp located in Berlin."),
        Document(text="Acme Corp acquired Beta Ltd in 2022."),
        Document(text="Bob is a software engineer at Acme Corp."),
        Document(text="Beta Ltd specializes in AI and machine learning."),
        Document(text="Charlie works as a data scientist in Berlin."),
    ]
    
    # Extract
    stats = extractor.extract_from_documents(documents, store_embeddings=True)
    
    # Show results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key:30s}: {value}")
    
    # Show graph stats
    graph_stats = store.get_stats()
    print(f"  {'nodes_in_graph':30s}: {graph_stats['nodes']}")
    print(f"  {'relationships_in_graph':30s}: {graph_stats['relationships']}")
    
    # Show sample triplets
    print("\n" + "=" * 60)
    print("🔍 SAMPLE TRIPLETS")
    print("=" * 60)
    triplets = store.get_triplets(limit=10)
    for i, t in enumerate(triplets, 1):
        print(f"  {i}. {t}")
    
    print("\n" + "=" * 60)
    print("✅ Extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()