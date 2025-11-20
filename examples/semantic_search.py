"""
Semantic Search Example
"""

import examples

from src.embeddings.ollama_embeddings import OllamaEmbedding
from src.storage.neo4j_store import Neo4jStore
from config.settings import AppConfig


def main():
    """Run semantic search example"""
    print("=" * 60)
    print("🔎 Semantic Search Demo")
    print("=" * 60)
    
    # Load configuration
    config = AppConfig.from_env()
    
    # Initialize components
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
    
    # Test queries
    queries = [
        "Who is the boss of the company?",
        "Software development and programming",
        "City in Germany"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Get query embedding
        query_embedding = embed_model.get_query_embedding(query)
        
        # Search
        results = store.semantic_search(query_embedding, limit=3)
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['entity']:20s} (similarity: {result['similarity']:.3f})")
    
    print("\n" + "=" * 60)
    print("✅ Search complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()