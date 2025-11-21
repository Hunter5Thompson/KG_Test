"""GraphRAG Agent Test"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from config.settings import AppConfig
from src.graphrag import create_graphrag_agent, run_agent
from src.graphrag.langchain_ollama_auth import create_authenticated_ollama_llm  # ← NUTZE DEINE KLASSE!
from src.embeddings.ollama_embeddings import OllamaEmbedding


def test_agent():
    print("🧪 GraphRAG Agent Test (Prompt-Based Tool Calling)\n")
    print("="*60)
    
    config = AppConfig.from_env()
    
    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password)
    )
    
    embedder = OllamaEmbedding(
        model_name=config.ollama.embedding_model,
        base_url=config.ollama.host,
        api_key=config.ollama.api_key
    )
    
    # ✅ NUTZE DEINE BESTEHENDE KLASSE!
    llm = create_authenticated_ollama_llm(
        model_name=config.ollama.llm_model,
        base_url=config.ollama.host,
        api_key=config.ollama.api_key,
        temperature=0.0
    )
    
    print(f"✅ Using model: {config.ollama.llm_model}")
    print(f"✅ Embeddings: {config.ollama.embedding_model}\n")
    
    agent = create_graphrag_agent(llm, driver, embedder.get_query_embedding, max_iterations=10)
    
    test_cases = [
        "What is wargaming?",
        "What are NATO's key contributions?",
        "Find entities related to simulation",
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'='*60}")
        
        result = run_agent(agent, query, verbose=True)
        
        print(f"\n📊 Stats:")
        print(f"  Tool Calls: {result['tool_calls']}")
        print(f"  Answer Length: {len(result['answer'])} chars")
    
    driver.close()
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_agent()