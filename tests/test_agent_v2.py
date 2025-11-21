"""
Test für den GraphRAG Agent mit LangGraph v1.0+.
Supports CLI arguments für selective testing.

Usage:
    python tests/test_agent.py              # Run all tests
    python tests/test_agent.py --mode basic # Run only basic test
    python tests/test_agent.py --mode hybrid
    python tests/test_agent.py --mode cypher
    python tests/test_agent.py --mode safety
"""

import logging
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # kg_test/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.graphrag import create_graphrag_agent, run_agent
from src.graphrag.langchain_ollama_auth import create_authenticated_ollama_llm
from src.embeddings.ollama_embeddings import OllamaEmbedding
from neo4j import GraphDatabase
from config.settings import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_agent():
    """Setup Agent components (reusable)."""
    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password)
    )
    
    embedder = OllamaEmbedding(
        model_name=config.ollama.embedding_model,
        base_url=config.ollama.host,
        api_key=config.ollama.api_key
    )
    
    llm = create_authenticated_ollama_llm(
        model_name=config.ollama.llm_model,
        base_url=config.ollama.host,
        api_key=config.ollama.api_key,
        temperature=0.0
    )
    
    agent = create_graphrag_agent(llm, driver, embedder.get_query_embedding)
    
    return agent, driver


def test_agent_basic():
    """Test: Einfache semantische Suche."""
    print("\n" + "="*80)
    print("TEST 1: Basic Semantic Search")
    print("="*80)
    
    agent, driver = setup_agent()
    
    try:
        query = "What is wargaming?"
        response = run_agent(agent, query, verbose=True)
        
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"RESPONSE:\n{response}")
        print(f"{'='*80}\n")
        
        assert len(response) > 0, "Response should not be empty"
        logger.info("✅ Test 1 PASSED")
        
    finally:
        driver.close()


def test_agent_hybrid_retrieval():
    """Test: Hybrid Retrieval mit komplexerer Query."""
    print("\n" + "="*80)
    print("TEST 2: Hybrid Retrieval")
    print("="*80)
    
    agent, driver = setup_agent()
    
    try:
        query = "What are the key components and best practices for designing wargames?"
        response = run_agent(agent, query, verbose=True)
        
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"RESPONSE:\n{response}")
        print(f"{'='*80}\n")
        
        assert len(response) > 0, "Response should not be empty"
        logger.info("✅ Test 2 PASSED")
        
    finally:
        driver.close()


def test_agent_cypher():
    """Test: Cypher Query für strukturierte Daten."""
    print("\n" + "="*80)
    print("TEST 3: Cypher Query")
    print("="*80)
    
    agent, driver = setup_agent()
    
    try:
        query = "List the top 5 entities with the most connections in the graph"
        response = run_agent(agent, query, verbose=True)
        
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"RESPONSE:\n{response}")
        print(f"{'='*80}\n")
        
        assert len(response) > 0, "Response should not be empty"
        logger.info("✅ Test 3 PASSED")
        
    finally:
        driver.close()


def test_agent_safety():
    """Test: Safety Check für WRITE operations."""
    print("\n" + "="*80)
    print("TEST 4: Safety - Block WRITE Operations")
    print("="*80)
    
    agent, driver = setup_agent()
    
    try:
        # Versuche WRITE Operation (sollte geblockt werden)
        query = "Execute: CREATE (n:Test {name: 'should_not_exist'}) RETURN n"
        response = run_agent(agent, query, verbose=True)
        
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"RESPONSE:\n{response}")
        print(f"{'='*80}\n")
        
        # Response sollte Error enthalten
        assert "ERROR" in response or "not allowed" in response.lower(), \
            "WRITE operation should be blocked"
        
        logger.info("✅ Test 4 PASSED - WRITE operations correctly blocked")
        
    finally:
        driver.close()


def main():
    """Main test runner with CLI args support."""
    parser = argparse.ArgumentParser(description="Test GraphRAG Agent")
    parser.add_argument(
        "--mode",
        choices=["basic", "hybrid", "cypher", "safety", "all"],
        default="all",
        help="Test mode to run"
    )
    
    args = parser.parse_args()
    
    test_map = {
        "basic": test_agent_basic,
        "hybrid": test_agent_hybrid_retrieval,
        "cypher": test_agent_cypher,
        "safety": test_agent_safety,
    }
    
    try:
        if args.mode == "all":
            # Run all tests
            for test_name, test_func in test_map.items():
                logger.info(f"Running {test_name} test...")
                test_func()
        else:
            # Run specific test
            test_func = test_map[args.mode]
            logger.info(f"Running {args.mode} test...")
            test_func()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()