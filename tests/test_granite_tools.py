"""
Test granite4's Tool Calling mit ECHTEN Agent Tools.
Nicht mit dummy 'multiply' - sondern mit semantic_search, etc.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # kg_test/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.graphrag.langchain_ollama_auth import create_authenticated_ollama_llm
from langchain_core.tools import tool
from config.settings import config

print("="*80)
print("Testing granite4 Tool Calling with REAL Agent Tools")
print("="*80)

# Create LLM
print("\nCreating granite4 LLM...")
llm = create_authenticated_ollama_llm(
    model_name=config.ollama.llm_model,
    base_url=config.ollama.host,
    api_key=config.ollama.api_key,
    temperature=0.0
)
print(f"✅ LLM: {llm.model}")


# ==================== REALISTIC TOOLS ====================
@tool
def semantic_search(query: str, top_k: int = 5) -> str:
    """
    Führt eine semantische Suche im Knowledge Graph durch.
    Nutze dieses Tool wenn du Informationen über Entities brauchst.
    """
    return f"[MOCK] Semantic search results for: {query}"


@tool
def hybrid_retrieve(query: str, top_k: int = 5) -> str:
    """
    Führt Hybrid-Retrieval durch (Vector + Keyword + Graph).
    Beste Option für komplexe Queries die mehrere Quellen benötigen.
    """
    return f"[MOCK] Hybrid retrieval results for: {query}"


@tool
def cypher_query(query: str) -> str:
    """
    Führt eine Cypher Query auf dem Knowledge Graph aus.
    Nutze für strukturierte Abfragen wie 'Liste alle X' oder 'Zähle Y'.
    """
    return f"[MOCK] Cypher query result: {query}"


# Bind ECHTE Tools
print("\nBinding real agent tools...")
llm_with_tools = llm.bind_tools([semantic_search, hybrid_retrieve, cypher_query])


# ==================== TESTS ====================
test_queries = [
    # Test 1: Sollte hybrid_retrieve oder semantic_search triggern
    "What is wargaming and what are its key components?",
    
    # Test 2: Sollte cypher_query triggern (strukturierte Abfrage)
    "List all entities in the knowledge graph",
    
    # Test 3: Sollte semantic_search triggern (einfache Suche)
    "Find information about NATO",
    
    # Test 4: Sollte hybrid_retrieve triggern (komplexe Multi-Source Query)
    "What are the best practices for wargame design according to multiple sources?",
]


for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}: {query}")
    print('='*80)
    
    try:
        response = llm_with_tools.invoke(query)
        
        # Check for tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"✅ TOOL CALLS DETECTED!")
            for tc in response.tool_calls:
                print(f"   - Tool: {tc['name']}")
                print(f"     Args: {tc['args']}")
        else:
            print(f"❌ NO TOOL CALLS")
            print(f"   Direct Response: {response.content[:200]}...")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")


# ==================== DIAGNOSE ====================
print("\n" + "="*80)
print("DIAGNOSE")
print("="*80)

print("\nMögliche Gründe für fehlende Tool Calls:")
print("1. granite4 braucht explizitere System Prompts")
print("2. granite4's Tool Format ist anders als LangChain Standard")
print("3. Ollama API gibt Tools nicht korrekt weiter")
print("4. granite4 denkt es kann Queries direkt beantworten")

print("\nNächste Schritte:")
print("- Falls KEINE Tool Calls: System Prompt im Agent anpassen")
print("- Falls Tool Calls funktionieren: Agent sollte funktionieren")