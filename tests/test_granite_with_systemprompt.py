"""
Test Qwen3 mit EXPLIZITEM System Prompt der Tool-Nutzung fordert.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # kg_test/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from src.graphrag.langchain_ollama_auth import create_authenticated_ollama_llm
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import config

print("="*80)
print("Testing granite4 with EXPLICIT System Prompt")
print("="*80)

# Create LLM
llm = create_authenticated_ollama_llm(
    model_name=config.ollama.llm_model,
    base_url=config.ollama.host,
    api_key=config.ollama.api_key,
    temperature=0.0
)

# Define tools
@tool
def search_knowledge_graph(query: str) -> str:
    """Search the knowledge graph for information about a topic."""
    return f"[MOCK RESULT] Found information about: {query}"

# Bind tools
llm_with_tools = llm.bind_tools([search_knowledge_graph])

# System prompt that FORCES tool usage
system_prompt = SystemMessage(content="""You are an assistant with access to a knowledge graph search tool.

CRITICAL: You MUST use the search_knowledge_graph tool to answer ANY question about topics, entities, or facts.

DO NOT answer from your own knowledge. ALWAYS call the tool first, then answer based on the tool's results.

Format:
User asks a question → You call search_knowledge_graph → You answer based on results

Now respond to the user's query.""")

# Test queries
test_cases = [
    "What is wargaming?",
    "Tell me about NATO",
    "What are the components of a wargame?",
]

for i, query in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}: {query}")
    print('='*80)
    
    messages = [
        system_prompt,
        HumanMessage(content=query)
    ]
    
    try:
        response = llm_with_tools.invoke(messages)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"✅ SUCCESS - Tool Called!")
            for tc in response.tool_calls:
                print(f"   Tool: {tc['name']}")
                print(f"   Args: {tc['args']}")
        else:
            print(f"❌ FAILED - No Tool Call")
            print(f"   Direct answer: {response.content[:150]}...")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If tools are called → granite4 works with explicit instructions")
print("If NO tools called → granite4 doesn't support tools in Ollama")