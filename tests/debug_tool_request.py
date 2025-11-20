"""
Debug: Was senden wir an Ollama API?
Zeigt exakt das Tool-Format das an granite4 gesendet wird.
"""

from langchain_core.tools import tool
from config.settings import config
import json


# Define tools wie im Agent
@tool
def semantic_search(query: str, top_k: int = 5) -> str:
    """
    Führt eine semantische Suche im Knowledge Graph durch.
    Nutze dieses Tool wenn du Informationen über Entities brauchst.
    """
    return "results"


@tool
def hybrid_retrieve(query: str, top_k: int = 5) -> str:
    """
    Führt Hybrid-Retrieval durch (Vector + Keyword + Graph).
    Beste Option für komplexe Queries.
    """
    return "results"


tools = [semantic_search, hybrid_retrieve]


# Build payload wie in AuthenticatedChatOllama
print("="*80)
print("TOOL PAYLOAD DEBUG")
print("="*80)

# Convert tools to Ollama format (wie in _create_chat_stream)
ollama_tools = []
for tool_obj in tools:
    tool_dict = {
        "type": "function",
        "function": {
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": tool_obj.args_schema.schema() if tool_obj.args_schema else {}
        }
    }
    ollama_tools.append(tool_dict)

print("\n--- OLLAMA TOOL FORMAT (was wir senden) ---")
print(json.dumps(ollama_tools, indent=2))


# Check if this matches Ollama's expected format
print("\n\n--- EXPECTED OLLAMA FORMAT (from docs) ---")
expected_format = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for, e.g. San Francisco, CA"
                    },
                    "format": {
                        "type": "string",
                        "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "format"]
            }
        }
    }
]
print(json.dumps(expected_format, indent=2))


print("\n\n--- COMPARISON ---")
print(f"Our format has 'type': {ollama_tools[0].get('type')}")
print(f"Our format has 'function': {bool(ollama_tools[0].get('function'))}")
print(f"Parameters structure matches: {bool(ollama_tools[0]['function'].get('parameters'))}")


# Test full payload
print("\n\n--- FULL REQUEST PAYLOAD ---")
full_payload = {
    "model": config.ollama.llm_model,
    "messages": [
        {"role": "user", "content": "What is wargaming?"}
    ],
    "stream": True,
    "options": {
        "temperature": 0.0,
    },
    "tools": ollama_tools
}
print(json.dumps(full_payload, indent=2))


print("\n\n--- MANUAL TEST COMMAND ---")
print("Test this payload manually:")
print(f"""
curl -X POST {config.ollama.host}/api/chat \\
  -H "Authorization: Bearer {config.ollama.api_key[:20]}..." \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(full_payload)}'
""")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Check if parameters structure matches Ollama docs")
print("2. Try manual curl command to see raw Ollama response")
print("3. Check if granite4 needs system prompt to use tools")