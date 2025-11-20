#graphrag/langchain_ollama_auth.py
"""
LangChain-kompatibler Ollama LLM mit Bearer Token Authentication.
Custom Subclass die Auth Headers direkt in HTTP Requests einfügt.
"""

from typing import Any, Dict, Iterator, Optional, Mapping, List
from langchain_ollama import ChatOllama
import httpx
import json
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class AuthenticatedChatOllama(ChatOllama):
    """
    Custom ChatOllama mit Bearer Token Auth.
    Überschreibt _create_chat_stream um Auth Header hinzuzufügen.
    """
    
    api_key: str = ""
    
    def __init__(self, api_key: str = "", **kwargs):
        """Initialize with API key."""
        super().__init__(**kwargs)
        self.api_key = api_key
    
    def _create_chat_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any
    ) -> Iterator[dict]:
        """
        Override to inject Auth headers.
        Makes direct HTTP calls with proper authentication.
        """
        # Convert LangChain messages to Ollama format
        ollama_messages = []
        for msg in messages:
            role = "user" if msg.type == "human" else "assistant"
            ollama_messages.append({
                "role": role,
                "content": msg.content
            })
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
            }
        }
        
        # Add tools if bound
        if hasattr(self, '_tools') and self._tools:
            tools = []
            for tool in self._tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args_schema.schema() if tool.args_schema else {}
                    }
                })
            payload["tools"] = tools
        
        # Make authenticated request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Ensure base_url has no trailing slash
        base_url = self.base_url.rstrip('/')
        url = f"{base_url}/api/chat"
        
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream('POST', url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                continue
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Ollama API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise ValueError(f"Failed to connect to Ollama: {e}")


def create_authenticated_ollama_llm(
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    **kwargs
) -> AuthenticatedChatOllama:
    """
    Erstellt authentifizierten ChatOllama LLM.
    
    Args:
        model_name: Ollama Model (z.B. "granite4:latest")
        base_url: Ollama Server URL (z.B. "http://host.com" ohne /api/chat)
        api_key: Bearer Token
        temperature: Temperature (default: 0.0)
        **kwargs: Zusätzliche ChatOllama Parameter
        
    Returns:
        AuthenticatedChatOllama instance
        
    Example:
        >>> llm = create_authenticated_ollama_llm(
        ...     model_name="granite4:latest",
        ...     base_url="http://test.ki-plattform.apps.gisamgmt.global",
        ...     api_key="your_token"
        ... )
        >>> response = llm.invoke("Hello!")
    """
    # Clean base_url (remove trailing slash and /api/chat if present)
    base_url = base_url.rstrip('/')
    if base_url.endswith('/api/chat'):
        base_url = base_url[:-9]  # Remove /api/chat
    
    llm = AuthenticatedChatOllama(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        **kwargs
    )
    
    return llm


# Convenience alias
def get_ollama_llm(
    model_name: str,
    base_url: str,
    api_key: str,
    **kwargs
) -> AuthenticatedChatOllama:
    """Alias für create_authenticated_ollama_llm."""
    return create_authenticated_ollama_llm(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )


__all__ = [
    "AuthenticatedChatOllama",
    "create_authenticated_ollama_llm",
    "get_ollama_llm"
]


if __name__ == "__main__":
    """Test authenticated LLM."""
    from config.settings import config
    
    print("="*60)
    print("Testing Authenticated Ollama LLM")
    print("="*60)
    
    print(f"\nConfig:")
    print(f"  Host: {config.ollama.host}")
    print(f"  Model: {config.ollama.llm_model}")
    print(f"  API Key: {config.ollama.api_key[:20]}...")
    
    print("\nCreating authenticated LLM...")
    llm = create_authenticated_ollama_llm(
        model_name=config.ollama.llm_model,
        base_url=config.ollama.host,
        api_key=config.ollama.api_key
    )
    
    print(f"✅ LLM created: {llm.model}")
    
    # Test 1: Simple completion
    print("\n--- Test 1: Simple Completion ---")
    try:
        response = llm.invoke("Say 'Hello World' in exactly 2 words.")
        print(f"✅ Response: {response.content}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import sys
        sys.exit(1)
    
    # Test 2: Tool calling
    from langchain_core.tools import tool
    
    @tool
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y
    
    print("\n--- Test 2: Tool Calling ---")
    llm_with_tools = llm.bind_tools([multiply])
    
    try:
        response = llm_with_tools.invoke("What is 7 times 8? Use the multiply tool.")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"✅ Tool Calling supported!")
            print(f"   Tool calls: {[tc['name'] for tc in response.tool_calls]}")
        else:
            print(f"⚠️  No tool calls detected")
            print(f"   Response: {response.content}")
            print(f"\n⚠️  WARNING: {config.ollama.llm_model} may not support tool calling")
            print(f"   Consider: ollama pull llama3.1:8b-instruct-q8_0")
    except Exception as e:
        print(f"⚠️  Tool calling test failed: {e}")
        print(f"   Model may not support function calling")
    
    print("\n" + "="*60)
    print("✅ Tests completed")
    print("="*60)