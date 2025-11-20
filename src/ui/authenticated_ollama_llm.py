"""
Authenticated Ollama LLM Integration
HTTP-basiert mit Authorization-Header Support
"""
from typing import Any, Optional
import httpx
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.bridge.pydantic import PrivateAttr


class AuthenticatedOllamaLLM(CustomLLM):
    """
    Custom Ollama LLM with authentication support
    
    Verwendet direkte HTTP-Calls mit Authorization-Header,
    analog zur OllamaEmbedding-Implementierung.
    
    Args:
        model_name: Ollama model name (e.g., "llama3.2:latest")
        base_url: Ollama API base URL
        api_key: API authentication key
        request_timeout: Request timeout in seconds
        max_tokens: Maximum tokens for completion
    
    Example:
        >>> llm = AuthenticatedOllamaLLM(
        ...     model_name="llama3.2:latest",
        ...     base_url="http://test.ki-plattform.apps.gisamgmt.global/",
        ...     api_key="your-key"
        ... )
        >>> response = llm.complete("Hello, how are you?")
        >>> print(response.text)
    """
    
    # Pydantic model fields (public)
    model_name: str
    
    # Private attributes (not validated by Pydantic)
    _base_url: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _request_timeout: float = PrivateAttr()
    _max_tokens: int = PrivateAttr()
    _temperature: float = PrivateAttr()
    _http_client: httpx.Client = PrivateAttr()
    
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        request_timeout: float = 120.0,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs
    ):
        # Call parent with public fields
        super().__init__(model_name=model_name, **kwargs)
        
        # Set private attributes
        self._base_url = base_url.rstrip('/')
        self._api_key = api_key
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens
        self._temperature = temperature
        
        # HTTP Client mit Authentication
        self._http_client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=request_timeout
        )
        
        print(f"✅ Authenticated Ollama LLM initialized: {model_name}")
    
    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata"""
        return LLMMetadata(
            context_window=4096,  # Adjust based on model
            num_output=self._max_tokens,
            model_name=self.model_name,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Synchronous completion
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
        
        Returns:
            CompletionResponse with generated text
        """
        # Try endpoints in priority order based on user feedback
        endpoints = [
            f"{self._base_url}/api/chat",         # Ollama Chat (PRIORITY - user confirmed)
            f"{self._base_url}/v1/completions",   # OpenAI-compatible
            f"{self._base_url}/api/generate",     # Standard Ollama
        ]
        
        last_error = None
        
        for endpoint in endpoints:
            try:
                # Build payload based on endpoint type - ORDER MATTERS!
                if "/api/chat" in endpoint:
                    # Ollama /api/chat requires messages array
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": kwargs.get("temperature", self._temperature),
                            "num_predict": kwargs.get("max_tokens", self._max_tokens),
                            # Disable thinking for reasoning models
                            "num_ctx": 4096,
                        }
                    }
                elif "/v1/completions" in endpoint:
                    # OpenAI-style payload
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": kwargs.get("max_tokens", self._max_tokens),
                        "temperature": kwargs.get("temperature", self._temperature),
                    }
                elif "/api/generate" in endpoint:
                    # Ollama /api/generate style
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": kwargs.get("temperature", self._temperature),
                            "num_predict": kwargs.get("max_tokens", self._max_tokens),
                        }
                    }
                else:
                    # Fallback
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                    }
                
                response = self._http_client.post(endpoint, json=payload)
                
                # If 404, try next endpoint
                if response.status_code == 404:
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Extract response text (different formats)
                text = None
                
                # OpenAI format
                if "choices" in data:
                    text = data["choices"][0]["text"]
                
                # Ollama /api/generate format
                elif "response" in data:
                    text = data["response"]
                
                # Ollama /api/chat format
                elif "message" in data:
                    msg = data["message"]
                    
                    # Handle message structure
                    if isinstance(msg, dict):
                        # Reasoning models may have 'thinking' + 'content'
                        thinking = msg.get("thinking", "")
                        content = msg.get("content", "")
                        
                        # Combine or prefer based on what's available
                        # Check for non-empty strings explicitly
                        if content:  # content exists and is non-empty
                            text = content
                        elif thinking:  # fallback to thinking
                            text = thinking
                        elif "text" in msg:
                            text = msg["text"]
                        elif "message" in msg:
                            text = msg["message"]
                        else:
                            # Fallback: look for any string value
                            for key, value in msg.items():
                                if isinstance(value, str) and len(value) > 0:
                                    text = value
                                    break
                    elif isinstance(msg, str):
                        text = msg
                    else:
                        text = str(msg)
                
                # Validate we got text
                if text and len(text.strip()) > 0:
                    print(f"✅ Using endpoint: {endpoint}")
                    return CompletionResponse(text=text.strip())
                else:
                    # Better error message
                    msg_preview = str(data.get("message", "N/A"))[:200]
                    raise ValueError(
                        f"Could not extract text from response. "
                        f"Keys: {list(data.keys())}. "
                        f"Message preview: {msg_preview}"
                    )
            
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 404:
                    continue  # Try next endpoint
                elif e.response.status_code == 403:
                    raise ValueError(
                        f"Authentication failed (403): Check OLLAMA_API_KEY. "
                        f"Response: {e.response.text}"
                    )
                elif e.response.status_code == 422:
                    # Validation error - likely wrong payload format
                    print(f"⚠️  Endpoint {endpoint} returned 422, trying next...")
                    continue
                else:
                    raise ValueError(f"HTTP error {e.response.status_code}: {e.response.text}")
            
            except Exception as e:
                last_error = e
                continue  # Try next endpoint
        
        # If all endpoints failed
        raise ValueError(
            f"All endpoints failed. Tried: {', '.join(endpoints)}. "
            f"Last error: {str(last_error)}"
        )
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        Streaming completion (not implemented for this POC)
        
        Falls benötigt, kann hier Streaming implementiert werden
        """
        # Fallback to non-streaming
        response = self.complete(prompt, **kwargs)
        yield response
    
    def __del__(self):
        """Cleanup HTTP client"""
        if hasattr(self, '_http_client'):
            self._http_client.close()