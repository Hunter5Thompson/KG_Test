# File: src/graphrag/authenticated_ollama_llm.py
"""
Authenticated Ollama LLM Integration
HTTP-basiert mit Authorization-Header Support
- Robustes Error-Handling für leere Chat-Antworten (message.content == "")
- Klare Fallback-Reihenfolge der Endpoints (/api/chat -> /v1/completions -> /api/generate)
"""

from __future__ import annotations

from typing import Any, Optional, Dict, List
import httpx
import logging

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.bridge.pydantic import PrivateAttr

# Reduce httpx logging noise (hide 404 fallback attempts)
logging.getLogger("httpx").setLevel(logging.WARNING)



class AuthenticatedOllamaLLM(CustomLLM):
    """
    Custom Ollama LLM with authentication support.

    Verwendet direkte HTTP-Calls mit Authorization-Header,
    analog zur OllamaEmbedding-Implementierung.

    Args:
        model_name: Ollama model name (e.g., "llama3.2:latest")
        base_url: Ollama API base URL
        api_key: API authentication key
        request_timeout: Request timeout in seconds
        max_tokens: Maximum tokens for completion
        temperature: Sampling temperature
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
    _preferred_endpoint: Optional[str] = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        request_timeout: float = 120.0,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        preferred_endpoint: Optional[str] = None,  # "chat", "completions", or "generate"
        **kwargs: Any,
    ):
        # Call parent with public fields
        super().__init__(model_name=model_name, **kwargs)

        # Set private attributes
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._request_timeout = request_timeout
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._preferred_endpoint = preferred_endpoint

        # HTTP Client mit Authentication
        self._http_client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=request_timeout,
        )

        print(f"✅ Authenticated Ollama LLM initialized: {model_name}")

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata"""
        return LLMMetadata(
            context_window=4096,  # ggf. an Modell anpassen
            num_output=self._max_tokens,
            model_name=self.model_name,
        )

    # ---------------------------
    # Helpers
    # ---------------------------

    def _extract_from_openai_completions(self, data: Dict[str, Any]) -> Optional[str]:
        # OpenAI-compatible /v1/completions
        # {"choices":[{"text":"..."}], ...}
        try:
            choices = data.get("choices") or []
            if not choices:
                return None
            text = choices[0].get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            return None
        except Exception:
            return None

    def _extract_from_ollama_generate(self, data: Dict[str, Any]) -> Optional[str]:
        # Ollama /api/generate
        # {"response": "...", ...}
        try:
            resp = data.get("response")
            if isinstance(resp, str) and resp.strip():
                return resp.strip()
            return None
        except Exception:
            return None

    def _extract_from_ollama_chat(self, data: Dict[str, Any]) -> Optional[str]:
        # Ollama /api/chat
        # {"message": {"role":"assistant","content":"..."}, ...}
        try:
            msg = data.get("message")
            if isinstance(msg, dict):
                # Nur akzeptierte Schlüssel (kein thinking etc.)
                for key in ("content", "text", "message"):
                    val = msg.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                # explizit KEIN heuristisches „irgendein String aus dem Dict“
                return None
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
            return None
        except Exception:
            return None

    def _endpoint_payload(self, endpoint: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        if endpoint.endswith("/api/chat"):
            return {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self._temperature),
                    "num_predict": kwargs.get("max_tokens", self._max_tokens),
                },
            }
        if endpoint.endswith("/v1/completions"):
            return {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", self._max_tokens),
                "temperature": kwargs.get("temperature", self._temperature),
            }
        # /api/generate
        return {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self._temperature),
                "num_predict": kwargs.get("max_tokens", self._max_tokens),
            },
        }

    def _extract_text_for_endpoint(self, endpoint: str, data: Dict[str, Any]) -> Optional[str]:
        if endpoint.endswith("/api/chat"):
            return self._extract_from_ollama_chat(data)
        if endpoint.endswith("/v1/completions"):
            return self._extract_from_openai_completions(data)
        return self._extract_from_ollama_generate(data)

    # ---------------------------
    # Completion
    # ---------------------------

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Synchronous completion mit robustem Fehlerhandling:
        - /api/chat → /v1/completions → /api/generate
        - Leere Chat-Responses werden als Fehler gewertet
        - Klare Fehlermeldung mit Preview und Keys
        """
        # Build endpoint list based on preference
        if self._preferred_endpoint == "chat":
            endpoints = [f"{self._base_url}/api/chat"]
        elif self._preferred_endpoint == "completions":
            endpoints = [f"{self._base_url}/v1/completions"]
        elif self._preferred_endpoint == "generate":
            endpoints = [f"{self._base_url}/api/generate"]
        else:
            # Default: try all endpoints in order
            endpoints = [
                f"{self._base_url}/api/chat",  # bevorzugt (User bestätigt)
                f"{self._base_url}/v1/completions",
                f"{self._base_url}/api/generate",
            ]

        last_error: Optional[Exception] = None

        for endpoint in endpoints:
            try:
                payload = self._endpoint_payload(endpoint, prompt, **kwargs)
                resp = self._http_client.post(endpoint, json=payload)

                # Bei 404 einfach weiter zum nächsten Endpoint
                if resp.status_code == 404:
                    continue

                # Andere Fehler ernst nehmen
                resp.raise_for_status()
                data = resp.json()

                text = self._extract_text_for_endpoint(endpoint, data)

                if text is not None and text.strip():
                    # Only print on first success (avoid log spam)
                    if not hasattr(self, '_endpoint_logged'):
                        print(f"✅ Using endpoint: {endpoint}")
                        self._endpoint_logged = True
                    return CompletionResponse(text=text.strip())

                # Wenn kein Text extrahiert werden konnte → sauberer Fehler
                msg_preview = data.get("message")
                if isinstance(msg_preview, dict):
                    # kompakter Preview
                    msg_preview = {k: v for k, v in msg_preview.items() if k in ("role", "content", "text")}
                raise ValueError(
                    "Could not extract text from response. "
                    f"Endpoint={endpoint} Keys={list(data.keys())} "
                    f"Message preview={repr(msg_preview)[:200]}"
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response is not None and e.response.status_code in (404, 422):
                    # 404/422 → nächster Endpoint
                    continue
                if e.response is not None and e.response.status_code == 403:
                    raise ValueError(
                        f"Authentication failed (403): Check OLLAMA_API_KEY. "
                        f"Response: {e.response.text}"
                    ) from e
                # Andere HTTP-Fehler
                raise ValueError(f"HTTP error {e.response.status_code if e.response else 'N/A'}: {str(e)}") from e
            except Exception as e:
                # Netzwerk-/JSON-/Extraktionsfehler → nächsten Endpoint versuchen
                last_error = e
                continue

        # Alle Endpunkte gescheitert
        raise ValueError(
            f"All endpoints failed. Tried: {', '.join(endpoints)}. "
            f"Last error: {str(last_error)}"
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        Streaming completion (optional)
        Fallback auf non-streaming.
        """
        response = self.complete(prompt, **kwargs)
        yield response

    def invoke(self, input, **kwargs):
        """LangChain invoke() support"""
        from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
        
        if isinstance(input, str):
            response = self.complete(input, **kwargs)
            return AIMessage(content=response.text)
        
        elif isinstance(input, list) and all(isinstance(m, BaseMessage) for m in input):
            prompt_parts = []
            
            for msg in input:
                if isinstance(msg, SystemMessage):
                    prompt_parts.append(f"System: {msg.content}\n\n")
                elif isinstance(msg, HumanMessage):
                    prompt_parts.append(f"User: {msg.content}\n\n")
                elif isinstance(msg, AIMessage):
                    prompt_parts.append(f"Assistant: {msg.content}\n\n")
                else:
                    prompt_parts.append(f"{msg.content}\n\n")
            
            prompt = "".join(prompt_parts) + "Assistant:"
            response = self.complete(prompt, **kwargs)
            return AIMessage(content=response.text)
        
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    
    def __del__(self):
        """Cleanup HTTP client"""
        if hasattr(self, '_http_client'):
            self._http_client.close()
