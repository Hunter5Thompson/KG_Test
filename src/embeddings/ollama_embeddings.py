"embeddings/ollama_embeddings.py"
"""
Ollama Embedding Integration
"""
import httpx
from typing import List, Optional
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr


class OllamaEmbedding(BaseEmbedding):
    """
    Ollama Embedding Model Integration
    
    Supports authentication and automatic dimension detection.
    
    Args:
        model_name: Name of the Ollama embedding model
        base_url: Ollama API base URL
        api_key: API authentication key
        embed_batch_size: Batch size for embeddings
    
    Example:
        >>> embed_model = OllamaEmbedding(
        ...     model_name="qwen3-embedding:4b-q8_0",
        ...     base_url="http://localhost:11434",
        ...     api_key="your-key"
        ... )
        >>> embedding = embed_model.get_text_embedding("Hello world")
    """
    
    # Pydantic model fields
    model_name: str
    _base_url: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _http_client: httpx.Client = PrivateAttr()
    _embed_dim: int = PrivateAttr()
    
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        embed_batch_size: int = 10,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs
        )
        
        self._base_url = base_url.rstrip('/')
        self._api_key = api_key
        self._http_client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        
        self._embed_dim = self._get_embedding_dim()
        print(f"✅ Ollama Embedding initialized: {model_name} (dim={self._embed_dim})")
    
    def _get_embedding_dim(self) -> int:
        """Auto-detect embedding dimension"""
        test_embedding = self._get_embedding_via_http("test")
        if test_embedding and len(test_embedding) > 1:
            return len(test_embedding)
        
        print(f"⚠️  Could not determine embedding dim, using default 2560")
        return 2560
    
    def _get_embedding_via_http(self, text: str) -> Optional[List[float]]:
        """
        Get embedding via HTTP request to /api/embed
        
        Handles nested list response format: {"embeddings": [[...]]}
        """
        endpoint = f"{self._base_url}/api/embed"
        payload = {"model": self.model_name, "input": text}
        
        try:
            response = self._http_client.post(endpoint, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if "embeddings" in data:
                    embeddings = data["embeddings"]
                    
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        # Handle nested list (batch response)
                        if isinstance(embeddings[0], list):
                            return embeddings[0]
                        else:
                            return embeddings
            
            return None
        
        except Exception:
            return None
    
    @classmethod
    def class_name(cls) -> str:
        return "OllamaEmbedding"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query"""
        embedding = self._get_embedding_via_http(query)
        return embedding if embedding else [0.0] * self._embed_dim
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        embedding = self._get_embedding_via_http(text)
        return embedding if embedding else [0.0] * self._embed_dim
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async query embedding"""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async text embedding"""
        return self._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch text embeddings"""
        return [self._get_text_embedding(text) for text in texts]
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, '_http_client'):
            self._http_client.close()