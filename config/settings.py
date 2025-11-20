"config/settings.py"
"""
Configuration management
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class OllamaConfig:
    """Ollama API configuration"""
    host: str
    api_key: str
    llm_model: str
    embedding_model: str
    
    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Load from environment variables"""
        load_dotenv()
        return cls(
            host=os.getenv("OLLAMA_HOST", ""),
            api_key=os.getenv("OLLAMA_API_KEY", ""),
            llm_model=os.getenv("OLLAMA_MODEL", ""),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "")
        )


@dataclass
class Neo4jConfig:
    """Neo4j configuration"""
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    
    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Load from environment variables"""
        load_dotenv()
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )


@dataclass
class AppConfig:
    """Application configuration"""
    ollama: OllamaConfig
    neo4j: Neo4jConfig
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load all config from environment"""
        return cls(
            ollama=OllamaConfig.from_env(),
            neo4j=Neo4jConfig.from_env()
        )
# ==================== CRITICAL: Global Config Instance ====================
# This must be exported for imports like: from config.settings import config
config = AppConfig.from_env()


# Exports
__all__ = ["AppConfig", "OllamaConfig", "Neo4jConfig", "config"]

# ==================== Optional: Config Test ====================
if __name__ == "__main__":
    """Test config loading"""
    print("="*60)
    print("Configuration Test")
    print("="*60)
    
    print("\n--- Ollama Config ---")
    print(f"Host: {config.ollama.host}")
    print(f"LLM Model: {config.ollama.llm_model}")
    print(f"Embedding Model: {config.ollama.embedding_model}")
    print(f"API Key: {config.ollama.api_key[:20]}..." if config.ollama.api_key else "API Key: NOT SET")
    
    print("\n--- Neo4j Config ---")
    print(f"URI: {config.neo4j.uri}")
    print(f"User: {config.neo4j.user}")
    print(f"Database: {config.neo4j.database}")
    print(f"Password: {'*' * len(config.neo4j.password)}" if config.neo4j.password else "Password: NOT SET")
    
    print("\n✅ Config loaded successfully!")