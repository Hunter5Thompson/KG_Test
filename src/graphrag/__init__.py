"""
GraphRAG Module
Knowledge Graph Retrieval-Augmented Generation
"""
from .hybrid_retriever import HybridGraphRetriever, RetrievalResult
from .agent.core import GraphRAGAgent, run_agent, stream_agent
from .agent.memory import ConversationMemory
from .tools.executor import ToolExecutor
from .tools.basic_tools import (
    SemanticSearchTool,
    HybridRetrieveTool,
    CypherQueryTool,
    SchemaOverviewTool
)
from .tools.multihop_tools import (
    CausalChainTool,
    PrerequisitesTool,
    InfluenceTool,
    AlternativesTool,
    ProcessSequenceTool,
    CriticalNodesTool
)
from .multihop.analyzer import MultihopAnalyzer


def create_graphrag_agent(llm, driver, embed_fn, max_iterations=10, enable_multihop=True):
    """
    Factory function to create a configured GraphRAG agent.

    Args:
        llm: LangChain-compatible LLM instance
        driver: Neo4j driver instance
        embed_fn: Embedding function (str -> List[float])
        max_iterations: Maximum agent iterations (default: 10)
        enable_multihop: Enable multihop analysis tools (default: True)

    Returns:
        Configured GraphRAG agent with compiled workflow

    Example:
        >>> from src.graphrag import create_graphrag_agent
        >>> from src.graphrag.langchain_ollama_auth import create_authenticated_ollama_llm
        >>> from src.embeddings.ollama_embeddings import OllamaEmbedding
        >>> from neo4j import GraphDatabase
        >>>
        >>> driver = GraphDatabase.driver(uri, auth=(user, password))
        >>> embedder = OllamaEmbedding(model_name="...", base_url="...")
        >>> llm = create_authenticated_ollama_llm(model_name="...", base_url="...")
        >>>
        >>> agent = create_graphrag_agent(llm, driver, embedder.get_query_embedding)
        >>> result = agent.run("What is wargaming?")
        >>> print(result['answer'])
    """
    # Create basic tools
    tools = [
        SemanticSearchTool(driver, embed_fn),
        HybridRetrieveTool(driver, embed_fn),
        CypherQueryTool(driver),
        SchemaOverviewTool(driver),
    ]

    # Add multihop tools if enabled
    if enable_multihop:
        analyzer = MultihopAnalyzer(driver, embed_fn)
        tools.extend([
            CausalChainTool(analyzer),
            PrerequisitesTool(analyzer),
            InfluenceTool(analyzer),
            AlternativesTool(analyzer),
            ProcessSequenceTool(analyzer),
            CriticalNodesTool(analyzer),
        ])

    # Create tool executor
    tool_executor = ToolExecutor(tools)

    # Create and return agent
    agent = GraphRAGAgent(llm, tool_executor, max_iterations)

    return agent


__all__ = [
    # Retrieval
    "HybridGraphRetriever",
    "RetrievalResult",
    # Agent
    "GraphRAGAgent",
    "ConversationMemory",
    "create_graphrag_agent",
    "run_agent",
    "stream_agent",
    # Tools
    "ToolExecutor",
    "SemanticSearchTool",
    "HybridRetrieveTool",
    "CypherQueryTool",
    "SchemaOverviewTool",
    "CausalChainTool",
    "PrerequisitesTool",
    "InfluenceTool",
    "AlternativesTool",
    "ProcessSequenceTool",
    "CriticalNodesTool",
    # Multihop
    "MultihopAnalyzer",
]

__version__ = "0.1.0"