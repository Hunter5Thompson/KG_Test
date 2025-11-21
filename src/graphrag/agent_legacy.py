"""
Legacy Agent Module - Backward Compatibility Shim
==================================================

This module provides backward compatibility for code that imports from
the old monolithic agent.py file.

DEPRECATED: Use `from src.graphrag import create_graphrag_agent` instead.
"""

import warnings

# Re-export from new modular structure
from .agent.core import GraphRAGAgent, run_agent, stream_agent
from .agent.memory import ConversationMemory
from .agent.state import AgentState
from .tools.executor import ToolExecutor as GraphRAGToolExecutor  # Old name
from .utils.parsing import ToolCall, parse_tool_calls
from .utils.llm_wrapper import OllamaLLMWrapper
from . import create_graphrag_agent


# Emit deprecation warning
warnings.warn(
    "Importing from src.graphrag.agent is deprecated. "
    "Use 'from src.graphrag import create_graphrag_agent' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "GraphRAGAgent",
    "ConversationMemory",
    "AgentState",
    "GraphRAGToolExecutor",
    "ToolCall",
    "parse_tool_calls",
    "OllamaLLMWrapper",
    "create_graphrag_agent",
    "run_agent",
    "stream_agent",
]
