"""
GraphRAG Agent Package
=======================

Prompt-based tool calling agent compatible with Ollama models.
"""

from .agent_core import (
    AgentState,
    ConversationMemory,
    GraphRAGToolExecutor,
    OllamaLLMWrapper,
    ToolCall,
    build_system_prompt,
    call_model_node,
    create_graphrag_agent,
    execute_tools_node,
    parse_tool_calls,
    run_agent,
    should_continue,
    stream_agent,
)

__all__ = [
    "AgentState",
    "ConversationMemory",
    "GraphRAGToolExecutor",
    "OllamaLLMWrapper",
    "ToolCall",
    "build_system_prompt",
    "call_model_node",
    "create_graphrag_agent",
    "execute_tools_node",
    "parse_tool_calls",
    "run_agent",
    "should_continue",
    "stream_agent",
]
