"""
Agent State Definition
=======================

State definition for LangGraph agent.
"""

from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(dict):
    """
    State for the LangGraph Agent

    Fields:
        messages: Conversation history with add_messages reducer
        iteration: Current iteration count
        max_iterations: Maximum allowed iterations before stopping
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
