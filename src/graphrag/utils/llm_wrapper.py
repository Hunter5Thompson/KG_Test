"""
LLM Wrapper
===========

LangChain-compatible wrapper for Ollama LLM.
"""

from typing import List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class OllamaLLMWrapper(LLM):
    """
    Wrapper for AuthenticatedOllamaLLM → LangChain LLM.

    Provides LangChain-compatible interface for Ollama models
    without native function calling support.
    """

    def __init__(self, llm):
        """
        Initialize wrapper.

        Args:
            llm: AuthenticatedOllamaLLM instance or compatible LLM
        """
        super().__init__()
        self.llm = llm

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        LangChain's _call method.

        Args:
            prompt: Prompt string
            stop: Optional stop sequences
            **kwargs: Additional arguments

        Returns:
            LLM response text
        """
        response = self.llm.complete(prompt)
        return response.text

    def _llm_type(self) -> str:
        """Return LLM type identifier"""
        return "ollama_wrapper"

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """
        Convert messages to prompt and call LLM.

        Args:
            messages: List of LangChain messages
            **kwargs: Additional arguments

        Returns:
            AIMessage with LLM response
        """
        # Convert messages to single prompt
        prompt_parts = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"SYSTEM: {msg.content}\n")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"USER: {msg.content}\n")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"ASSISTANT: {msg.content}\n")
            else:
                prompt_parts.append(f"{msg.content}\n")

        prompt = "\n".join(prompt_parts)

        # Call underlying LLM
        response = self.llm.complete(prompt)

        return AIMessage(content=response.text)
