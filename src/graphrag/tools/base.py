"""
Base Tool Interface for GraphRAG Tools
======================================

Provides abstract base classes and standardized result types for all tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ToolResult:
    """Standardized tool result with success status and metadata"""
    success: bool
    content: str
    metadata: Dict[str, Any]
    error_code: Optional[str] = None


class BaseTool(ABC):
    """
    Abstract base class for all GraphRAG tools.

    Each tool must implement:
    - name: Tool identifier for LLM
    - description: What the tool does (for LLM prompt)
    - parameters: JSON schema for tool parameters
    - execute: Main execution logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for LLM to reference"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of tool's purpose"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        Parameter schema for this tool.

        Format:
        {
            "param_name": {
                "type": "string" | "integer" | "boolean",
                "required": True | False,
                "default": value
            }
        }
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool with given arguments.

        Returns:
            ToolResult with success status, content, and metadata
        """
        pass
