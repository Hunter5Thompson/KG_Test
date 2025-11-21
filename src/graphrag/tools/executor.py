"""
Tool Executor
=============

Orchestrates tool execution with telemetry and error handling.
"""

import logging
import time
from typing import Dict, List

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Orchestrates tool execution with telemetry.

    Features:
    - Dynamic tool registry
    - Performance telemetry (p50, p95, max latency)
    - Standardized error handling
    - Tool description generation for LLM prompts
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initialize ToolExecutor with a list of tools.

        Args:
            tools: List of BaseTool instances
        """
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.telemetry: Dict[str, List[float]] = {}

    def execute(self, tool_name: str, arguments: Dict) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome
        """
        if tool_name not in self.tools:
            logger.warning(f"Unknown tool requested: {tool_name}")
            return ToolResult(
                success=False,
                content=f"Unknown tool: {tool_name}. Available tools: {', '.join(self.tools.keys())}",
                metadata={"tool_name": tool_name},
                error_code="ERR_UNKNOWN_TOOL"
            )

        tool = self.tools[tool_name]

        start = time.perf_counter()
        try:
            result = tool.execute(**arguments)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed with exception: {e}", exc_info=True)
            result = ToolResult(
                success=False,
                content=f"Tool execution failed: {str(e)}",
                metadata={"tool_name": tool_name, "arguments": arguments},
                error_code="ERR_TOOL_EXCEPTION"
            )
        finally:
            duration = time.perf_counter() - start
            self._record_latency(tool_name, duration)

        return result

    def _record_latency(self, tool_name: str, duration: float) -> None:
        """Record tool execution latency for telemetry"""
        self.telemetry.setdefault(tool_name, []).append(duration)

    def get_tool_descriptions(self) -> List[Dict]:
        """
        Get tool descriptions for LLM prompt.

        Returns:
            List of tool descriptions with names, descriptions, and parameters
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]

    def telemetry_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics summary.

        Returns:
            Dictionary with p50, p95, max latencies per tool
        """
        summary = {}
        for tool, samples in self.telemetry.items():
            if samples:
                sorted_samples = sorted(samples)
                n = len(sorted_samples)
                summary[tool] = {
                    "p50": sorted_samples[int(0.5 * (n - 1))],
                    "p95": sorted_samples[int(0.95 * (n - 1))],
                    "max": max(sorted_samples),
                    "count": n
                }
        return summary

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the executor.

        Args:
            tool: BaseTool instance to add
        """
        self.tools[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")

    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the executor.

        Args:
            tool_name: Name of tool to remove
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Removed tool: {tool_name}")

    def clear_telemetry(self) -> None:
        """Clear all telemetry data"""
        self.telemetry.clear()
