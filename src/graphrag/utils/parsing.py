"""
Tool Call Parsing
==================

Parse tool calls from LLM output.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Parsed tool call from LLM output"""
    name: str
    arguments: Dict


# Regex to extract tool calls from <tool_call> tags
_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE
)


def parse_tool_calls(content: str) -> List[ToolCall]:
    """
    Extract tool calls from LLM output.

    Format expected: <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    Args:
        content: LLM output string

    Returns:
        List of ToolCall objects
    """
    if not isinstance(content, str) or "<tool_call" not in content:
        return []

    calls = []

    for match in _TOOL_CALL_REGEX.finditer(content):
        json_str = match.group(1)

        try:
            obj = json.loads(json_str)

            if isinstance(obj, dict) and "name" in obj:
                name = obj["name"]
                args = obj.get("arguments", {})

                if isinstance(name, str) and isinstance(args, dict):
                    calls.append(ToolCall(name=name, arguments=args))

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call JSON: {json_str[:100]}")

    return calls
