#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen3_tool_forcing_FIXED.py

KORREKTE Implementierung für Qwen3 Tool Calling mit Ollama.

WICHTIGE FIXES:
1. Ollama /api/chat hat KEIN "tools" oder "tool_choice" Parameter
2. Qwen3 nutzt PROMPT-BASED Tool Calling (im System Prompt definieren)
3. Tool Results müssen als USER Messages zurückgegeben werden
4. Ollama kennt nur roles: user, assistant, system

Abhängigkeiten:
    pip install requests python-dotenv

Usage:
    python qwen3_tool_forcing_FIXED.py \
        --model qwen3:32b \
        --queries "What is wargaming?" "Tell me about NATO"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv

logger = logging.getLogger("qwen3_tool_fixed")

DEFAULT_QUERIES: Sequence[str] = (
    "What is wargaming?",
    "Tell me about NATO",
    "What are the components of a wargame?",
)


# ──────────────────────────────────────────────────────────────────────────────
# KORREKTER SYSTEM PROMPT für Qwen3 Tool Calling
# ──────────────────────────────────────────────────────────────────────────────

def build_tool_system_prompt() -> str:
    """
    Qwen3 erwartet Tool-Definitionen IM SYSTEM PROMPT.
    Format: XML-ähnliche Tags mit JSON-Schema.
    """
    return """You are an assistant with access to tools. When you need information, call a tool using this format:

<tool_call>
{"name": "search_knowledge_graph", "arguments": {"query": "your search query"}}
</tool_call>

Available tools:

1. search_knowledge_graph
   Description: Search the knowledge graph for information about a topic.
   Parameters:
     - query (string, required): Topic or entity to look up

RULES:
- When user asks about facts, entities, or knowledge: ALWAYS call the tool first
- Format tool calls EXACTLY as shown above
- After receiving tool results, synthesize a natural answer
- Do NOT make up information - use tool results only"""


# ──────────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────────

class ChatAPIError(RuntimeError):
    """Raised when the /api/chat endpoint returns an error or bad status."""


def post_chat(
    base_url: str,
    api_key: Optional[str],
    payload: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    """POST to /api/chat with JSON payload (non-stream)."""
    url = base_url.rstrip("/") + "/api/chat"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = dict(payload)
    payload.setdefault("stream", False)

    logger.debug("POST %s payload=%s", url, _safe_json(payload))
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise ChatAPIError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    if "error" in data and data["error"]:
        raise ChatAPIError(str(data["error"]))
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


def exec_tool_local(name: str, arguments: Dict[str, Any]) -> str:
    """Lokale Tool-Implementierung (Mock)."""
    if name == "search_knowledge_graph":
        query = arguments.get("query", "")
        payload = {
            "result_type": "mock",
            "query": query,
            "hits": [
                {
                    "entity": "Wargaming",
                    "summary": "Wargaming is a simulation method used for military training and strategic planning.",
                    "score": 0.91
                },
                {
                    "entity": "NATO",
                    "summary": "NATO (North Atlantic Treaty Organization) is a military alliance of 31 member countries.",
                    "score": 0.88
                },
            ],
            "note": "Replace this mock with your real KG call (Neo4j/GraphRAG).",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    raise ValueError(f"Unknown tool: {name}")


# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_tool_calls_from_content(content: str) -> List[ToolCall]:
    """
    Parse Qwen3 tool calls from content:
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    if not isinstance(content, str) or "<tool_call" not in content:
        return []
    
    calls: List[ToolCall] = []
    for m in _TOOL_CALL_BLOCK_RE.finditer(content):
        raw = m.group(1)
        obj = _json_parse_loose(raw)
        if not isinstance(obj, dict):
            continue
        
        name = obj.get("name")
        args = obj.get("arguments") or {}
        
        if isinstance(name, str) and isinstance(args, dict):
            calls.append(ToolCall(name=name, arguments=args))
    
    return calls


def _json_parse_loose(x: Any) -> Dict[str, Any]:
    """Parse JSON from string or return {}."""
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return {}
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        # Try fixing quotes
        fixed = x.replace("'", '"')
        try:
            return json.loads(fixed)
        except Exception:
            logger.warning("Failed to parse JSON: %s", x[:200])
            return {}


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)[:2000]
    except Exception:
        return str(obj)[:2000]


# ──────────────────────────────────────────────────────────────────────────────
# Core flow - KORRIGIERT
# ──────────────────────────────────────────────────────────────────────────────

def run_single_query(
    base_url: str,
    api_key: Optional[str],
    model: str,
    user_query: str,
    temperature: float,
    timeout: int,
    max_iterations: int = 3,
) -> Tuple[bool, str]:
    """
    KORREKTE Implementierung für Qwen3 Tool Calling:
    
    1. System Prompt enthält Tool-Definitionen
    2. Model generiert <tool_call> Tags im Content
    3. Tool Results werden als USER Messages zurückgegeben
    4. Iteriert bis Model fertig ist (kein Tool Call mehr)
    """
    
    system_prompt = build_tool_system_prompt()
    
    # Start conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    
    used_tools = False
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info("Iteration %d/%d", iteration, max_iterations)
        
        # Call Ollama /api/chat (KORREKT - ohne tools/tool_choice)
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": 2000,  # Allow longer responses
            },
        }
        
        t0 = time.time()
        data = post_chat(base_url, api_key, payload, timeout)
        dt = time.time() - t0
        
        msg = data.get("message") or {}
        content = msg.get("content", "")
        
        # Check for thinking field (reasoning models)
        if not content and "thinking" in msg:
            content = msg["thinking"]
        
        logger.info("Response in %.2fs: %s", dt, content[:200])
        
        # Parse tool calls from content
        tool_calls = extract_tool_calls_from_content(content)
        
        if not tool_calls:
            # No more tool calls - model is done
            return used_tools, content
        
        # We found tool calls!
        used_tools = True
        logger.info("Found %d tool call(s)", len(tool_calls))
        
        # Add assistant message to history
        messages.append({"role": "assistant", "content": content})
        
        # Execute tools and add results as USER messages
        for i, call in enumerate(tool_calls, 1):
            try:
                logger.info("Executing tool %d: %s(%s)", i, call.name, call.arguments)
                result = exec_tool_local(call.name, call.arguments)
                
                # KRITISCH: Tool result als USER message (nicht "tool" role!)
                tool_result_msg = f"""Tool result for {call.name}:
```json
{result}
```

Now synthesize a natural answer based on this information."""
                
                messages.append({
                    "role": "user",
                    "content": tool_result_msg
                })
                
                logger.info("Tool executed successfully")
                
            except Exception as exc:
                logger.exception("Tool execution failed: %s", exc)
                error_msg = f"Tool {call.name} failed: {exc}"
                messages.append({
                    "role": "user",
                    "content": f"Error: {error_msg}"
                })
        
        # Continue loop - model will synthesize answer in next iteration
    
    # Max iterations reached
    logger.warning("Max iterations reached without final answer")
    return used_tools, "⚠️ Max iterations reached. Last response: " + content


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FIXED: Qwen3 Tool Calling with Ollama")
    p.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")),
        help="Ollama base URL"
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("OLLAMA_API_KEY"),
        help="Optional API key"
    )
    p.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen3:32b"),
        help="Model name"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "0.0")),
        help="Sampling temperature"
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("TIMEOUT", "120")),
        help="HTTP timeout seconds"
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max tool calling iterations"
    )
    p.add_argument(
        "--queries",
        nargs="*",
        default=list(DEFAULT_QUERIES),
        help="Test queries"
    )
    p.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level"
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    
    logger.info("=" * 80)
    logger.info("FIXED Qwen3 Tool Calling Runner")
    logger.info("=" * 80)
    logger.info("Model: %s", args.model)
    logger.info("Base URL: %s", args.base_url)
    logger.info("Temperature: %.2f", args.temperature)
    logger.info("Max iterations: %d", args.max_iterations)
    logger.info("=" * 80)

    exit_code = 0
    
    for i, q in enumerate(args.queries, 1):
        print("\n" + "=" * 80)
        print(f"TEST {i} | {q}")
        print("=" * 80)
        
        try:
            used, answer = run_single_query(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                user_query=q,
                temperature=args.temperature,
                timeout=args.timeout,
                max_iterations=args.max_iterations,
            )
            
            tag = "✅ USED TOOLS" if used else "ℹ️  DIRECT ANSWER"
            print(f"\n[{tag}]\n{'-'*80}\n{answer}\n")
            
        except ChatAPIError as e:
            logger.error("API error: %s", e)
            print(f"\n[❌ API ERROR]\n{'-'*80}\n{e}\n")
            exit_code = 1
            
        except Exception as e:
            logger.exception("Unhandled error: %s", e)
            print(f"\n[❌ ERROR]\n{'-'*80}\n{e}\n")
            exit_code = 1

    logger.info("=" * 80)
    logger.info("Finished. Exit code: %d", exit_code)
    logger.info("=" * 80)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
