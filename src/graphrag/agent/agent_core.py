"""
GraphRAG Agent mit Prompt-Based Tool Calling (Ollama-kompatibel)
================================================================

WICHTIG: Ollama hat KEIN natives Function Calling!
Stattdessen: Tool-Definitionen im System Prompt + XML-Tag Parsing

Funktioniert mit: qwen3, llama3.2, mistral, granite4 (alle Ollama Models)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Annotated, Callable, Dict, Iterable, List, Literal, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from neo4j import Driver
from neo4j.exceptions import CypherSyntaxError, Neo4jError
from langchain_core.language_models.llms import LLM

logger = logging.getLogger(__name__)

class OllamaLLMWrapper(LLM):
    """Wrapper für AuthenticatedOllamaLLM → LangChain LLM"""
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """LangChain's _call Methode"""
        response = self.llm.complete(prompt)
        return response.text
    
    def _llm_type(self) -> str:
        return "ollama_wrapper"
    
    # Für LangChain Messages Support
    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Convert messages to prompt and call LLM"""
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


# ==================== STATE DEFINITION ====================
class AgentState(dict):
    """
    State für den LangGraph Agent
    
    Fields:
        messages: Conversation history
        iteration: Current iteration count
        max_iterations: Maximum allowed iterations
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
    

class ConversationMemory:
    """Simple buffer to retain a limited number of messages across turns."""

    def __init__(self, max_messages: int = 12):
        self.max_messages = max_messages
        self.history: List[BaseMessage] = []

    def append(self, messages: Iterable[BaseMessage]) -> None:
        self.history.extend(messages)
        # keep most recent messages within limit
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages :]

    def snapshot(self) -> List[BaseMessage]:
        return list(self.history)

# ==================== TOOL EXECUTION ====================
@dataclass
class ToolCall:
    """Parsed Tool Call"""
    name: str
    arguments: Dict


class GraphRAGToolExecutor:
    """Executes GraphRAG Tools with telemetry and structured error codes."""

    ERR_UNKNOWN_TOOL = "ERR_UNKNOWN_TOOL"
    ERR_TOOL_EXCEPTION = "ERR_TOOL_EXCEPTION"
    ERR_CYPHER_GUARDRAIL = "ERR_CYPHER_GUARDRAIL"
    ERR_CYPHER_RUNTIME = "ERR_CYPHER_RUNTIME"
    ERR_EMBEDDING = "ERR_EMBEDDING"

    def __init__(self, driver: Driver, embed_fn: Callable[[str], List[float]]):
        self.driver = driver
        self.embed_fn = embed_fn
        self.telemetry: Dict[str, List[float]] = {}

    def _record_latency(self, tool_name: str, duration: float) -> None:
        self.telemetry.setdefault(tool_name, []).append(duration)

    def telemetry_summary(self) -> Dict[str, Dict[str, float]]:
        """Return histogram-friendly latency summary (p50/p95/max)."""
        summary: Dict[str, Dict[str, float]] = {}
        for tool, samples in self.telemetry.items():
            if not samples:
                continue
            sorted_samples = sorted(samples)
            n = len(sorted_samples)
            p50 = sorted_samples[int(0.5 * (n - 1))]
            p95 = sorted_samples[int(0.95 * (n - 1))]
            summary[tool] = {"p50": p50, "p95": p95, "max": max(sorted_samples)}
        return summary

    def execute(self, tool_name: str, arguments: Dict) -> str:
        """Execute tool and return result as string"""
        start = time.perf_counter()
        try:
            if tool_name == "semantic_search":
                result = self._semantic_search(arguments)
            elif tool_name == "hybrid_retrieve":
                result = self._hybrid_retrieve(arguments)
            elif tool_name == "cypher_query":
                result = self._cypher_query(arguments)
            elif tool_name == "schema_overview":
                result = self._schema_overview()
            else:
                return f"[{self.ERR_UNKNOWN_TOOL}] Unknown tool: {tool_name}"
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            result = f"[{self.ERR_TOOL_EXCEPTION}] Tool error: {str(e)}"
        finally:
            self._record_latency(tool_name, time.perf_counter() - start)

        return result
    
    def _semantic_search(self, args: Dict) -> str:
        """Semantic Vector Search"""
        from src.graphrag.hybrid_retriever import HybridGraphRetriever, EmbeddingReranker
        
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        
        retriever = HybridGraphRetriever(
            self.driver,
            self.embed_fn,
            reranker=EmbeddingReranker(self.embed_fn),
        )
        results = retriever.retrieve(query, strategy="vector", top_k=top_k)
        
        if not results:
            return f"No results found for query: {query}"
        
        output = [f"SEMANTIC SEARCH - Found {len(results)} entities:\n"]
        for i, res in enumerate(results, 1):
            output.append(
                f"{i}. {res.entity_id} (score: {res.score:.3f})\n"
                f"   {res.context[:150]}...\n"
            )
        
        return "".join(output)
    
    def _hybrid_retrieve(self, args: Dict) -> str:
        """Hybrid Retrieval (Vector + Graph + Keyword)"""
        from src.graphrag.hybrid_retriever import HybridGraphRetriever, EmbeddingReranker
        
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        expand_hops = args.get("expand_hops", 1)
        
        retriever = HybridGraphRetriever(
            self.driver,
            self.embed_fn,
            reranker=EmbeddingReranker(self.embed_fn),
        )
        results = retriever.retrieve(
            query, 
            strategy="hybrid", 
            top_k=top_k, 
            expand_hops=expand_hops
        )
        
        if not results:
            return f"No results found for query: {query}"
        
        # Build output with context
        output = [f"HYBRID SEARCH - Found {len(results)} entities:\n"]
        
        for i, res in enumerate(results, 1):
            sources = res.metadata.get("sources", [res.source])
            sources_str = ", ".join(sources) if isinstance(sources, list) else str(sources)
            
            output.append(
                f"{i}. {res.entity_id} (score: {res.score:.3f}, sources: {sources_str})\n"
                f"   {res.context[:150]}...\n"
            )
        
        # Add relationship context
        context = retriever.get_context_for_entities(results, include_neighbors=True)
        output.append("\n--- RELATIONSHIPS & CONTEXT ---\n")
        output.append(context[:1500])
        
        return "".join(output)
    
    def _cypher_query(self, args: Dict) -> str:
        """Execute Cypher Query"""
        description = args.get("description", "")
        cypher = args.get("cypher", "")
        
        # Safety check
        query_upper = cypher.upper().strip()
        dangerous = ["CREATE", "DELETE", "SET", "REMOVE", "MERGE", "DROP", "ALTER", "DETACH"]
        
        if any(kw in query_upper for kw in dangerous):
            return f"[{self.ERR_CYPHER_GUARDRAIL}] WRITE operations not allowed. Use READ-ONLY queries."
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher, timeout=10.0)
                records = list(result)
                
                if not records:
                    return f"Query returned no results.\nDescription: {description}\nQuery: {cypher}"
                
                output = [f"CYPHER RESULTS ({len(records)} records):\n"]
                output.append(f"Description: {description}\n\n")
                
                for i, record in enumerate(records[:20], 1):
                    output.append(f"{i}. {dict(record)}\n")
                
                if len(records) > 20:
                    output.append(f"\n... ({len(records) - 20} more rows omitted)")
                
                return "".join(output)
        
        except CypherSyntaxError as e:
            return f"[{self.ERR_CYPHER_RUNTIME}] Cypher syntax error: {str(e)}\nQuery: {cypher}"
        except Neo4jError as e:
            return f"[{self.ERR_CYPHER_RUNTIME}] Cypher execution error: {str(e)}\nQuery: {cypher}"

    def _schema_overview(self) -> str:
        """Provide a compact schema snapshot for the agent."""
        with self.driver.session() as session:
            try:
                labels = session.run("CALL db.labels() YIELD label RETURN label").value()
                rels = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                ).value()
                props = session.run(
                    "CALL db.schema.nodeTypeProperties()"
                    " YIELD nodeLabels, propertyName RETURN nodeLabels, propertyName"
                ).data()
            except Neo4jError as e:
                return f"[{self.ERR_CYPHER_RUNTIME}] Failed to fetch schema: {str(e)}"

        lines = ["Schema Overview:"]
        lines.append(f"• Labels: {', '.join(sorted(set(labels or [])))}")
        lines.append(f"• Relationship Types: {', '.join(sorted(set(rels or [])))}")

        prop_lines = []
        for entry in props:
            labels_list = entry.get("nodeLabels") or []
            pname = entry.get("propertyName")
            if labels_list and pname:
                prop_lines.append(f"  - {','.join(labels_list)}: {pname}")

        if prop_lines:
            lines.append("• Node Properties:\n" + "\n".join(sorted(set(prop_lines))))

        return "\n".join(lines)


# ==================== TOOL CALL PARSING ====================
_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE
)


def parse_tool_calls(content: str) -> List[ToolCall]:
    """
    Extract tool calls from LLM output
    
    Format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
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


# ==================== SYSTEM PROMPT ====================
def build_system_prompt() -> str:
    """
    System Prompt for Qwen3 on Ollama /api/chat with prompt-based tool calling.
    Enforces ONE TOOL PER RESPONSE, strict property/schema limits,
    and a stepwise multi-hop workflow. Compatible with <tool_call> tags.
    """
    return """You are an expert GraphRAG assistant analyzing a knowledge graph about military wargaming.You MUST use tools to answer questions.

⚠️ CRITICAL RULE: NEVER answer based on your own knowledge!
⚠️ You MUST ALWAYS call up a tool!

Before you generate Cypher, request the schema via `schema_overview` to stay aligned with labels/properties unless you already fetched it during this conversation.

You have access to 4 tools. To use a tool, output EXACTLY:

<tool_call>
{"name": "TOOL_NAME", "arguments": {"arg1": "value1", "arg2": "value2"}}
</tool_call>

Do NOT add explanations inside the <tool_call> block.

=== CRITICAL EXECUTION MODEL (READ CAREFULLY) ===
• ONE TOOL PER RESPONSE.
• After you emit a tool call, STOP and WAIT for the tool result.
• Then analyze the result and decide the next step.
• Never answer from memory; synthesize ONLY from tool outputs.

=== AVAILABLE TOOLS ===
1) schema_overview
   Description: Retrieve labels, relationship types, and known node properties to inform Cypher planning.
   Args: none
   Good for: understanding the current graph schema.

2) semantic_search
   Description: Find entities by semantic/vector similarity (meaning-based).
   Args:
     - query (string, required)
     - top_k (int, optional, default=5)
   Good for: definitions, “similar to Y”, exploratory lookups.
   Example:
   <tool_call>
   {"name":"semantic_search","arguments":{"query":"wargaming methodologies","top_k":3}}
   </tool_call>

3) hybrid_retrieve
   Description: Combined search (Vector + Graph + Keyword) — BEST default.
   Args:
     - query (string, required)
     - top_k (int, optional, default=5)
     - expand_hops (int, optional, default=1)
   Good for: comprehensive “Tell me about X”, multi-perspective summaries.
   Example:
   <tool_call>
   {"name":"hybrid_retrieve","arguments":{"query":"NATO wargaming exercises","top_k":5,"expand_hops":2}}
   </tool_call>

4) cypher_query
   Description: Execute a READ-ONLY Cypher query for precise, multi-hop structure.
   Args:
     - description (string, required): plain-English intent
     - cypher (string, required): Cypher using ONLY MATCH/RETURN/WHERE/WITH
   Entity properties available: id, name, title, summary, content
   Not available: chapter, page, etc.
   Example:
   <tool_call>
   {"name":"cypher_query","arguments":{"description":"Find entities mentioning 90%","cypher":"MATCH (n:Entity) WHERE n.content CONTAINS '90%' RETURN n.id, n.content LIMIT 5"}}
   </tool_call>

=== DECISION HEURISTICS ===
• SIMPLE fact or overview → use hybrid_retrieve(query, top_k=5) first.
• STRUCTURED relation or keyword filter → cypher_query with precise WHERE clauses.
• FUZZY definition or similarity → semantic_search.

=== MULTI-HOP STRATEGY (SEQUENTIAL) ===
For complex questions, decompose into steps and call ONE tool per step.

Example complex question:
"Which recommendation addresses the problem causing 90% losses, and which country is most affected?"

Correct approach:
Step 1 (locate the '90% losses' problem):
<tool_call>
{"name":"cypher_query","arguments":{"description":"Locate the 90% losses reference","cypher":"MATCH (n:Entity) WHERE n.content CONTAINS '90%' AND (n.content CONTAINS 'loss' OR n.content CONTAINS 'losses') RETURN n.id, n.content LIMIT 3"}}
</tool_call>

[WAIT FOR RESULT]

Step 2 (find recommendation for the identified problem):
<tool_call>
{"name":"hybrid_retrieve","arguments":{"query":"Chapter 7 recommendation for <problem identified in Step 1>","top_k":5}}
</tool_call>

[WAIT FOR RESULT]

Step 3 (identify most affected country, if needed):
<tool_call>
{"name":"cypher_query","arguments":{"description":"Find most affected country for that problem","cypher":"MATCH (n:Entity) WHERE toLower(n.content) CONTAINS 'linchpin' OR toLower(n.content) CONTAINS 'most affected' RETURN n.id, n.content LIMIT 3"}}
</tool_call>

[WAIT FOR RESULT]  Then synthesize the final answer from tool outputs.

=== STRICT RULES ===
• ONE TOOL PER RESPONSE.
• NEVER call multiple tools in a single message.
• NEVER invent properties (only: id, name, title, summary, content).
• NEVER use placeholder IDs (like P-1234, X, Y).
• If a tool returns no results, clearly say so and adjust the next query.
• If a tool result starts with [ERR_*], explain the issue briefly and adjust your next step/tool choice.
• Final answers MUST be based solely on returned tool content.

=== RESPONSE TEMPLATE (PER TURN) ===
**Analysis:** Briefly state why a tool is needed and which one.
<tool_call>…</tool_call>

(After tool result, in the next turn you will:)
**Result Analysis:** Summarize what the tool returned.
If more info is needed, emit exactly one new <tool_call>. 
If enough info is gathered, provide:
**Final Answer:** Concise synthesis grounded ONLY in tool outputs.
**Sources:** List the relevant entity ids/titles from tool results."""


# ==================== AGENT NODES ====================
def call_model_node(state: AgentState, llm) -> AgentState:
    """
    Call LLM to generate response (possibly with tool calls)
    """
    messages = state["messages"]
    iteration = state.get("iteration", 0)
    
    # Prepend system message if first call
    if iteration == 0:
        system_msg = SystemMessage(content=build_system_prompt())
        messages = [system_msg] + list(messages)
    
        # ⚠️ DEBUG: Print was gesendet wird
        print(f"\n{'='*60}")
        print(f"DEBUG: Sending {len(messages)} messages to LLM")
        print(f"Message 1 (System): {messages[0].content[:200]}...")
        print(f"Message 2 (User): {messages[1].content}")
        print(f"{'='*60}\n")
    
    # Call LLM
    response = llm.invoke(messages)
    
    # ⚠️ DEBUG: Print response
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)
    
    # Extract content
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)
    
    # Return state with new message
    return {
        "messages": [AIMessage(content=content)],
        "iteration": iteration + 1
    }


def execute_tools_node(state: AgentState, tool_executor: GraphRAGToolExecutor) -> AgentState:
    """
    Parse tool calls from last message and execute them
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Extract content
    if hasattr(last_message, "content"):
        content = last_message.content
    else:
        content = str(last_message)
    
    # Parse tool calls
    tool_calls = parse_tool_calls(content)
    
    if not tool_calls:
        logger.warning("execute_tools_node called but no tool calls found!")
        return {"messages": []}
    
    logger.info(f"Executing {len(tool_calls)} tool call(s)")
    
    # Execute each tool and collect results
    new_messages = []
    
    for i, call in enumerate(tool_calls, 1):
        logger.info(f"Tool {i}: {call.name}({call.arguments})")

        result = tool_executor.execute(call.name, call.arguments)

        # detect structured error codes
        guidance = ""
        if isinstance(result, str) and result.startswith("["):
            code = result.split("]", 1)[0].lstrip("[")
            guidance = f"Detected error code {code}. Adjust your next tool choice or query accordingly."

        # Return result as HumanMessage (user role)
        tool_result_content = f"""Tool result for {call.name}:
```
{result}
```
{guidance}
Now synthesize an answer based on this information."""

        new_messages.append(HumanMessage(content=tool_result_content))
    
    return {"messages": new_messages}


def should_continue(state: AgentState) -> Literal["execute_tools", "end"]:
    """
    Decide if we should execute tools or end
    """
    messages = state["messages"]
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    
    # Check max iterations
    if iteration >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return "end"
    
    # Check last message for tool calls
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    if hasattr(last_message, "content"):
        content = last_message.content
    else:
        content = str(last_message)
    
    # Check if content has tool calls
    has_tool_calls = "<tool_call" in content
    
    if has_tool_calls:
        logger.info("Tool calls detected, routing to execute_tools")
        return "execute_tools"
    else:
        logger.info("No tool calls, ending")
        return "end"


# ==================== AGENT CREATION ====================
def create_graphrag_agent(
    llm,
    driver: Driver,
    embed_fn: Callable[[str], List[float]],
    max_iterations: int = 10
):
    """
    Create GraphRAG Agent with Prompt-Based Tool Calling
    
    Args:
        llm: LangChain LLM (any model, no tool calling required!)
        driver: Neo4j Driver
        embed_fn: Embedding function
        max_iterations: Max tool calling iterations
    
    Returns:
        Compiled LangGraph StateGraph
    """
    
    # Tool executor
    tool_executor = GraphRAGToolExecutor(driver, embed_fn)
    
    # Create StateGraph
    workflow = StateGraph(AgentState)
    
    # Add nodes with closures to capture dependencies
    workflow.add_node(
        "call_model",
        lambda state: call_model_node(state, llm)
    )
    
    workflow.add_node(
        "execute_tools",
        lambda state: execute_tools_node(state, tool_executor)
    )
    
    # Set entry point
    workflow.set_entry_point("call_model")
    
    # Conditional edge: model -> tools or end
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "end": END
        }
    )
    
    # Edge: tools -> model (for next iteration)
    workflow.add_edge("execute_tools", "call_model")
    
    # Compile
    app = workflow.compile()

    # Expose executor for telemetry/inspection
    app.tool_executor = tool_executor

    logger.info(f"✅ GraphRAG Agent created (max_iterations={max_iterations})")
    return app


# ==================== CONVENIENCE WRAPPER ====================
def stream_agent(
    app,
    query: str,
    max_iterations: int = 10,
    memory: Optional[ConversationMemory] = None,
    verbose: bool = False,
    stream_handler: Optional[Callable[[Dict], None]] = None,
):
    """Stream agent execution for UI/CLI consumption."""

    base_messages: List[BaseMessage] = memory.snapshot() if memory else []
    base_messages.append(HumanMessage(content=query))

    initial_state = {
        "messages": base_messages,
        "iteration": 0,
        "max_iterations": max_iterations
    }

    tool_calls_count = 0
    final_state = None

    for step_output in app.stream(initial_state):
        final_state = step_output
        if stream_handler:
            stream_handler(step_output)
        if verbose:
            for node_name, node_state in step_output.items():
                print(f"[{node_name}]")

                if "messages" in node_state:
                    last_msg = node_state["messages"][-1]
                    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

                    if "<tool_call" in content:
                        tool_calls = parse_tool_calls(content)
                        tool_calls_count += len(tool_calls)
                        print(f"  Tool Calls: {[tc.name for tc in tool_calls]}")
                    else:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"  {preview}")

                print()

    answer = "No response generated"
    messages = []

    if final_state:
        last_state = list(final_state.values())[-1]
        messages = last_state.get("messages", [])
        if messages:
            final_message = messages[-1]
            answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    if memory is not None:
        memory.append(messages)

    return {
        "answer": answer,
        "tool_calls": tool_calls_count,
        "messages": messages
    }


def run_agent(
    app,
    query: str,
    max_iterations: int = 10,
    memory: Optional[ConversationMemory] = None,
    verbose: bool = True,
    stream_handler: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Convenience wrapper around :func:`stream_agent` with optional streaming callback."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}\n")

    return stream_agent(
        app,
        query,
        max_iterations=max_iterations,
        memory=memory,
        verbose=verbose,
        stream_handler=stream_handler,
    )


# ==================== MAIN (TEST) ====================
if __name__ == "__main__":
    from src.graphrag.langchain_ollama_auth import create_authenticated_ollama_llm  # ← NEU
    from src.embeddings.ollama_embeddings import OllamaEmbedding
    from neo4j import GraphDatabase
    from config.settings import AppConfig
    
    logging.basicConfig(level=logging.INFO)
    
    config = AppConfig.from_env()
    
    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password)
    )
    
    embedder = OllamaEmbedding(
        model_name=config.ollama.embedding_model,
        base_url=config.ollama.host,
        api_key=config.ollama.api_key
    )
    
    
    llm = create_authenticated_ollama_llm(
        model_name=config.ollama.llm_model,
        base_url=config.ollama.host,
        api_key=config.ollama.api_key,
        temperature=0.0,
        max_tokens=3000
    )
    
    # Create Agent
    agent = create_graphrag_agent(
        llm, 
        driver, 
        embedder.get_query_embedding,
        max_iterations=10
    )
    
    # Test Queries
    test_queries = [
        "What is wargaming?",
        "What are the main components of a wargame?",
        "Tell me about NATO's role in wargaming",
    ]
    
    for query in test_queries:
        result = run_agent(agent, query, verbose=True)
        
        print(f"{'='*60}")
        print("FINAL ANSWER:")
        print(f"{'='*60}")
        print(result['answer'])
        print(f"\n📊 Tool Calls: {result['tool_calls']}")
        print("\n\n")
    
    driver.close()