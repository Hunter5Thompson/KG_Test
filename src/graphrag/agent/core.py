"""
GraphRAG Agent Core
===================

Main agent orchestration using LangGraph.
"""

import logging
from typing import Dict, Literal, Optional, Callable

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .state import AgentState
from .prompts import PromptBuilder
from ..tools.executor import ToolExecutor
from ..utils.parsing import parse_tool_calls

logger = logging.getLogger(__name__)


class GraphRAGAgent:
    """
    Core agent orchestration with LangGraph.

    The agent:
    1. Calls LLM to generate response (possibly with tool calls)
    2. Executes tools if present
    3. Feeds results back to LLM
    4. Repeats until final answer or max iterations

    Architecture:
    - Modular tool system
    - Dynamic prompt generation
    - Telemetry and error handling
    - Stateful conversation flow
    """

    def __init__(self, llm, tool_executor: ToolExecutor, max_iterations: int = 10):
        """
        Initialize GraphRAG Agent.

        Args:
            llm: LangChain-compatible LLM
            tool_executor: ToolExecutor with registered tools
            max_iterations: Maximum iterations before stopping
        """
        self.llm = llm
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations

        # Build system prompt from available tools
        tool_descriptions = tool_executor.get_tool_descriptions()
        self.system_prompt = PromptBuilder.build_system_prompt(tool_descriptions)

        # Create LangGraph workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

        # Expose executor for telemetry/inspection
        self.app.tool_executor = tool_executor

        logger.info(f"✅ GraphRAG Agent created (max_iterations={max_iterations})")

    def _create_workflow(self) -> StateGraph:
        """
        Create LangGraph workflow.

        Workflow:
        - call_model: Invoke LLM
        - execute_tools: Execute tool calls
        - Conditional routing based on tool presence
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("call_model", self._call_model_node)
        workflow.add_node("execute_tools", self._execute_tools_node)

        # Set entry point
        workflow.set_entry_point("call_model")

        # Conditional edge: model -> tools or end
        workflow.add_conditional_edges(
            "call_model",
            self._should_continue,
            {
                "execute_tools": "execute_tools",
                "end": END
            }
        )

        # Edge: tools -> model (for next iteration)
        workflow.add_edge("execute_tools", "call_model")

        return workflow

    def _call_model_node(self, state: AgentState) -> AgentState:
        """
        Call LLM to generate response.

        Args:
            state: Current agent state

        Returns:
            Updated state with LLM response
        """
        messages = state["messages"]
        iteration = state.get("iteration", 0)

        # Add system prompt on first iteration
        if iteration == 0:
            system_msg = SystemMessage(content=self.system_prompt)
            messages = [system_msg] + list(messages)

        # Call LLM
        response = self.llm.invoke(messages)

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

    def _execute_tools_node(self, state: AgentState) -> AgentState:
        """
        Parse tool calls from last message and execute them.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
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

            result = self.tool_executor.execute(call.name, call.arguments)

            # Detect structured error codes
            guidance = ""
            if not result.success and result.error_code:
                guidance = f"\nDetected error code {result.error_code}. Adjust your next tool choice or query accordingly."

            # Return result as HumanMessage (user role)
            tool_result_content = f"""Tool result for {call.name}:
```
{result.content}
```
{guidance}
Now synthesize an answer based on this information."""

            new_messages.append(HumanMessage(content=tool_result_content))

        return {"messages": new_messages}

    def _should_continue(self, state: AgentState) -> Literal["execute_tools", "end"]:
        """
        Decide if we should execute tools or end.

        Args:
            state: Current agent state

        Returns:
            Next node name or "end"
        """
        messages = state["messages"]
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)

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

    def run(self, query: str, memory=None) -> Dict:
        """
        Run agent on query.

        Args:
            query: User query
            memory: Optional ConversationMemory

        Returns:
            Dictionary with answer, tool_calls count, messages, and telemetry
        """
        base_messages = memory.snapshot() if memory else []
        base_messages.append(HumanMessage(content=query))

        initial_state = {
            "messages": base_messages,
            "iteration": 0,
            "max_iterations": self.max_iterations
        }

        final_state = None
        tool_calls_count = 0

        for step_output in self.app.stream(initial_state):
            final_state = step_output

            # Count tool calls
            for node_name, node_state in step_output.items():
                if "messages" in node_state:
                    last_msg = node_state["messages"][-1]
                    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                    if "<tool_call" in content:
                        tool_calls_count += len(parse_tool_calls(content))

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
            "messages": messages,
            "telemetry": self.tool_executor.telemetry_summary()
        }


def stream_agent(
    app,
    query: str,
    max_iterations: int = 10,
    memory: Optional = None,
    verbose: bool = False,
    stream_handler: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    Stream agent execution for UI/CLI consumption.

    Args:
        app: Compiled LangGraph application
        query: User query
        max_iterations: Maximum iterations
        memory: Optional conversation memory
        verbose: Print progress
        stream_handler: Optional callback for streaming

    Returns:
        Dictionary with answer, tool_calls, and messages
    """
    base_messages = memory.snapshot() if memory else []
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
    memory: Optional = None,
    verbose: bool = True,
    stream_handler: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    Convenience wrapper around stream_agent with optional streaming callback.

    Args:
        app: Compiled LangGraph application
        query: User query
        max_iterations: Maximum iterations
        memory: Optional conversation memory
        verbose: Print progress
        stream_handler: Optional callback for streaming

    Returns:
        Dictionary with answer, tool_calls, and messages
    """
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
