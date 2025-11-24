"""
Flow tracker for capturing and storing agent execution steps for visualization.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


@dataclass
class FlowNode:
    """Represents a node in the agent execution flow."""

    id: str
    label: str
    node_type: str  # "call_model", "execute_tools", "start", "end"
    iteration: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class FlowEdge:
    """Represents an edge/transition in the agent execution flow."""

    source: str
    target: str
    label: Optional[str] = None
    edge_type: str = "normal"  # "normal", "conditional", "tool_call"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "edge_type": self.edge_type,
            "metadata": self.metadata
        }


class AgentFlowTracker:
    """Tracks agent execution flow for visualization."""

    def __init__(self):
        """Initialize the flow tracker."""
        self.nodes: List[FlowNode] = []
        self.edges: List[FlowEdge] = []
        self.current_iteration: int = 0
        self.previous_node_id: Optional[str] = None
        self.tool_calls: List[Dict[str, Any]] = []
        self.start_time: datetime = datetime.now()
        self._node_counter: int = 0

        # Add start node
        start_node = FlowNode(
            id="start",
            label="START",
            node_type="start",
            iteration=0,
            timestamp=self.start_time
        )
        self.nodes.append(start_node)
        self.previous_node_id = "start"

    def _generate_node_id(self, node_type: str, iteration: int) -> str:
        """Generate unique node ID."""
        self._node_counter += 1
        return f"{node_type}_{iteration}_{self._node_counter}"

    def record_step(self, step_output: Dict[str, Any]) -> None:
        """
        Record a step in the agent execution flow.

        Args:
            step_output: The step output from LangGraph app.stream()
                Format: {node_name: node_state}
        """
        for node_name, node_state in step_output.items():
            # Extract iteration if available
            iteration = node_state.get("iteration", self.current_iteration)
            self.current_iteration = max(self.current_iteration, iteration)

            # Generate node ID
            node_id = self._generate_node_id(node_name, iteration)

            # Extract metadata
            metadata = {
                "iteration": iteration,
                "max_iterations": node_state.get("max_iterations", "N/A")
            }

            # Handle different node types
            if node_name == "call_model":
                label = f"LLM Call\n(iter {iteration})"

                # Check if there are messages with tool calls
                messages = node_state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        content = str(last_msg.content)
                        # Check for tool calls in content
                        if "<tool_call>" in content:
                            metadata["has_tool_calls"] = True

            elif node_name == "execute_tools":
                label = f"Tool Execution\n(iter {iteration})"

                # Try to extract tool information
                messages = node_state.get("messages", [])
                tools_used = []

                for msg in messages:
                    if hasattr(msg, "content"):
                        content = str(msg.content)
                        # Parse tool calls from content
                        if "<tool_call>" in content:
                            try:
                                # Extract tool name from JSON
                                start = content.find("{")
                                end = content.rfind("}") + 1
                                if start >= 0 and end > start:
                                    tool_data = json.loads(content[start:end])
                                    tool_name = tool_data.get("name", "unknown")
                                    tools_used.append(tool_name)

                                    # Track tool call
                                    self.tool_calls.append({
                                        "iteration": iteration,
                                        "tool_name": tool_name,
                                        "arguments": tool_data.get("arguments", {}),
                                        "node_id": node_id
                                    })
                            except json.JSONDecodeError:
                                pass

                if tools_used:
                    metadata["tools_used"] = tools_used
                    label = f"Execute:\n{', '.join(tools_used)}"

            else:
                label = node_name

            # Create node
            node = FlowNode(
                id=node_id,
                label=label,
                node_type=node_name,
                iteration=iteration,
                timestamp=datetime.now(),
                metadata=metadata
            )
            self.nodes.append(node)

            # Create edge from previous node
            if self.previous_node_id:
                edge_label = None
                edge_type = "normal"

                # Determine edge type
                if node_name == "execute_tools":
                    edge_label = "has tools"
                    edge_type = "tool_call"
                elif self.previous_node_id.startswith("execute_tools"):
                    edge_label = "return"
                    edge_type = "normal"

                edge = FlowEdge(
                    source=self.previous_node_id,
                    target=node_id,
                    label=edge_label,
                    edge_type=edge_type
                )
                self.edges.append(edge)

            # Update previous node
            self.previous_node_id = node_id

    def finalize(self) -> None:
        """Finalize the flow by adding an END node."""
        if self.previous_node_id and self.previous_node_id != "end":
            end_node = FlowNode(
                id="end",
                label="END",
                node_type="end",
                iteration=self.current_iteration,
                timestamp=datetime.now()
            )
            self.nodes.append(end_node)

            # Create edge to end
            edge = FlowEdge(
                source=self.previous_node_id,
                target="end",
                label="complete",
                edge_type="normal"
            )
            self.edges.append(edge)
            self.previous_node_id = "end"

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution flow."""
        tool_usage = {}
        for tool_call in self.tool_calls:
            tool_name = tool_call["tool_name"]
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "iterations": self.current_iteration,
            "total_tool_calls": len(self.tool_calls),
            "tool_usage": tool_usage,
            "duration": (datetime.now() - self.start_time).total_seconds()
        }

    def reset(self) -> None:
        """Reset the tracker for a new execution."""
        self.__init__()
