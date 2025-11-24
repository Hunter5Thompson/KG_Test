"""
Visualization component for rendering agent execution flow graphs.
"""
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from typing import List, Optional
from .flow_tracker import AgentFlowTracker, FlowNode, FlowEdge


# Color scheme for different node types
NODE_COLORS = {
    "start": "#4CAF50",  # Green
    "end": "#F44336",  # Red
    "call_model": "#2196F3",  # Blue
    "execute_tools": "#FF9800",  # Orange
    "default": "#9E9E9E"  # Grey
}

# Node shapes for different types
NODE_SHAPES = {
    "start": "star",
    "end": "star",
    "call_model": "ellipse",
    "execute_tools": "box",
    "default": "dot"
}


def create_graph_config(directed: bool = True, hierarchical: bool = True) -> Config:
    """
    Create configuration for the graph visualization.

    Args:
        directed: Whether to show directed edges
        hierarchical: Whether to use hierarchical layout

    Returns:
        Config object for agraph
    """
    config = Config(
        width="100%",
        height=600,
        directed=directed,
        physics=not hierarchical,  # Disable physics if hierarchical
        hierarchical=hierarchical,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={
            "labelProperty": "label",
            "renderLabel": True,
            "size": 400,
            "fontSize": 14
        },
        link={
            "labelProperty": "label",
            "renderLabel": True,
            "fontSize": 12,
            "fontColor": "#666666"
        }
    )
    return config


def render_agent_flow_graph(
    tracker: AgentFlowTracker,
    show_summary: bool = True,
    hierarchical: bool = True
) -> None:
    """
    Render the agent execution flow as an interactive graph.

    Args:
        tracker: The AgentFlowTracker containing the flow data
        show_summary: Whether to show the execution summary
        hierarchical: Whether to use hierarchical layout (top-down)
    """
    if not tracker.nodes:
        st.info("🔄 Noch keine Ausführungsschritte aufgezeichnet.")
        return

    # Convert tracker nodes to agraph Nodes
    nodes: List[Node] = []
    for flow_node in tracker.nodes:
        color = NODE_COLORS.get(flow_node.node_type, NODE_COLORS["default"])
        shape = NODE_SHAPES.get(flow_node.node_type, NODE_SHAPES["default"])

        # Build node title (tooltip)
        title_parts = [
            f"<b>{flow_node.label}</b>",
            f"Type: {flow_node.node_type}",
            f"Iteration: {flow_node.iteration}",
            f"Time: {flow_node.timestamp.strftime('%H:%M:%S')}"
        ]

        # Add metadata to title
        if flow_node.metadata:
            for key, value in flow_node.metadata.items():
                if key not in ["iteration", "max_iterations"]:
                    title_parts.append(f"{key}: {value}")

        title = "<br/>".join(title_parts)

        node = Node(
            id=flow_node.id,
            label=flow_node.label,
            size=25 if flow_node.node_type in ["start", "end"] else 20,
            shape=shape,
            color=color,
            title=title
        )
        nodes.append(node)

    # Convert tracker edges to agraph Edges
    edges: List[Edge] = []
    for flow_edge in tracker.edges:
        # Determine edge style based on type
        if flow_edge.edge_type == "tool_call":
            color = "#FF9800"  # Orange for tool calls
            dashes = False
        elif flow_edge.edge_type == "conditional":
            color = "#9C27B0"  # Purple for conditionals
            dashes = True
        else:
            color = "#757575"  # Grey for normal flow
            dashes = False

        edge = Edge(
            source=flow_edge.source,
            target=flow_edge.target,
            label=flow_edge.label or "",
            color=color,
            dashes=dashes
        )
        edges.append(edge)

    # Create graph configuration
    config = create_graph_config(directed=True, hierarchical=hierarchical)

    # Render the graph
    st.subheader("📊 Agenten-Ausführungsgraph")

    # Layout options
    col1, col2 = st.columns([3, 1])
    with col2:
        layout_option = st.selectbox(
            "Layout:",
            ["Hierarchisch", "Force-Directed"],
            index=0 if hierarchical else 1
        )
        hierarchical = (layout_option == "Hierarchisch")
        config.hierarchical = hierarchical
        config.physics = not hierarchical

    # Render graph
    with col1:
        st.caption("🖱️ Interaktiv: Klicken und ziehen Sie Knoten, um die Ansicht anzupassen")

    return_value = agraph(nodes=nodes, edges=edges, config=config)

    # Show summary if requested
    if show_summary:
        st.divider()
        show_execution_summary(tracker)


def show_execution_summary(tracker: AgentFlowTracker) -> None:
    """
    Display a summary of the execution flow.

    Args:
        tracker: The AgentFlowTracker containing the flow data
    """
    summary = tracker.get_summary()

    st.subheader("📈 Ausführungsstatistik")

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Iterationen", summary["iterations"])

    with col2:
        st.metric("Knoten", summary["total_nodes"])

    with col3:
        st.metric("Tool-Aufrufe", summary["total_tool_calls"])

    with col4:
        st.metric("Dauer (s)", f"{summary['duration']:.2f}")

    # Show tool usage breakdown
    if summary["tool_usage"]:
        st.subheader("🔧 Tool-Verwendung")

        # Create a horizontal bar chart for tool usage
        tool_names = list(summary["tool_usage"].keys())
        tool_counts = list(summary["tool_usage"].values())

        # Display as columns with metrics
        cols = st.columns(len(tool_names))
        for idx, (tool_name, count) in enumerate(summary["tool_usage"].items()):
            with cols[idx]:
                st.metric(tool_name, count)

    # Show detailed tool calls
    if tracker.tool_calls:
        with st.expander("🔍 Detaillierte Tool-Aufrufe", expanded=False):
            for idx, tool_call in enumerate(tracker.tool_calls, 1):
                st.markdown(f"**{idx}. {tool_call['tool_name']}** (Iteration {tool_call['iteration']})")
                st.json(tool_call['arguments'])


def render_realtime_flow_graph(tracker: AgentFlowTracker) -> None:
    """
    Render a live-updating version of the flow graph.
    This can be called repeatedly to update the visualization as the agent runs.

    Args:
        tracker: The AgentFlowTracker containing the flow data
    """
    # Create a placeholder for the graph
    graph_placeholder = st.empty()

    with graph_placeholder.container():
        render_agent_flow_graph(tracker, show_summary=False, hierarchical=True)
