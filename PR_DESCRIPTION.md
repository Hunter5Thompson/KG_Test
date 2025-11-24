# Add graph visualization and multihop query tool

## Summary

This PR adds two major features to the GraphRAG agent:

1. **Interactive Agent Execution Flow Visualization**
2. **Multi-Hop Query Tool for Complex Relationship Traversal**

## 1. Agent Execution Flow Visualization 📊

Implements interactive graph visualization to track the agent's execution flow in real-time.

### Features:
- **Flow Tracker** (`src/ui/flow_tracker.py`): Captures execution steps, nodes, edges, and tool calls
- **Graph Visualizer** (`src/ui/flow_visualizer.py`): Renders interactive graphs using streamlit-agraph
- **Integration**: Added to both agent chat and playground UIs
- **Visual Elements**:
  - 🟢 Green: START node
  - 🔵 Blue: LLM calls (call_model)
  - 🟠 Orange: Tool executions
  - 🔴 Red: END node
- **Statistics**: Shows iterations, tool usage, execution time, and detailed tool call arguments
- **Layout Options**: Hierarchical (top-down) or Force-Directed layouts

### UI Integration:
- **Chat Interface**: Collapsible expander "📊 Ausführungsgraph anzeigen"
- **Playground**: Always visible after execution
- **Interactive**: Click and drag nodes, hover for tooltips

## 2. Multi-Hop Query Tool 🔗

Solves the problem where the agent wasn't using the existing `MultihopAnalyzer` module for complex multi-hop questions.

### Problem:
The agent only had access to 4 tools and relied on `hybrid_retrieve` with `expand_hops=1` (insufficient for complex queries) or multiple sequential tool calls (inefficient).

### Solution:
Added `multihop_query` tool that integrates the `MultihopAnalyzer` module with 5 specialized query types:

1. **`causal_chain`**: Find paths between concepts ("How does X lead to Y?")
2. **`prerequisites`**: Find dependencies ("What's needed before X?")
3. **`influence`**: Find downstream effects ("What does X affect?")
4. **`process_sequence`**: Find sequential steps ("What comes after X?")
5. **`critical_nodes`**: Find most important nodes in context

### System Prompt Improvements:
- Added comprehensive tool documentation
- Improved decision heuristics to recognize multi-hop questions
- Added clear examples showing when to use `multihop_query`
- Updated multi-hop strategy to prefer the new tool

### Example Usage:
```json
{
  "name": "multihop_query",
  "arguments": {
    "query_type": "causal_chain",
    "from_concept": "90% losses problem",
    "to_concept": "Chapter 7 recommendation",
    "max_hops": 4
  }
}
```

## Changes:

### Modified Files:
- `pyproject.toml`: Added `streamlit-agraph>=0.0.45` dependency
- `src/graphrag/agent/agent_core.py`:
  - Added `_multihop_query()` method (+133 lines)
  - Updated system prompt with tool description and heuristics (+51 lines)
  - Integrated multihop tool into executor
- `src/ui/agent_ui.py`:
  - Integrated flow tracker into chat and playground
  - Added graph visualization display

### New Files:
- `src/ui/flow_tracker.py` (232 lines): Tracks agent execution flow
- `src/ui/flow_visualizer.py` (227 lines): Renders interactive graphs

## Testing:

To test the changes:

1. Install dependencies: `pip install -e .`
2. Start UI: `streamlit run src/ui/app.py`
3. Go to "Query Graph" or "Playground" tab
4. Ask a multi-hop question (e.g., "How does X lead to Y?")
5. View the execution graph visualization

## Impact:

- **Better UX**: Users can now visualize how the agent reasons
- **More Powerful**: Agent can handle complex multi-hop questions efficiently
- **Transparent**: Clear visibility into tool usage and execution flow
- **Performant**: Single tool call instead of multiple sequential calls for multi-hop queries

## Statistics:

- **Files changed**: 5
- **Lines added**: +676
- **New tools**: 1 (multihop_query with 5 query types)
- **New UI components**: 2 (flow_tracker, flow_visualizer)

## Commits:

1. `f9f0d65` - Add agent execution flow graph visualization
2. `86f03ce` - Add multihop_query tool for complex multi-hop questions
