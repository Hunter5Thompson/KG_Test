"""
Multihop Analysis Tools
========================

Advanced tools for multi-hop graph analysis:
- CausalChainTool: Find causal relationships between concepts
- PrerequisitesTool: Identify dependencies and prerequisites
- InfluenceTool: Map influence networks
- AlternativesTool: Discover alternative concepts
- ProcessSequenceTool: Analyze sequential processes
- CriticalNodesTool: Identify critical/central nodes
"""

import logging
from typing import Dict, Any, List

from .base import BaseTool, ToolResult
from ..multihop.analyzer import MultihopAnalyzer

logger = logging.getLogger(__name__)


class CausalChainTool(BaseTool):
    """Find causal chains between concepts"""

    def __init__(self, analyzer: MultihopAnalyzer):
        self.analyzer = analyzer

    @property
    def name(self) -> str:
        return "multihop_causal_chain"

    @property
    def description(self) -> str:
        return "Find HOW concept A leads to concept B (causal reasoning). Use for questions like 'How does X cause Y?' or 'What's the causal relationship between X and Y?'"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "from_concept": {"type": "string", "required": True},
            "to_concept": {"type": "string", "required": True},
            "max_hops": {"type": "integer", "required": False, "default": 5},
            "limit": {"type": "integer", "required": False, "default": 10}
        }

    def execute(self, from_concept: str, to_concept: str,
                max_hops: int = 5, limit: int = 10, **kwargs) -> ToolResult:
        """Execute causal chain analysis"""
        try:
            results = self.analyzer.find_causal_chain(
                from_concept, to_concept, max_hops, limit
            )

            if not results:
                return ToolResult(
                    success=False,
                    content=f"No causal chain found from '{from_concept}' to '{to_concept}' within {max_hops} hops.",
                    metadata={
                        "from_concept": from_concept,
                        "to_concept": to_concept,
                        "max_hops": max_hops
                    }
                )

            # Format output
            output = self._format_causal_chain(results, from_concept, to_concept)

            return ToolResult(
                success=True,
                content=output,
                metadata={
                    "from_concept": from_concept,
                    "to_concept": to_concept,
                    "num_paths": len(results)
                }
            )

        except Exception as e:
            logger.error(f"Causal chain analysis failed: {e}")
            return ToolResult(
                success=False,
                content=f"Error during causal chain analysis: {str(e)}",
                metadata={
                    "from_concept": from_concept,
                    "to_concept": to_concept
                },
                error_code="ERR_CAUSAL_CHAIN"
            )

    def _format_causal_chain(self, results: List[Dict], from_concept: str, to_concept: str) -> str:
        """Format causal chain results"""
        lines = [f"CAUSAL CHAIN: {from_concept} → {to_concept}\n"]
        lines.append(f"Found {len(results)} path(s):\n\n")

        for i, path in enumerate(results[:5], 1):  # Show top 5 paths
            lines.append(f"Path {i} ({path['length']} hops):\n")

            nodes = path['nodes']
            for j, node in enumerate(nodes):
                lines.append(f"  {j + 1}. {node['name']}")
                if node.get('content'):
                    content_preview = node['content'][:100].replace('\n', ' ')
                    lines.append(f"     └─ {content_preview}...\n")

                # Add relationship info
                if j < len(nodes) - 1 and j < len(path.get('relationships', [])):
                    rel = path['relationships'][j]
                    lines.append(f"     └─ [{rel.get('type', 'RELATED_TO')}] ↓\n")

            lines.append("\n")

        return "".join(lines)


class PrerequisitesTool(BaseTool):
    """Find prerequisites and dependencies"""

    def __init__(self, analyzer: MultihopAnalyzer):
        self.analyzer = analyzer

    @property
    def name(self) -> str:
        return "multihop_prerequisites"

    @property
    def description(self) -> str:
        return "Find prerequisites/dependencies for a concept. Use for questions like 'What do I need before X?' or 'What are the dependencies for X?'"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "concept": {"type": "string", "required": True},
            "max_depth": {"type": "integer", "required": False, "default": 3},
            "limit": {"type": "integer", "required": False, "default": 10}
        }

    def execute(self, concept: str, max_depth: int = 3, limit: int = 10, **kwargs) -> ToolResult:
        """Execute prerequisites analysis"""
        try:
            results = self.analyzer.find_prerequisites(concept, max_depth, limit)

            if not results:
                return ToolResult(
                    success=False,
                    content=f"No prerequisites found for '{concept}'.",
                    metadata={"concept": concept, "max_depth": max_depth}
                )

            # Format output
            output = self._format_prerequisites(results, concept)

            return ToolResult(
                success=True,
                content=output,
                metadata={
                    "concept": concept,
                    "num_prerequisites": len(results)
                }
            )

        except Exception as e:
            logger.error(f"Prerequisites analysis failed: {e}")
            return ToolResult(
                success=False,
                content=f"Error during prerequisites analysis: {str(e)}",
                metadata={"concept": concept},
                error_code="ERR_PREREQUISITES"
            )

    def _format_prerequisites(self, results: List[Dict], concept: str) -> str:
        """Format prerequisites results"""
        lines = [f"PREREQUISITES FOR: {concept}\n"]
        lines.append(f"Found {len(results)} prerequisite(s):\n\n")

        for i, prereq in enumerate(results, 1):
            lines.append(f"{i}. {prereq['name']} (depth: {prereq['depth']})\n")
            if prereq.get('content'):
                content_preview = prereq['content'][:150].replace('\n', ' ')
                lines.append(f"   └─ {content_preview}...\n")
            lines.append(f"   ID: {prereq['id']}\n\n")

        return "".join(lines)


class InfluenceTool(BaseTool):
    """Find what a concept influences"""

    def __init__(self, analyzer: MultihopAnalyzer):
        self.analyzer = analyzer

    @property
    def name(self) -> str:
        return "multihop_influence"

    @property
    def description(self) -> str:
        return "Find what a concept influences (downstream effects). Use for questions like 'What does X influence?' or 'What are the consequences of X?'"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "concept": {"type": "string", "required": True},
            "max_depth": {"type": "integer", "required": False, "default": 3},
            "limit": {"type": "integer", "required": False, "default": 10}
        }

    def execute(self, concept: str, max_depth: int = 3, limit: int = 10, **kwargs) -> ToolResult:
        """Execute influence analysis"""
        try:
            results = self.analyzer.find_influence(concept, max_depth, limit)

            if not results:
                return ToolResult(
                    success=False,
                    content=f"No influences found for '{concept}'.",
                    metadata={"concept": concept, "max_depth": max_depth}
                )

            # Format output
            output = self._format_influences(results, concept)

            return ToolResult(
                success=True,
                content=output,
                metadata={
                    "concept": concept,
                    "num_influences": len(results)
                }
            )

        except Exception as e:
            logger.error(f"Influence analysis failed: {e}")
            return ToolResult(
                success=False,
                content=f"Error during influence analysis: {str(e)}",
                metadata={"concept": concept},
                error_code="ERR_INFLUENCE"
            )

    def _format_influences(self, results: List[Dict], concept: str) -> str:
        """Format influence results"""
        lines = [f"INFLUENCES OF: {concept}\n"]
        lines.append(f"Found {len(results)} influenced concept(s):\n\n")

        for i, influenced in enumerate(results, 1):
            lines.append(f"{i}. {influenced['name']} (depth: {influenced['depth']})\n")
            if influenced.get('content'):
                content_preview = influenced['content'][:150].replace('\n', ' ')
                lines.append(f"   └─ {content_preview}...\n")
            lines.append(f"   ID: {influenced['id']}\n\n")

        return "".join(lines)


class AlternativesTool(BaseTool):
    """Find alternative concepts"""

    def __init__(self, analyzer: MultihopAnalyzer):
        self.analyzer = analyzer

    @property
    def name(self) -> str:
        return "multihop_alternatives"

    @property
    def description(self) -> str:
        return "Find alternative concepts that serve similar purposes. Use for questions like 'What can I use instead of X?' or 'What are alternatives to X?'"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "concept": {"type": "string", "required": True},
            "similarity_threshold": {"type": "number", "required": False, "default": 0.75},
            "limit": {"type": "integer", "required": False, "default": 10}
        }

    def execute(self, concept: str, similarity_threshold: float = 0.75,
                limit: int = 10, **kwargs) -> ToolResult:
        """Execute alternatives analysis"""
        try:
            results = self.analyzer.find_alternatives(concept, similarity_threshold, limit)

            if not results:
                return ToolResult(
                    success=False,
                    content=f"No alternatives found for '{concept}'.",
                    metadata={"concept": concept, "similarity_threshold": similarity_threshold}
                )

            # Format output
            output = self._format_alternatives(results, concept)

            return ToolResult(
                success=True,
                content=output,
                metadata={
                    "concept": concept,
                    "num_alternatives": len(results)
                }
            )

        except Exception as e:
            logger.error(f"Alternatives analysis failed: {e}")
            return ToolResult(
                success=False,
                content=f"Error during alternatives analysis: {str(e)}",
                metadata={"concept": concept},
                error_code="ERR_ALTERNATIVES"
            )

    def _format_alternatives(self, results: List[Dict], concept: str) -> str:
        """Format alternatives results"""
        lines = [f"ALTERNATIVES TO: {concept}\n"]
        lines.append(f"Found {len(results)} alternative(s):\n\n")

        for i, alt in enumerate(results, 1):
            lines.append(f"{i}. {alt['name']} (similarity: {alt['similarity_score']:.3f})\n")
            if alt.get('content'):
                content_preview = alt['content'][:150].replace('\n', ' ')
                lines.append(f"   └─ {content_preview}...\n")
            lines.append(f"   Connections: {alt['neighbor_count']}, Relationships: {', '.join(alt.get('relationship_types', [])[:3])}\n")
            lines.append(f"   ID: {alt['id']}\n\n")

        return "".join(lines)


class ProcessSequenceTool(BaseTool):
    """Find sequential processes"""

    def __init__(self, analyzer: MultihopAnalyzer):
        self.analyzer = analyzer

    @property
    def name(self) -> str:
        return "multihop_process_sequence"

    @property
    def description(self) -> str:
        return "Find sequential processes or workflows. Use for questions like 'What comes after X?' or 'What's the process starting with X?'"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "start_concept": {"type": "string", "required": True},
            "max_steps": {"type": "integer", "required": False, "default": 10},
            "limit": {"type": "integer", "required": False, "default": 5}
        }

    def execute(self, start_concept: str, max_steps: int = 10,
                limit: int = 5, **kwargs) -> ToolResult:
        """Execute process sequence analysis"""
        try:
            results = self.analyzer.find_process_sequence(start_concept, max_steps, limit)

            if not results:
                return ToolResult(
                    success=False,
                    content=f"No process sequences found starting with '{start_concept}'.",
                    metadata={"start_concept": start_concept, "max_steps": max_steps}
                )

            # Format output
            output = self._format_sequences(results, start_concept)

            return ToolResult(
                success=True,
                content=output,
                metadata={
                    "start_concept": start_concept,
                    "num_sequences": len(results)
                }
            )

        except Exception as e:
            logger.error(f"Process sequence analysis failed: {e}")
            return ToolResult(
                success=False,
                content=f"Error during process sequence analysis: {str(e)}",
                metadata={"start_concept": start_concept},
                error_code="ERR_PROCESS_SEQUENCE"
            )

    def _format_sequences(self, results: List[Dict], start_concept: str) -> str:
        """Format process sequence results"""
        lines = [f"PROCESS SEQUENCES STARTING WITH: {start_concept}\n"]
        lines.append(f"Found {len(results)} sequence(s):\n\n")

        for i, seq in enumerate(results, 1):
            lines.append(f"Sequence {i} ({seq['steps']} steps):\n")

            for j, node in enumerate(seq['sequence'], 1):
                lines.append(f"  Step {j}: {node['name']}\n")
                if node.get('content'):
                    content_preview = node['content'][:80].replace('\n', ' ')
                    lines.append(f"          {content_preview}...\n")

                # Add transition info
                if j < len(seq['sequence']) and j - 1 < len(seq.get('transitions', [])):
                    trans = seq['transitions'][j - 1]
                    lines.append(f"          ↓ [{trans.get('type', 'NEXT')}]\n")

            lines.append("\n")

        return "".join(lines)


class CriticalNodesTool(BaseTool):
    """Identify critical/central nodes"""

    def __init__(self, analyzer: MultihopAnalyzer):
        self.analyzer = analyzer

    @property
    def name(self) -> str:
        return "multihop_critical_nodes"

    @property
    def description(self) -> str:
        return "Find the most critical/central nodes in a context using network centrality. Use for questions like 'What are the most important concepts?' or 'What are the key nodes?'"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "context": {"type": "string", "required": True},
            "limit": {"type": "integer", "required": False, "default": 10}
        }

    def execute(self, context: str, limit: int = 10, **kwargs) -> ToolResult:
        """Execute critical nodes analysis"""
        try:
            results = self.analyzer.find_critical_nodes(context, limit)

            if not results:
                return ToolResult(
                    success=False,
                    content=f"No critical nodes found for context '{context}'.",
                    metadata={"context": context}
                )

            # Format output
            output = self._format_critical_nodes(results, context)

            return ToolResult(
                success=True,
                content=output,
                metadata={
                    "context": context,
                    "num_critical_nodes": len(results)
                }
            )

        except Exception as e:
            logger.error(f"Critical nodes analysis failed: {e}")
            return ToolResult(
                success=False,
                content=f"Error during critical nodes analysis: {str(e)}",
                metadata={"context": context},
                error_code="ERR_CRITICAL_NODES"
            )

    def _format_critical_nodes(self, results: List[Dict], context: str) -> str:
        """Format critical nodes results"""
        lines = [f"CRITICAL NODES FOR: {context}\n"]
        lines.append(f"Found {len(results)} critical node(s) (ranked by centrality):\n\n")

        for i, node in enumerate(results, 1):
            lines.append(f"{i}. {node['name']} (criticality: {node['criticality_score']:.2f})\n")
            lines.append(f"   Degree: {node['degree']}, Betweenness: {node['betweenness']}\n")
            if node.get('content'):
                content_preview = node['content'][:120].replace('\n', ' ')
                lines.append(f"   └─ {content_preview}...\n")
            lines.append(f"   ID: {node['id']}\n\n")

        return "".join(lines)
