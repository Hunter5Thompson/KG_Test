"""
Basic GraphRAG Tools
====================

Core retrieval and query tools for GraphRAG:
- SemanticSearchTool: Vector-based semantic search
- HybridRetrieveTool: Combined vector + graph + keyword search
- CypherQueryTool: Direct Cypher query execution
- SchemaOverviewTool: Graph schema inspection
"""

import logging
from typing import Dict, Any, Callable, List
from neo4j import Driver
from neo4j.exceptions import CypherSyntaxError, Neo4jError

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class SemanticSearchTool(BaseTool):
    """Semantic vector search tool for finding entities by meaning"""

    def __init__(self, driver: Driver, embed_fn: Callable[[str], List[float]]):
        self.driver = driver
        self.embed_fn = embed_fn

    @property
    def name(self) -> str:
        return "semantic_search"

    @property
    def description(self) -> str:
        return "Find entities by semantic/vector similarity (meaning-based)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "required": True},
            "top_k": {"type": "integer", "required": False, "default": 5}
        }

    def execute(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """Execute semantic search"""
        try:
            from src.graphrag.hybrid_retriever import HybridGraphRetriever, EmbeddingReranker

            retriever = HybridGraphRetriever(
                self.driver,
                self.embed_fn,
                reranker=EmbeddingReranker(self.embed_fn),
            )
            results = retriever.retrieve(query, strategy="vector", top_k=top_k)

            if not results:
                return ToolResult(
                    success=False,
                    content=f"No results found for query: {query}",
                    metadata={"query": query, "top_k": top_k}
                )

            output = [f"SEMANTIC SEARCH - Found {len(results)} entities:\n"]
            for i, res in enumerate(results, 1):
                output.append(
                    f"{i}. {res.entity_id} (score: {res.score:.3f})\n"
                    f"   {res.context[:150]}...\n"
                )

            return ToolResult(
                success=True,
                content="".join(output),
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "num_results": len(results)
                }
            )

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return ToolResult(
                success=False,
                content=f"Error during semantic search: {str(e)}",
                metadata={"query": query},
                error_code="ERR_SEMANTIC_SEARCH"
            )


class HybridRetrieveTool(BaseTool):
    """Hybrid retrieval combining vector, graph, and keyword search"""

    def __init__(self, driver: Driver, embed_fn: Callable[[str], List[float]]):
        self.driver = driver
        self.embed_fn = embed_fn

    @property
    def name(self) -> str:
        return "hybrid_retrieve"

    @property
    def description(self) -> str:
        return "Combined search (Vector + Graph + Keyword) — BEST default for comprehensive retrieval"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "required": True},
            "top_k": {"type": "integer", "required": False, "default": 5},
            "expand_hops": {"type": "integer", "required": False, "default": 1}
        }

    def execute(self, query: str, top_k: int = 5, expand_hops: int = 1, **kwargs) -> ToolResult:
        """Execute hybrid retrieval"""
        try:
            from src.graphrag.hybrid_retriever import HybridGraphRetriever, EmbeddingReranker

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
                return ToolResult(
                    success=False,
                    content=f"No results found for query: {query}",
                    metadata={"query": query, "top_k": top_k, "expand_hops": expand_hops}
                )

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

            return ToolResult(
                success=True,
                content="".join(output),
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "expand_hops": expand_hops,
                    "num_results": len(results)
                }
            )

        except Exception as e:
            logger.error(f"Hybrid retrieve failed: {e}")
            return ToolResult(
                success=False,
                content=f"Error during hybrid retrieval: {str(e)}",
                metadata={"query": query},
                error_code="ERR_HYBRID_RETRIEVE"
            )


class CypherQueryTool(BaseTool):
    """Execute read-only Cypher queries"""

    ERR_CYPHER_GUARDRAIL = "ERR_CYPHER_GUARDRAIL"
    ERR_CYPHER_RUNTIME = "ERR_CYPHER_RUNTIME"

    def __init__(self, driver: Driver):
        self.driver = driver

    @property
    def name(self) -> str:
        return "cypher_query"

    @property
    def description(self) -> str:
        return "Execute a READ-ONLY Cypher query for precise, multi-hop structure"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "description": {"type": "string", "required": True},
            "cypher": {"type": "string", "required": True}
        }

    def execute(self, description: str, cypher: str, **kwargs) -> ToolResult:
        """Execute Cypher query with safety checks"""
        # Safety check
        query_upper = cypher.upper().strip()
        dangerous = ["CREATE", "DELETE", "SET", "REMOVE", "MERGE", "DROP", "ALTER", "DETACH"]

        if any(kw in query_upper for kw in dangerous):
            return ToolResult(
                success=False,
                content="WRITE operations not allowed. Use READ-ONLY queries.",
                metadata={"description": description, "cypher": cypher},
                error_code=self.ERR_CYPHER_GUARDRAIL
            )

        try:
            with self.driver.session() as session:
                result = session.run(cypher, timeout=10.0)
                records = list(result)

                if not records:
                    return ToolResult(
                        success=False,
                        content=f"Query returned no results.\nDescription: {description}\nQuery: {cypher}",
                        metadata={"description": description, "cypher": cypher}
                    )

                output = [f"CYPHER RESULTS ({len(records)} records):\n"]
                output.append(f"Description: {description}\n\n")

                for i, record in enumerate(records[:20], 1):
                    output.append(f"{i}. {dict(record)}\n")

                if len(records) > 20:
                    output.append(f"\n... ({len(records) - 20} more rows omitted)")

                return ToolResult(
                    success=True,
                    content="".join(output),
                    metadata={
                        "description": description,
                        "cypher": cypher,
                        "num_records": len(records)
                    }
                )

        except CypherSyntaxError as e:
            return ToolResult(
                success=False,
                content=f"Cypher syntax error: {str(e)}\nQuery: {cypher}",
                metadata={"description": description, "cypher": cypher},
                error_code=self.ERR_CYPHER_RUNTIME
            )
        except Neo4jError as e:
            return ToolResult(
                success=False,
                content=f"Cypher execution error: {str(e)}\nQuery: {cypher}",
                metadata={"description": description, "cypher": cypher},
                error_code=self.ERR_CYPHER_RUNTIME
            )


class SchemaOverviewTool(BaseTool):
    """Provide graph schema information"""

    def __init__(self, driver: Driver):
        self.driver = driver

    @property
    def name(self) -> str:
        return "schema_overview"

    @property
    def description(self) -> str:
        return "Retrieve labels, relationship types, and known node properties to inform Cypher planning"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {}  # No parameters required

    def execute(self, **kwargs) -> ToolResult:
        """Fetch and format graph schema"""
        try:
            with self.driver.session() as session:
                labels = session.run("CALL db.labels() YIELD label RETURN label").value()
                rels = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                ).value()
                props = session.run(
                    "CALL db.schema.nodeTypeProperties()"
                    " YIELD nodeLabels, propertyName RETURN nodeLabels, propertyName"
                ).data()

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

            return ToolResult(
                success=True,
                content="\n".join(lines),
                metadata={
                    "num_labels": len(labels or []),
                    "num_rel_types": len(rels or []),
                    "num_properties": len(prop_lines)
                }
            )

        except Neo4jError as e:
            return ToolResult(
                success=False,
                content=f"Failed to fetch schema: {str(e)}",
                metadata={},
                error_code="ERR_SCHEMA_FETCH"
            )
