"""
Multihop Graph Analysis
========================

Provides sophisticated multi-hop analysis capabilities for knowledge graphs:
- Causal chain discovery
- Prerequisites analysis
- Influence mapping
- Alternative paths
- Process sequences
- Critical node identification
"""

import logging
from typing import Callable, List, Dict, Any, Optional
from neo4j import Driver
from neo4j.exceptions import Neo4jError

logger = logging.getLogger(__name__)


class MultihopAnalyzer:
    """
    Analyzes complex relationships across multiple hops in the knowledge graph.

    This class provides methods for:
    - Finding causal chains between concepts
    - Identifying prerequisites and dependencies
    - Mapping influence networks
    - Discovering alternative paths
    - Analyzing process sequences
    - Identifying critical nodes
    """

    def __init__(self, driver: Driver, embed_fn: Callable[[str], List[float]]):
        """
        Initialize MultihopAnalyzer.

        Args:
            driver: Neo4j driver instance
            embed_fn: Function to generate embeddings for semantic search
        """
        self.driver = driver
        self.embed_fn = embed_fn

    def find_causal_chain(
        self,
        from_concept: str,
        to_concept: str,
        max_hops: int = 5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find causal chains from one concept to another.

        This answers questions like:
        - "How does X lead to Y?"
        - "What's the causal relationship between X and Y?"

        Args:
            from_concept: Starting concept
            to_concept: Target concept
            max_hops: Maximum path length to search
            limit: Maximum number of paths to return

        Returns:
            List of paths with nodes and relationships
        """
        # First, find relevant entities using semantic search
        from_embedding = self.embed_fn(from_concept)
        to_embedding = self.embed_fn(to_concept)

        cypher = """
        // Find starting nodes
        CALL db.index.vector.queryNodes('entity-embeddings', $from_top_k, $from_embedding)
        YIELD node AS start_node, score AS start_score
        WHERE start_score > 0.7

        WITH collect(start_node)[0..3] AS start_nodes
        UNWIND start_nodes AS start

        // Find ending nodes
        CALL db.index.vector.queryNodes('entity-embeddings', $to_top_k, $to_embedding)
        YIELD node AS end_node, score AS end_score
        WHERE end_score > 0.7

        WITH start, collect(end_node)[0..3] AS end_nodes
        UNWIND end_nodes AS end

        // Find paths between them
        MATCH path = allShortestPaths((start)-[*1..$max_hops]-(end))
        WHERE start <> end

        WITH path,
             [node IN nodes(path) | {id: node.id, name: coalesce(node.name, node.title, node.id), content: node.content}] AS path_nodes,
             [rel IN relationships(path) | {type: type(rel), properties: properties(rel)}] AS path_rels,
             length(path) AS path_length

        RETURN path_nodes, path_rels, path_length
        ORDER BY path_length ASC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    from_embedding=from_embedding,
                    to_embedding=to_embedding,
                    from_top_k=3,
                    to_top_k=3,
                    max_hops=max_hops,
                    limit=limit
                )

                paths = []
                for record in result:
                    paths.append({
                        "nodes": record["path_nodes"],
                        "relationships": record["path_rels"],
                        "length": record["path_length"]
                    })

                return paths

        except Neo4jError as e:
            logger.error(f"Error finding causal chain: {e}")
            return []

    def find_prerequisites(
        self,
        concept: str,
        max_depth: int = 3,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find prerequisites/dependencies for a concept.

        This answers questions like:
        - "What do I need to know before learning X?"
        - "What are the dependencies for X?"

        Args:
            concept: Target concept
            max_depth: How many levels deep to search
            limit: Maximum results

        Returns:
            List of prerequisite entities with their relationships
        """
        embedding = self.embed_fn(concept)

        cypher = """
        // Find target entity
        CALL db.index.vector.queryNodes('entity-embeddings', 3, $embedding)
        YIELD node AS target, score
        WHERE score > 0.7

        WITH target
        LIMIT 1

        // Find incoming dependencies (things that lead TO this concept)
        MATCH path = (prereq)-[*1..$max_depth]->(target)
        WHERE prereq <> target

        WITH prereq,
             [node IN nodes(path) | {id: node.id, name: coalesce(node.name, node.title, node.id), content: node.content}] AS path_nodes,
             length(path) AS depth,
             path

        RETURN prereq.id AS prereq_id,
               coalesce(prereq.name, prereq.title, prereq.id) AS prereq_name,
               prereq.content AS prereq_content,
               path_nodes,
               depth
        ORDER BY depth ASC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    embedding=embedding,
                    max_depth=max_depth,
                    limit=limit
                )

                prerequisites = []
                for record in result:
                    prerequisites.append({
                        "id": record["prereq_id"],
                        "name": record["prereq_name"],
                        "content": record["prereq_content"],
                        "path": record["path_nodes"],
                        "depth": record["depth"]
                    })

                return prerequisites

        except Neo4jError as e:
            logger.error(f"Error finding prerequisites: {e}")
            return []

    def find_influence(
        self,
        concept: str,
        max_depth: int = 3,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find what a concept influences (downstream effects).

        This answers questions like:
        - "What does X influence?"
        - "What are the consequences of X?"

        Args:
            concept: Source concept
            max_depth: How many levels deep to search
            limit: Maximum results

        Returns:
            List of influenced entities
        """
        embedding = self.embed_fn(concept)

        cypher = """
        // Find source entity
        CALL db.index.vector.queryNodes('entity-embeddings', 3, $embedding)
        YIELD node AS source, score
        WHERE score > 0.7

        WITH source
        LIMIT 1

        // Find outgoing influences (things this concept affects)
        MATCH path = (source)-[*1..$max_depth]->(influenced)
        WHERE influenced <> source

        WITH influenced,
             [node IN nodes(path) | {id: node.id, name: coalesce(node.name, node.title, node.id), content: node.content}] AS path_nodes,
             length(path) AS depth

        RETURN influenced.id AS influenced_id,
               coalesce(influenced.name, influenced.title, influenced.id) AS influenced_name,
               influenced.content AS influenced_content,
               path_nodes,
               depth
        ORDER BY depth ASC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    embedding=embedding,
                    max_depth=max_depth,
                    limit=limit
                )

                influences = []
                for record in result:
                    influences.append({
                        "id": record["influenced_id"],
                        "name": record["influenced_name"],
                        "content": record["influenced_content"],
                        "path": record["path_nodes"],
                        "depth": record["depth"]
                    })

                return influences

        except Neo4jError as e:
            logger.error(f"Error finding influences: {e}")
            return []

    def find_alternatives(
        self,
        concept: str,
        similarity_threshold: float = 0.75,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find alternative concepts that serve similar purposes.

        This answers questions like:
        - "What can I use instead of X?"
        - "What are alternatives to X?"

        Args:
            concept: Source concept
            similarity_threshold: Minimum similarity score
            limit: Maximum results

        Returns:
            List of alternative entities with similarity scores
        """
        embedding = self.embed_fn(concept)

        cypher = """
        // Find semantically similar entities
        CALL db.index.vector.queryNodes('entity-embeddings', $limit * 2, $embedding)
        YIELD node, score
        WHERE score > $threshold AND score < 0.95  // Not too similar (avoid finding exact match)

        // Check if they share similar relationship patterns
        OPTIONAL MATCH (node)-[r1]-(neighbor)
        WITH node, score, collect(DISTINCT type(r1)) AS rel_types, count(DISTINCT neighbor) AS neighbor_count

        RETURN node.id AS alt_id,
               coalesce(node.name, node.title, node.id) AS alt_name,
               node.content AS alt_content,
               score,
               rel_types,
               neighbor_count
        ORDER BY score DESC, neighbor_count DESC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    embedding=embedding,
                    threshold=similarity_threshold,
                    limit=limit
                )

                alternatives = []
                for record in result:
                    alternatives.append({
                        "id": record["alt_id"],
                        "name": record["alt_name"],
                        "content": record["alt_content"],
                        "similarity_score": record["score"],
                        "relationship_types": record["rel_types"],
                        "neighbor_count": record["neighbor_count"]
                    })

                return alternatives

        except Neo4jError as e:
            logger.error(f"Error finding alternatives: {e}")
            return []

    def find_process_sequence(
        self,
        start_concept: str,
        max_steps: int = 10,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find sequential processes or workflows starting from a concept.

        This answers questions like:
        - "What comes after X?"
        - "What's the process starting with X?"

        Args:
            start_concept: Starting point
            max_steps: Maximum number of steps in sequence
            limit: Maximum number of sequences

        Returns:
            List of sequential paths
        """
        embedding = self.embed_fn(start_concept)

        cypher = """
        // Find starting entity
        CALL db.index.vector.queryNodes('entity-embeddings', 3, $embedding)
        YIELD node AS start, score
        WHERE score > 0.7

        WITH start
        LIMIT 1

        // Find sequential paths (looking for chains, not branches)
        MATCH path = (start)-[*1..$max_steps]->(end)
        WHERE start <> end

        // Filter for relatively linear paths (not too many branches)
        WITH path,
             nodes(path) AS path_nodes,
             relationships(path) AS path_rels,
             length(path) AS steps

        // Return sequential information
        RETURN [node IN path_nodes | {
                   id: node.id,
                   name: coalesce(node.name, node.title, node.id),
                   content: node.content
               }] AS sequence,
               [rel IN path_rels | {
                   type: type(rel),
                   properties: properties(rel)
               }] AS transitions,
               steps
        ORDER BY steps ASC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    embedding=embedding,
                    max_steps=max_steps,
                    limit=limit
                )

                sequences = []
                for record in result:
                    sequences.append({
                        "sequence": record["sequence"],
                        "transitions": record["transitions"],
                        "steps": record["steps"]
                    })

                return sequences

        except Neo4jError as e:
            logger.error(f"Error finding process sequence: {e}")
            return []

    def find_critical_nodes(
        self,
        context: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find the most critical/central nodes related to a context.

        This uses network centrality metrics to identify important nodes:
        - Degree centrality (how connected)
        - Betweenness (how often on shortest paths)

        Args:
            context: Context or domain to analyze
            limit: Maximum results

        Returns:
            List of critical nodes with centrality scores
        """
        embedding = self.embed_fn(context)

        cypher = """
        // Find relevant subgraph
        CALL db.index.vector.queryNodes('entity-embeddings', 20, $embedding)
        YIELD node AS seed, score
        WHERE score > 0.6

        WITH collect(seed) AS seeds

        // Get expanded neighborhood
        UNWIND seeds AS seed
        MATCH (seed)-[*0..2]-(related)

        WITH DISTINCT related

        // Calculate degree centrality
        OPTIONAL MATCH (related)-[r]-()
        WITH related, count(r) AS degree

        // Calculate betweenness approximation (via path counting)
        OPTIONAL MATCH path = allShortestPaths((a)-[*1..3]-(b))
        WHERE a <> b AND related IN nodes(path)
        WITH related, degree, count(path) AS betweenness_approx

        // Combine metrics
        RETURN related.id AS node_id,
               coalesce(related.name, related.title, related.id) AS node_name,
               related.content AS node_content,
               degree,
               betweenness_approx,
               (degree * 0.6 + betweenness_approx * 0.4) AS criticality_score
        ORDER BY criticality_score DESC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    embedding=embedding,
                    limit=limit
                )

                critical_nodes = []
                for record in result:
                    critical_nodes.append({
                        "id": record["node_id"],
                        "name": record["node_name"],
                        "content": record["node_content"],
                        "degree": record["degree"],
                        "betweenness": record["betweenness_approx"],
                        "criticality_score": record["criticality_score"]
                    })

                return critical_nodes

        except Neo4jError as e:
            logger.error(f"Error finding critical nodes: {e}")
            return []
