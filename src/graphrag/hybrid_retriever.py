# src/graphrag/hybrid_retriever.py
"""
GraphRAG Hybrid Retriever
Kombiniert Vector Search + Graph Traversal + Keyword Search

Features:
- Robuste Index-Guards (ONLINE-Check, POPULATING-Handling)
- Fallback für Keyword-Suche (CONTAINS), falls Fulltext-Index fehlt
- Sicherer Relationship-Filter (Whitelist) gegen Injection
- Distanz-basierte Graph-Scoring-Logik (min(length(path)))
- Gewichtetete Fusion (Vector/Keyword/Graph) mit Multi-Source-Bonus
- Saubere Typen, Logging, Fehlerbehandlung, Google-Style Docstrings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import logging
import math

import numpy as np
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ClientError, Neo4jError

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------
ALLOWED_REL_TYPES: set[str] = {
    # Passe an dein Domain-Schema an:
    "RELATED_TO",
    "CITES",
    "MENTIONS",
    "DERIVED_FROM",
    "BELONGS_TO",
    "AUTHOR_OF",
}

SOURCE_WEIGHT: Dict[str, float] = {
    "vector": 1.0,
    "keyword": 0.85,
    "graph": 0.70,
}

DEFAULT_FULLTEXT_INDEX = "entity_fulltext_index"
DEFAULT_VECTOR_INDEX = "entity_vector_index"
DEFAULT_ENTITY_LABEL = "Entity"
DEFAULT_ID_PROP = "id"

# -----------------------------------------------------------------------------
# Datenklassen
# -----------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """Einzelnes Retrieval-Ergebnis."""
    entity_id: str
    score: float
    context: str
    source: str  # 'vector', 'graph', 'keyword'
    metadata: Dict


# -----------------------------------------------------------------------------
# Hilfsfunktionen (privat)
# -----------------------------------------------------------------------------
def _to_list(vec: Sequence[float] | np.ndarray) -> List[float]:
    """Konvertiert Embedding in List[float] für Neo4j."""
    if isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()
    return [float(x) for x in vec]


def _index_online(session, name: str) -> bool:
    """Prüft, ob ein Index ONLINE ist."""
    rec = session.run(
        "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state",
        name=name,
    ).single()
    return bool(rec and rec.get("state") == "ONLINE")


def _build_rel_filter(hops: int, relation_types: Optional[List[str]]) -> str:
    """Erstellt den Relationship-Pfadfilter (String) mit Whitelisting.

    Args:
        hops: Anzahl Hops (1..3, wird begrenzt).
        relation_types: Liste erlaubter Relationship-Typen.

    Returns:
        Cypher-Teilstring für den Relationship-Pfad (z. B. '-[:CITES|:MENTIONS*1..2]-').
    """
    hops = max(1, min(3, int(hops)))
    if relation_types:
        safe = [rt for rt in relation_types if rt in ALLOWED_REL_TYPES]
        if safe:
            rel = "|".join(f":{rt}" for rt in safe)
            return f"-[{rel}*1..{hops}]-"
    return f"-[*1..{hops}]-"


def _safe_min_distance(lengths: List[int]) -> int:
    """Sicheres Minimum, falls Liste leer wäre (sollte nicht passieren)."""
    return min(lengths) if lengths else 0


# -----------------------------------------------------------------------------
# Retriever
# -----------------------------------------------------------------------------
class HybridGraphRetriever:
    """3-stufiger Hybrid-Retriever (Vector, Keyword, Graph) für GraphRAG.

    Stage 1: Vector Search (Semantic Similarity) via db.index.vector.queryNodes
    Stage 2: Graph Expansion (Relationship Traversal)
    Stage 3: Fusion & Re-ranking (gewichtete Score-Summe + Bonus)

    Beispiel:
        >>> retriever = HybridGraphRetriever(driver, embed_fn)
        >>> results = retriever.retrieve("What are wargaming methodologies?", "hybrid", 5)
    """

    def __init__(
        self,
        driver: Driver,
        embed_fn: Callable[[str], Sequence[float] | np.ndarray],
        database: str = "neo4j",
        fulltext_index: str = DEFAULT_FULLTEXT_INDEX,
        vector_index: str = DEFAULT_VECTOR_INDEX,
        entity_label: str = DEFAULT_ENTITY_LABEL,
        id_property: str = DEFAULT_ID_PROP,
    ) -> None:
        """Initialisiert den Retriever.

        Args:
            driver: Neo4j Driver.
            embed_fn: Funktion, die Query-Text -> Embedding (List[float]/ndarray) liefert.
            database: Neo4j-Datenbankname.
            fulltext_index: Name des Full-Text-Indexes.
            vector_index: Name des Vector-Indexes.
            entity_label: Knotenlabel für Entitäten.
            id_property: Property-Name der Entitäts-ID.
        """
        self.driver = driver
        self.embed_fn = embed_fn
        self.database = database
        self.fulltext_index = fulltext_index
        self.vector_index = vector_index
        self.entity_label = entity_label
        self.id_property = id_property

    # --------------------------- Public API ---------------------------------

    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5,
        expand_hops: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Haupteinstiegspunkt für das Retrieval.

        Args:
            query: Benutzer-Query.
            strategy: 'vector' | 'keyword' | 'graph' | 'hybrid'.
            top_k: Anzahl gewünschter Ergebnisse.
            expand_hops: Graph-Expansionstiefe (empfohlen 1–2).
            relation_types: Whitelist für Relations (optional).

        Returns:
            Liste von RetrievalResult.
        """
        strategy = strategy.lower().strip()
        if strategy == "vector":
            return self._vector_search(query, k=top_k)
        if strategy == "keyword":
            return self._keyword_search(query, k=top_k)
        if strategy == "graph":
            seed = self._vector_search(query, k=min(3, top_k))
            return self._graph_expansion(seed, hops=expand_hops, relation_types=relation_types)
        if strategy == "hybrid":
            return self._hybrid_search(query, top_k, expand_hops, relation_types)
        raise ValueError(f"Unknown strategy: {strategy}")

    def get_context_for_entities(
        self,
        results: List[RetrievalResult],
        include_neighbors: bool = True,
        max_neighbors: int = 5,
    ) -> str:
        """Erzeugt kompakten Kontexttext für LLM.

        Args:
            results: Retrieval-Ergebnisse.
            include_neighbors: Ob Nachbarn (ausgehende Kanten) gelistet werden.
            max_neighbors: Max. Anzahl von nachbarschaftlichen Beziehungen.

        Returns:
            Formatierter Kontextstring.
        """
        if not results:
            return "No relevant entities found."

        lines: List[str] = []
        cypher = f"""
            MATCH (e:{self.entity_label} {{{self.id_property}: $entity_id}})
            OPTIONAL MATCH (e)-[r]->(nbr:{self.entity_label})
            WITH e, collect(DISTINCT {{type: type(r), target: nbr.{self.id_property}}}) AS rels
            RETURN e.{self.id_property} AS entity,
                   [x IN rels WHERE x.type IS NOT NULL AND x.target IS NOT NULL][..$maxn] AS relationships
        """

        with self.driver.session(database=self.database) as session:
            for res in results:
                try:
                    rec = session.run(
                        cypher, entity_id=res.entity_id, maxn=max_neighbors
                    ).single()
                except Neo4jError as e:
                    logger.error("get_context_for_entities failed for %s: %s", res.entity_id, e)
                    continue

                if not rec:
                    continue

                entity = rec.get("entity")
                rels = rec.get("relationships") or []
                entry = f"• **{entity}**\n  (Score: {res.score:.3f}, Source: {res.source})"
                if include_neighbors and rels:
                    pairs = [f"{r['type']} → {r['target']}" for r in rels if r.get("type") and r.get("target")]
                    if pairs:
                        entry += f"\n  Relationships: {', '.join(pairs)}"
                lines.append(entry)

        return "\n\n".join(lines)

    # --------------------------- Internals ----------------------------------

    def _vector_search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Stage 1: Vector Similarity Search (Node-Index)."""
        try:
            embedding = _to_list(self.embed_fn(query))
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return []

        with self.driver.session(database=self.database) as session:
            if not _index_online(session, self.vector_index):
                logger.warning("Vector index '%s' not ONLINE; returning empty results.", self.vector_index)
                return []

            rows = session.run(
                """
                CALL db.index.vector.queryNodes($index, $k, $vec)
                YIELD node, score
                RETURN node.$idprop AS entity_id, score
                ORDER BY score DESC
                LIMIT $k
                """.replace("$idprop", self.id_property),
                index=self.vector_index,
                k=k,
                vec=embedding,
            ).data()

        return [
            RetrievalResult(
                entity_id=r["entity_id"],
                score=float(r["score"]),
                context=f"Entity: {r['entity_id']}",
                source="vector",
                metadata={"embedding_similarity": float(r["score"])},
            )
            for r in rows
            if r.get("entity_id") is not None
        ]

    def _keyword_search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Stage 2: Keyword Search (Full-Text) mit Fallback auf CONTAINS."""
        with self.driver.session(database=self.database) as session:
            try:
                if not _index_online(session, self.fulltext_index):
                    logger.warning(
                        "Fulltext index '%s' missing/POPULATING; falling back to CONTAINS.",
                        self.fulltext_index,
                    )
                    rows = session.run(
                        f"""
                        MATCH (n:{self.entity_label})
                        WHERE ANY(p IN [n.name, n.title, n.text] WHERE p CONTAINS $q)
                        RETURN n.{self.id_property} AS entity_id, 0.5 AS score
                        LIMIT $k
                        """,
                        q=query,
                        k=k,
                    ).data()
                else:
                    rows = session.run(
                        """
                        CALL db.index.fulltext.queryNodes($index, $q)
                        YIELD node, score
                        RETURN node.$idprop AS entity_id, score
                        ORDER BY score DESC
                        LIMIT $k
                        """.replace("$idprop", self.id_property),
                        index=self.fulltext_index,
                        q=query,
                        k=k,
                    ).data()
            except ClientError as e:
                logger.error("Keyword search failed: %s", e)
                return []

        return [
            RetrievalResult(
                entity_id=r["entity_id"],
                score=float(r["score"]),
                context=f"Entity: {r['entity_id']}",
                source="keyword",
                metadata={"keyword_score": float(r["score"])},
            )
            for r in rows
            if r.get("entity_id") is not None
        ]

    def _graph_expansion(
        self,
        seed_entities: List[RetrievalResult],
        hops: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Stage 3: Graph Traversal von Seed-Entities.

        Scoring: 1 / (distance + 1), mit minimaler Pfaddistanz pro Nachbar.
        """
        if not seed_entities:
            return []

        entity_ids = [e.entity_id for e in seed_entities]
        rel_filter = _build_rel_filter(hops, relation_types)

        # APOC-freie Variante (Set-Erzeugung via REDUCE)
        cypher = f"""
            MATCH (seed:{self.entity_label})
            WHERE seed.{self.id_property} IN $entity_ids
            MATCH path = (seed){rel_filter}(neighbor:{self.entity_label})
            WITH neighbor,
                 [rel IN relationships(path) WHERE rel IS NOT NULL | type(rel)] AS rel_types_list,
                 length(path) AS len
            WITH neighbor,
                 REDUCE(s = [], r IN rel_types_list |
                       CASE WHEN r IN s THEN s ELSE s + r END) AS rel_types,
                 len
            WITH neighbor, rel_types, min(len) AS distance
            RETURN neighbor.{self.id_property} AS entity_id, rel_types, distance
            ORDER BY distance ASC
            LIMIT 20
        """

        with self.driver.session(database=self.database) as session:
            try:
                rows = session.run(cypher, entity_ids=entity_ids).data()
            except Neo4jError as e:
                logger.error("Graph expansion failed: %s", e)
                return []

        out: List[RetrievalResult] = []
        for r in rows:
            eid = r.get("entity_id")
            if not eid:
                continue
            distance = int(r.get("distance", 0))
            rels = [rt for rt in (r.get("rel_types") or []) if rt]
            # Distance-Score: 1/(d+1), leicht abflachend ab d>2
            base = 1.0 / (distance + 1.0)
            score = base if distance <= 2 else base * (1.0 / (1.0 + math.log1p(distance - 2)))
            ctx = f"Entity: {eid}" + (f" (via {', '.join(rels)})" if rels else "")
            out.append(
                RetrievalResult(
                    entity_id=eid,
                    score=float(score),
                    context=ctx,
                    source="graph",
                    metadata={"distance": distance, "relation_types": rels},
                )
            )
        return out

    def _hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        expand_hops: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Kombiniert Vector + Keyword + Graph und führt gewichtetes Re-Ranking durch."""
        vector_results = self._vector_search(query, k=top_k)
        keyword_results = self._keyword_search(query, k=top_k)

        # Graph-Seed: top 3 Vector-Ergebnisse (falls vorhanden)
        graph_results = self._graph_expansion(
            seed_entities=vector_results[:3],
            hops=expand_hops,
            relation_types=relation_types,
        )

        all_results = vector_results + keyword_results + graph_results
        return self._fuse(all_results, top_k)

    def _fuse(self, buckets: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """Gewichtete Fusion + kleiner Multi-Source-Bonus."""
        agg: Dict[str, RetrievalResult] = {}
        seen_sources: Dict[str, set] = {}
        for r in buckets:
            w = SOURCE_WEIGHT.get(r.source, 0.7)
            eid = r.entity_id
            if eid not in agg:
                agg[eid] = RetrievalResult(
                    entity_id=eid,
                    score=r.score * w,
                    context=r.context,
                    source=r.source,  # „erste“ Quelle
                    metadata={"sources": [r.source], **r.metadata},
                )
                seen_sources[eid] = {r.source}
            else:
                agg[eid].score += r.score * w
                if r.source not in seen_sources[eid]:
                    seen_sources[eid].add(r.source)
                    agg[eid].score *= 1.05  # leichter Bonus für Mehrfachtreffer
                    agg[eid].metadata["sources"].append(r.source)

        final = sorted(agg.values(), key=lambda x: x.score, reverse=True)
        return final[:top_k]


# -----------------------------------------------------------------------------
# Optionales CLI-Testing / Smoke-Test
# -----------------------------------------------------------------------------
def main() -> None:
    """Einfacher Smoke-Test (nutzt Umgebungs-Config und Ollama-Embeddings)."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    try:
        from config.settings import AppConfig
        from src.embeddings.ollama_embeddings import OllamaEmbedding
    except Exception as e:
        logger.error("Required modules not found for main(): %s", e)
        return

    config = AppConfig.from_env()

    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password),
    )
    embedder = OllamaEmbedding(
        model_name=config.ollama.embedding_model,
        base_url=config.ollama.host,
        api_key=config.ollama.api_key,
    )
    logger.info(
        "✅ Ollama Embedding initialized: %s",
        getattr(config.ollama, "embedding_model", "<unknown>"),
    )

    retriever = HybridGraphRetriever(
        driver=driver,
        embed_fn=lambda x: embedder.get_query_embedding(x),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )

    queries = [
        "What are wargaming methodologies?",
        "Who authored papers about NATO?",
        "Find topics related to simulation",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        for strategy in ["vector", "keyword", "hybrid"]:
            print(f"\n{strategy.upper()} Search:")
            try:
                results = retriever.retrieve(query, strategy=strategy, top_k=3)
                if not results:
                    print("  (no results)")
                for i, r in enumerate(results, 1):
                    print(f"  {i}. {r.entity_id} (score: {r.score:.3f}) [{','.join(r.metadata.get('sources', [r.source]))}]")
            except Exception as e:
                print(f"  ⚠️  Error: {e}")

        print("\nHYBRID CONTEXT:")
        try:
            hybrid_results = retriever.retrieve(query, strategy="hybrid", top_k=5)
            ctx = retriever.get_context_for_entities(hybrid_results)
            print(ctx)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")

    driver.close()


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Quellen / Referenzen
#  - Neo4j Manual: Full-Text Index (db.index.fulltext.queryNodes)
#  - Neo4j Manual: Vector Index (db.index.vector.queryNodes)
#  - Neo4j Cypher: SHOW INDEXES, relationships(path), type(rel), length(path)
# -----------------------------------------------------------------------------
