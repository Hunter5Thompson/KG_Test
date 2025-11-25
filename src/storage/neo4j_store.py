"""
Neo4j Storage Backend (UPGRADED)
=================================

Fixed:
- Adds name/title/summary/content properties
- Native relationship types (not RELATION {type: X})
- Creates necessary indexes
- Matches smoke_test.py behavior
"""
from typing import List, Dict, Optional
from neo4j import GraphDatabase
import math
import re
import logging

# Handle imports for both direct execution and module import
try:
    from ..models.triplets import Triplet
except ImportError:
    from src.models.triplets import Triplet

logger = logging.getLogger(__name__)


class Neo4jStore:
    """
    Neo4j storage backend for knowledge graphs
    
    UPGRADED to match smoke_test.py best practices:
    - Stores name/title/summary/content for entities
    - Uses native relationship types (e.g., :WORKS_AT)
    - Creates Fulltext, Vector, and BTREE indexes
    """
    
    FULLTEXT_INDEX = "entity_fulltext_index"
    VECTOR_INDEX = "entity_vector_index"
    NAME_INDEX = "entity_name_idx"
    ENTITY_LABEL = "Entity"
    ID_PROP = "id"
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        
        # Test connection
        self._test_connection()
        print(f"✅ Neo4j connected: {uri}")
        
        # Ensure indexes exist
        self._ensure_indexes()
    
    def _test_connection(self) -> None:
        """Test Neo4j connectivity"""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            with driver.session(database=self.database) as session:
                session.run("RETURN 1")
        finally:
            driver.close()
    
    def _ensure_indexes(self) -> None:
        """Create indexes (idempotent) - matches smoke_test.py"""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        try:
            with driver.session(database=self.database) as session:
                # Fulltext Index
                session.run(f"""
                    CREATE FULLTEXT INDEX {self.FULLTEXT_INDEX} IF NOT EXISTS
                    FOR (n:{self.ENTITY_LABEL}) ON EACH [n.title, n.summary, n.content]
                """).consume()
                
                # Vector Index (dimension will be set on first write)
                # Note: We'll create this dynamically when we know the dimension
                
                # BTREE Index on name
                session.run(f"""
                    CREATE INDEX {self.NAME_INDEX} IF NOT EXISTS
                    FOR (n:{self.ENTITY_LABEL}) ON (n.name)
                """).consume()
                
                logger.info("Indexes ensured: Fulltext, BTREE(name)")
        
        except Exception as e:
            logger.error(f"Failed to ensure indexes: {e}")
        finally:
            driver.close()
    
    def set_display_caption(self) -> None:
        """Set Neo4j Browser caption to show 'name' property instead of embedding"""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        try:
            with driver.session(database=self.database) as session:
                # Add caption property that mirrors name
                # This ensures Neo4j Browser shows readable labels
                # Fix old nodes without caption
                result = session.run(f"""
                    MATCH (n:{self.ENTITY_LABEL})
                    WHERE n.name IS NOT NULL AND (n.caption IS NULL OR n.caption = '')
                    SET n.caption = n.name
                    RETURN count(n) as updated
                """).single()
                
                if result and result['updated'] > 0:
                    logger.info(f"Updated {result['updated']} nodes with caption property")
                
                logger.info("Caption property set for Entity nodes")
        
        except Exception as e:
            logger.error(f"Failed to set caption property: {e}")
        finally:
            driver.close()
    
    def _ensure_vector_index(self, dim: int) -> None:
        """Create vector index with specific dimension"""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        try:
            with driver.session(database=self.database) as session:
                # Check if index exists
                result = session.run(
                    "SHOW INDEXES YIELD name WHERE name = $n RETURN count(*) as count",
                    n=self.VECTOR_INDEX
                ).single()
                
                if result["count"] == 0:
                    session.run(f"""
                        CREATE VECTOR INDEX {self.VECTOR_INDEX} IF NOT EXISTS
                        FOR (n:{self.ENTITY_LABEL}) ON (n.embedding)
                        OPTIONS {{
                          indexConfig: {{
                            `vector.dimensions`: $dim,
                            `vector.similarity_function`: 'cosine'
                          }}
                        }}
                    """, dim=int(dim)).consume()
                    logger.info(f"Vector index created (dim={dim})")
        
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
        finally:
            driver.close()
    
    @staticmethod
    def _sanitize_rel_type(raw: str) -> Optional[str]:
        """Create safe relationship type (A-Z, 0-9, _) - from smoke_test.py"""
        if not raw:
            return None
        t = re.sub(r"[^A-Za-z0-9_]+", "_", raw.strip())
        t = re.sub(r"_+", "_", t).strip("_")
        if not t:
            return None
        if re.match(r"^[0-9]", t):
            t = f"REL_{t}"
        return t[:50].upper()
    
    @staticmethod
    def _make_title(text: str) -> str:
        """Generate title from text - from smoke_test.py"""
        if not text:
            return "Untitled"
        first_line = text.split("\n", 1)[0].strip()
        if not first_line:
            first_line = text.strip()
        return (first_line[:117] + "…") if len(first_line) > 120 else first_line
    
    @staticmethod
    def _make_summary(text: str) -> str:
        """Generate summary from text - from smoke_test.py"""
        if not text:
            return "No summary available."
        cleaned = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        if len(parts) >= 2:
            summary = " ".join(parts[:2])
        else:
            summary = cleaned[:400]
        return summary[:400]
    
    def clear(self) -> None:
        """Delete all nodes and relationships"""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            with driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("🧹 Neo4j cleared")
        finally:
            driver.close()
    
    def write_triplets(
        self,
        triplets: List[Triplet],
        entity_embeddings: Optional[Dict[str, List[float]]] = None,
        entity_summaries: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Write triplets to Neo4j with properties and embeddings

        UPGRADED: Now writes name/title/summary/content + native rel types
        ENHANCED: Accepts entity_summaries for contextual descriptions
        """
        if not triplets:
            return

        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        try:
            with driver.session(database=self.database) as session:
                # Get unique entities
                entities = set()
                for t in triplets:
                    entities.add(t.subject)
                    entities.add(t.object)

                # Prepare entity data with properties
                entity_data = []
                for ent in entities:
                    safe_name = str(ent).strip() or "Unnamed Entity"
                    title = self._make_title(safe_name)

                    # Use LLM-generated summary if available
                    if entity_summaries and safe_name in entity_summaries:
                        summary = entity_summaries[safe_name]
                    else:
                        # CRITICAL: No fallback summary should be needed if extraction worked properly
                        # If we reach here, it indicates a failure in the entity summary generation
                        logger.warning(f"⚠️  Missing summary for entity: {safe_name} - This should not happen!")
                        summary = f"⚠️  MISSING SUMMARY: {safe_name} (Entity summary generation failed - please review extraction logs)"

                    data = {
                        "id": safe_name,
                        "name": safe_name,
                        "title": title,
                        "summary": summary,
                        "content": None,  # Could be enhanced with context
                        "caption": safe_name,  # ✅ Human-readable caption for Neo4j Browser display
                        "embedding": entity_embeddings.get(ent) if entity_embeddings else None
                    }
                    entity_data.append(data)
                
                # Upsert entities with all properties
                upsert_query = f"""
                UNWIND $batch AS row
                MERGE (e:{self.ENTITY_LABEL} {{ {self.ID_PROP}: row.id }})
                SET
                  e.caption = row.caption,
                  e.name = row.name,
                  e.title = coalesce(row.title, e.title),
                  e.summary = coalesce(row.summary, e.summary),
                  e.content = coalesce(row.content, e.content)
                FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END |
                  SET e.embedding = row.embedding
                )
                """
                session.run(upsert_query, batch=entity_data).consume()
                
                # Ensure vector index exists if we have embeddings
                if entity_embeddings:
                    first_emb = next(iter(entity_embeddings.values()))
                    if first_emb:
                        self._ensure_vector_index(len(first_emb))
                
                # Write relationships with native types
                # Group by relationship type for batch processing
                rels_by_type: Dict[str, List[Dict]] = {}
                fallback_rels = []
                
                for t in triplets:
                    rel_type = self._sanitize_rel_type(t.predicate)
                    
                    if rel_type:
                        rels_by_type.setdefault(rel_type, []).append({
                            "a": t.subject,
                            "b": t.object,
                            "type": t.predicate
                        })
                    else:
                        fallback_rels.append({
                            "a": t.subject,
                            "b": t.object,
                            "type": t.predicate
                        })
                
                # Write native relationship types
                for rel_type, rels in rels_by_type.items():
                    query = f"""
                    UNWIND $rels AS rel
                    MATCH (a:{self.ENTITY_LABEL} {{ {self.ID_PROP}: rel.a }})
                    MATCH (b:{self.ENTITY_LABEL} {{ {self.ID_PROP}: rel.b }})
                    MERGE (a)-[r:{rel_type}]->(b)
                    """
                    session.run(query, rels=rels).consume()
                
                # Fallback for unsanitizable relationships
                if fallback_rels:
                    query = f"""
                    UNWIND $rels AS rel
                    MATCH (a:{self.ENTITY_LABEL} {{ {self.ID_PROP}: rel.a }})
                    MATCH (b:{self.ENTITY_LABEL} {{ {self.ID_PROP}: rel.b }})
                    MERGE (a)-[r:RELATION {{type: rel.type}}]->(b)
                    """
                    session.run(query, rels=fallback_rels).consume()
        
        finally:
            driver.close()
        
        # Fix caption for existing nodes (for backwards compatibility)
        self.set_display_caption()
        
        print(f"✅ Written {len(triplets)} triplets to Neo4j")
    
    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics"""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        try:
            with driver.session(database=self.database) as session:
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            return {"nodes": node_count, "relationships": rel_count}
        finally:
            driver.close()
    
    def get_triplets(self, limit: int = 100) -> List[str]:
        """Retrieve triplets as strings for display"""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(f"""
                    MATCH (a:{self.ENTITY_LABEL})-[r]->(b:{self.ENTITY_LABEL})
                    RETURN a.{self.ID_PROP} as subject, 
                           coalesce(r.type, type(r)) as predicate, 
                           b.{self.ID_PROP} as object
                    LIMIT {limit}
                """)
                
                return [
                    f"({rec['subject']}) --[{rec['predicate']}]--> ({rec['object']})"
                    for rec in result
                ]
        finally:
            driver.close()
    
    def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 5
    ) -> List[Dict]:
        """Semantic search using vector index or fallback cosine"""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        try:
            with driver.session(database=self.database) as session:
                # Try vector index first
                try:
                    result = session.run(f"""
                        CALL db.index.vector.queryNodes($index, $k, $vec)
                        YIELD node, score
                        RETURN node.{self.ID_PROP} AS entity, 
                               node.name AS name, 
                               score
                        ORDER BY score DESC
                        LIMIT $k
                    """, index=self.VECTOR_INDEX, k=limit, vec=query_embedding).data()
                    
                    return [
                        {"entity": r["entity"], "name": r.get("name"), "similarity": float(r["score"])}
                        for r in result
                    ]
                
                except Exception:
                    # Fallback to cosine similarity
                    result = session.run(f"""
                        MATCH (n:{self.ENTITY_LABEL})
                        WHERE n.embedding IS NOT NULL
                        RETURN n.{self.ID_PROP} as entity, 
                               n.name as name, 
                               n.embedding as embedding
                    """)
                    
                    scored = []
                    for record in result:
                        sim = self._cosine_similarity(query_embedding, record["embedding"])
                        scored.append({
                            "entity": record["entity"],
                            "name": record.get("name"),
                            "similarity": sim
                        })
                    
                    scored.sort(key=lambda x: x["similarity"], reverse=True)
                    return scored[:limit]
        
        finally:
            driver.close()
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def query(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute custom Cypher query"""
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, params or {})
                return [dict(record) for record in result]
        finally:
            driver.close()
