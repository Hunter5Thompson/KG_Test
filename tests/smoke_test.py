from __future__ import annotations
import os
import re
import math
import httpx
import logging
from typing import List, Tuple, Optional, Dict, Iterable
from dataclasses import dataclass
from dotenv import load_dotenv

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError
from llama_index.core import Document
from llama_index.llms.ollama import Ollama
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from ollama import Client


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("kg_ingest")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings (Ollama)
# ─────────────────────────────────────────────────────────────────────────────
class OllamaEmbedding(BaseEmbedding):
    """
    Ollama Embedding Model Integration – /api/embed Endpoint.
    """

    model_name: str
    _base_url: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _http_client: httpx.Client = PrivateAttr()
    _embed_dim: int = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        embed_batch_size: int = 10,
        **kwargs
    ):
        super().__init__(model_name=model_name, embed_batch_size=embed_batch_size, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http_client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=30.0,
        )
        self._embed_dim = self._get_embedding_dim()
        print(f"✅ Ollama Embedding initialized: {model_name} (dim={self._embed_dim})")

    def _get_embedding_dim(self) -> int:
        test_embedding = self._get_embedding_via_http("test")
        if test_embedding and len(test_embedding) > 1:
            return len(test_embedding)
        logger.warning("Could not determine embedding dim, using default 2560")
        return 2560

    def _get_embedding_via_http(self, text: str) -> Optional[List[float]]:
        endpoint = f"{self._base_url}/api/embed"
        payload = {"model": self.model_name, "input": text}
        try:
            resp = self._http_client.post(endpoint, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                emb = data.get("embeddings")
                if isinstance(emb, list) and emb:
                    return emb[0] if isinstance(emb[0], list) else emb
            return None
        except Exception as e:
            logger.error("Embedding HTTP error: %s", e)
            return None

    @classmethod
    def class_name(cls) -> str:
        return "OllamaEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        emb = self._get_embedding_via_http(query)
        return emb if emb else [0.0] * self._embed_dim

    def _get_text_embedding(self, text: str) -> List[float]:
        emb = self._get_embedding_via_http(text)
        return emb if emb else [0.0] * self._embed_dim

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(t) for t in texts]

    def __del__(self):
        if hasattr(self, "_http_client"):
            try:
                self._http_client.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Datamodel
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Triplet:
    """Represents a knowledge graph triplet"""
    subject: str
    predicate: str
    object: str

    def __str__(self) -> str:
        return f"({self.subject}) --[{self.predicate}]--> ({self.object})"


# ─────────────────────────────────────────────────────────────────────────────
# KG Extractor + Ingest
# ─────────────────────────────────────────────────────────────────────────────
class KnowledgeGraphExtractor:
    """
    KG Extractor mit:
    - automatischem Setzen von name/title/summary/content
    - sicheren Relationship-Typen
    - Index-Bootstrap (Fulltext + Vector + BTREE(name))
    - optionalem Vector-Index Retrieval
    """

    FULLTEXT_INDEX = "entity_fulltext_index"
    VECTOR_INDEX = "entity_vector_index"
    NAME_INDEX = "entity_name_idx"
    ENTITY_LABEL = "Entity"
    ID_PROP = "id"

    def __init__(
        self,
        ollama_host: str,
        ollama_api_key: str,
        ollama_model: str,
        ollama_embedding_model: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        timeout: int = 120
    ):
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.ollama_embedding_model = ollama_embedding_model

        self.driver: Driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # LLM
        ollama_client = Client(host=ollama_host, headers={"Authorization": f"Bearer {ollama_api_key}"})
        self.llm = Ollama(
            model=ollama_model,
            base_url=ollama_host,
            request_timeout=timeout,
            client=ollama_client,
        )

        # Embeddings
        self.embed_model = OllamaEmbedding(
            model_name=ollama_embedding_model,
            base_url=ollama_host,
            api_key=ollama_api_key,
        )
        self._embed_dim = getattr(self.embed_model, "_embed_dim", 2560)

        print("✅ KG Extractor initialized")
        print(f"   LLM: {ollama_model}")
        print(f"   Embeddings: {ollama_embedding_model}")
        print(f"   Neo4j: {neo4j_uri}")

        # Indizes sicherstellen (idempotent)
        self._ensure_indexes()

    # ───────────────────── Utilities ─────────────────────
    def _ensure_indexes(self) -> None:
        """Erzeugt Fulltext-, Vector- und Name-Index (idempotent)."""
        try:
            with self.driver.session() as s:
                # Full-Text auf title/summary/content
                s.run(
                    f"""
                    CREATE FULLTEXT INDEX {self.FULLTEXT_INDEX} IF NOT EXISTS
                    FOR (n:{self.ENTITY_LABEL}) ON EACH [n.title, n.summary, n.content]
                    """
                ).consume()

                # Vector Index (Dimension aus Embedding)
                s.run(
                    f"""
                    CREATE VECTOR INDEX {self.VECTOR_INDEX} IF NOT EXISTS
                    FOR (n:{self.ENTITY_LABEL}) ON (n.embedding)
                    OPTIONS {{
                      indexConfig: {{
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                      }}
                    }}
                    """,
                    dim=int(self._embed_dim),
                ).consume()

                # BTREE-Index auf name für schnelle Captions/Lookups
                s.run(
                    f"""
                    CREATE INDEX {self.NAME_INDEX} IF NOT EXISTS
                    FOR (n:{self.ENTITY_LABEL}) ON (n.name)
                    """
                ).consume()

                # Status-Log
                idx = s.run(
                    """
                    SHOW INDEXES
                    YIELD name, type, state, labelsOrTypes, properties
                    WHERE name IN [$ft, $vx, $nx]
                    RETURN name, type, state, labelsOrTypes, properties
                    """,
                    ft=self.FULLTEXT_INDEX,
                    vx=self.VECTOR_INDEX,
                    nx=self.NAME_INDEX,
                ).data()
                for r in idx:
                    logger.info("Index %s: %s (%s) on %s %s", r["name"], r["type"], r["state"], r["labelsOrTypes"], r["properties"])
        except Neo4jError as e:
            logger.error("Failed to ensure indexes: %s", e)

    @staticmethod
    def _sanitize_rel_type(raw: str) -> Optional[str]:
        """Erzeugt sicheren Relationship-Type (A–Z, 0–9, _)."""
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
        """Einfache Heuristik: erste Zeile/Satz, 120 Zeichen."""
        if not text:
            return "Untitled"
        first_line = text.split("\n", 1)[0].strip()
        if not first_line:
            first_line = text.strip()
        return (first_line[:117] + "…") if len(first_line) > 120 else first_line

    @staticmethod
    def _make_summary(text: str) -> str:
        """Heuristik: 1–2 Sätze, 400 Zeichen max."""
        if not text:
            return "No summary available."
        cleaned = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        if len(parts) >= 2:
            summary = " ".join(parts[:2])
        else:
            summary = cleaned[:400]
        return summary[:400]

    # ───────────────────── LLM Triplets ─────────────────────
    def extract_triplets_from_text(self, text: str, verbose: bool = True) -> List[Triplet]:
        prompt = self._build_extraction_prompt(text)
        response = self.llm.complete(prompt)
        triplets = self._parse_triplets(response.text)
        if verbose:
            print(f"📝 Extracted {len(triplets)} triplets from: '{text[:50]}...'")
            for t in triplets:
                print(f"   • {t}")
        return triplets

    @staticmethod
    def _build_extraction_prompt(text: str) -> str:
        return f"""Extract ALL knowledge graph triplets from the following text.

RULES:
1. Return ONLY triplets in format: (subject, relation, object)
2. One triplet per line
3. Use clear, concise relation names (e.g., works_at, located_in, acquired)
4. Include ALL entities and their relationships

EXAMPLES:
(Alice, works_at, Acme Corp)
(Acme Corp, located_in, Berlin)
(Bob, is_a, engineer)

TEXT: {text}

TRIPLETS:"""

    @staticmethod
    def _parse_triplets(llm_response: str) -> List[Triplet]:
        triplets: List[Triplet] = []
        pattern = r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)"
        matches = re.findall(pattern, llm_response)
        for a, r, b in matches:
            a, r, b = a.strip(), r.strip(), b.strip()
            if all([a, r, b]) and all(len(x) < 200 for x in [a, r, b]):
                triplets.append(Triplet(a, r, b))
        return triplets

    # ───────────────────── Ingest Pipeline ─────────────────────
    def build_graph_from_documents(
        self,
        documents: List[Document],
        clear_existing: bool = False,
        store_embeddings: bool = True,
        batch_size: int = 500,
        backfill_names: bool = True,
    ) -> dict:
        if clear_existing:
            self._clear_neo4j()

        print(f"\n📊 Processing {len(documents)} documents...")

        all_triplets: List[Triplet] = []
        entity_context: Dict[str, List[str]] = {}

        # 1) Triplets extrahieren & Kontext sammeln
        for i, doc in enumerate(documents, 1):
            print(f"\n[{i}/{len(documents)}] Processing document...")
            text = doc.text or ""
            triplets = self.extract_triplets_from_text(text)
            all_triplets.extend(triplets)

            sentences = re.split(r"(?<=[.!?])\s+", text)
            for t in triplets:
                for ent in (t.subject, t.object):
                    picks = [s for s in sentences if ent in s][:2]
                    if picks:
                        entity_context.setdefault(ent, []).extend(picks)

        # 2) Einzigartige Entities
        entities = sorted({t.subject for t in all_triplets} | {t.object for t in all_triplets})
        print(f"\n🧮 Preparing {len(entities)} entities...")

        # 3) Embeddings (optional)
        entity_embeddings: Dict[str, List[float]] = {}
        if store_embeddings:
            print(f"🧮 Computing embeddings for {len(entities)} entities...")
            for i, ent in enumerate(entities, 1):
                entity_embeddings[ent] = self.embed_model.get_text_embedding(ent)
                if i % 10 == 0 or i == len(entities):
                    print(f"   Progress: {i}/{len(entities)}")
            print(f"✅ Embeddings computed (dim={len(next(iter(entity_embeddings.values())) or [])})")

        # 4) Entities upserten (inkl. name/title/summary/content)
        def _entity_rows(chunk: Iterable[str]) -> List[Dict]:
            rows = []
            for ent in chunk:
                contexts = " ".join(entity_context.get(ent, []))
                title = self._make_title(ent)  # für Entities aus Text ist der Name oft schon ok
                summary_source = contexts if contexts else f"{ent} – entity mentioned in corpus."
                summary = self._make_summary(summary_source)
                content = contexts[:2000] if contexts else None
                rows.append({
                    "id": ent,
                    "name": title,        # <<< für Anzeige in Browser/Bloom
                    "title": title,
                    "summary": summary,
                    "content": content,
                    "embedding": entity_embeddings.get(ent),
                })
            return rows

        upsert_entities_cypher = f"""
        UNWIND $batch AS row
        MERGE (e:{self.ENTITY_LABEL} {{ {self.ID_PROP}: row.id }})
        SET
          e.name    = coalesce(row.name, e.name, row.title, row.{self.ID_PROP}),
          e.title   = coalesce(row.title, e.title),
          e.summary = coalesce(row.summary, e.summary),
          e.content = coalesce(row.content, e.content)
        FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END |
          SET e.embedding = row.embedding
        )
        """

        with self.driver.session() as session:
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                rows = _entity_rows(batch)
                session.run(upsert_entities_cypher, batch=rows).consume()
                logger.info("Upserted entities %d/%d", min(i + batch_size, len(entities)), len(entities))

        # 5) Beziehungen schreiben (sicherer Rel-Type)
        def _chunks(lst: List[Triplet], n: int):
            for j in range(0, len(lst), n):
                yield lst[j:j + n]

        with self.driver.session() as session:
            for batch in _chunks(all_triplets, batch_size):
                safe_part = []
                fallback_part = []
                for t in batch:
                    rel = self._sanitize_rel_type(t.predicate)
                    rec = {"a": t.subject, "b": t.object, "type": t.predicate, "rel_type": rel}
                    if rel:
                        safe_part.append(rec)
                    else:
                        fallback_part.append(rec)

                if safe_part:
                    by_type: Dict[str, List[Dict]] = {}
                    for r in safe_part:
                        by_type.setdefault(r["rel_type"], []).append(r)

                    for rel_type, rows in by_type.items():
                        cypher = f"""
                        UNWIND $rows AS row
                        MATCH (a:{self.ENTITY_LABEL} {{ {self.ID_PROP}: row.a }})
                        MATCH (b:{self.ENTITY_LABEL} {{ {self.ID_PROP}: row.b }})
                        MERGE (a)-[r:{rel_type}]->(b)
                        """
                        session.run(cypher, rows=rows).consume()

                if fallback_part:
                    cypher_fallback = f"""
                    UNWIND $rows AS row
                    MATCH (a:{self.ENTITY_LABEL} {{ {self.ID_PROP}: row.a }})
                    MATCH (b:{self.ENTITY_LABEL} {{ {self.ID_PROP}: row.b }})
                    MERGE (a)-[r:RELATION {{type: row.type}}]->(b)
                    """
                    session.run(cypher_fallback, rows=fallback_part).consume()

        # 6) Optional: Backfill für alte Knoten ohne name
        if backfill_names:
            self.backfill_entity_names()

        stats = self._get_neo4j_stats()
        return {
            "documents_processed": len(documents),
            "triplets_extracted": len(all_triplets),
            "nodes_in_graph": stats["nodes"],
            "relationships_in_graph": stats["relationships"],
            "embeddings_stored": store_embeddings,
        }

    def backfill_entity_names(self) -> None:
        """Setzt e.name nachträglich sinnvoll, falls noch NULL."""
        cypher = f"""
        MATCH (e:{self.ENTITY_LABEL})
        WHERE e.name IS NULL OR e.name = ''
        SET e.name = coalesce(e.title, toString(e.{self.ID_PROP}))
        """
        try:
            with self.driver.session() as s:
                s.run(cypher).consume()
            logger.info("Backfilled missing e.name from title/id")
        except Neo4jError as e:
            logger.error("Backfill names failed: %s", e)

    # ───────────────────── Retrieval Helpers ─────────────────────
    def _index_online(self, name: str) -> bool:
        with self.driver.session() as s:
            rec = s.run(
                "SHOW INDEXES YIELD name, state WHERE name = $n RETURN state",
                n=name
            ).single()
            return bool(rec and rec.get("state") == "ONLINE")

    # Prefer Vector-Index; fallback zu Cosine
    def semantic_search(self, query: str, limit: int = 5) -> List[dict]:
        query_embedding = self.embed_model.get_query_embedding(query)

        if self._index_online(self.VECTOR_INDEX):
            try:
                with self.driver.session() as s:
                    rows = s.run(
                        f"""
                        CALL db.index.vector.queryNodes($index, $k, $vec)
                        YIELD node, score
                        RETURN node.{self.ID_PROP} AS entity, node.name AS name, score
                        ORDER BY score DESC
                        LIMIT $k
                        """,
                        index=self.VECTOR_INDEX,
                        k=limit,
                        vec=query_embedding,
                    ).data()
                return [{"entity": r["entity"], "name": r.get("name"), "similarity": float(r["score"])} for r in rows]
            except Neo4jError as e:
                logger.error("Vector index search failed, fallback to cosine: %s", e)

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (n:{self.ENTITY_LABEL})
                WHERE n.embedding IS NOT NULL
                RETURN n.{self.ID_PROP} AS entity, n.name AS name, n.embedding AS embedding
                """
            )
            scored: List[dict] = []
            for record in result:
                entity = record["entity"]
                name = record.get("name")
                embedding = record["embedding"]
                sim = self._cosine_similarity(query_embedding, embedding)
                scored.append({"entity": entity, "name": name, "similarity": sim})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:limit]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        m1 = math.sqrt(sum(a * a for a in vec1))
        m2 = math.sqrt(sum(b * b for b in vec2))
        if m1 == 0 or m2 == 0:
            return 0.0
        return dot_product / (m1 * m2)

    # ───────────────────── Admin / Utils ─────────────────────
    def _clear_neo4j(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
        print("🧹 Neo4j cleared")

    def _get_neo4j_stats(self) -> dict:
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        return {"nodes": node_count, "relationships": rel_count}

    def query_graph(self, query: str, limit: int = 10) -> List[dict]:
        with self.driver.session() as session:
            result = session.run(query + f" LIMIT {limit}")
            return [dict(record) for record in result]

    def get_sample_triplets(self, limit: int = 10) -> List[Triplet]:
        query = f"""
        MATCH (a:{self.ENTITY_LABEL})-[r]->(b:{self.ENTITY_LABEL})
        RETURN a.{self.ID_PROP} as subject, type(r) as predicate, b.{self.ID_PROP} as object
        """
        results = self.query_graph(query, limit=limit)
        return [Triplet(r["subject"], r["predicate"], r["object"]) for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# Main (Smoke Test)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Example usage with real embeddings + ingest best practices + readable names."""
    load_dotenv()

    print("=" * 60)
    print("🚀 Production KG Extractor with Real Embeddings (Best Practices Ingest + Names)")
    print("=" * 60)

    extractor = KnowledgeGraphExtractor(
        ollama_host=os.getenv("OLLAMA_HOST", "http://test.ki-plattform.apps.gisamgmt.global/"),
        ollama_api_key=os.getenv("OLLAMA_API_KEY", ""),
        ollama_model=os.getenv("OLLAMA_MODEL", ""),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b-q8_0"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
    )

    documents = [
        Document(text="Alice is the CEO of Acme Corp located in Berlin."),
        Document(text="Acme Corp acquired Beta Ltd in 2022."),
        Document(text="Bob is a software engineer at Acme Corp."),
        Document(text="Beta Ltd specializes in AI and machine learning."),
        Document(text="Charlie works as a data scientist in Berlin."),
    ]

    stats = extractor.build_graph_from_documents(
        documents,
        clear_existing=True,
        store_embeddings=True,
        backfill_names=True,    # <<< stellt sicher, dass alte Knoten Namen bekommen
    )

    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key:30s}: {value}")

    print("\n" + "=" * 60)
    print("🔍 SAMPLE TRIPLETS")
    print("=" * 60)
    for i, t in enumerate(extractor.get_sample_triplets(limit=10), 1):
        print(f"  {i}. {t}")

    print("\n" + "=" * 60)
    print("🔎 SEMANTIC SEARCH DEMO (shows id + name)")
    print("=" * 60)
    for q in ["Who is the boss of the company?", "Software development and programming", "City in Germany"]:
        print(f"\nQuery: '{q}'")
        res = extractor.semantic_search(q, limit=3)
        for i, r in enumerate(res, 1):
            nm = f" [{r['name']}]" if r.get("name") else ""
            print(f"  {i}. {r['entity']:30s}{nm} (similarity: {r['similarity']:.3f})")

    print("\n" + "=" * 60)
    print("✅ Complete with readable node names!")
    print("=" * 60)


if __name__ == "__main__":
    main()
