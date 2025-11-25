#!/usr/bin/env python3
"""
Neo4j Wargaming Knowledge Graph Cleanup
=======================================
Post-Ingest Entity Resolution ohne Re-Ingestion.

Features:
- Synonym-basiertes Node-Merging
- Typo-Korrektur
- Case-Normalisierung
- Peripherie-Analyse

Usage:
    # Nur Cypher generieren (dry-run)
    python neo4j_graph_cleanup.py --dry-run --output cleanup_queries.cypher
    
    # Direkt auf Neo4j ausführen
    python neo4j_graph_cleanup.py --uri bolt://localhost:7687 --user neo4j --password <pw>
"""

import argparse
import json
from collections import defaultdict
from typing import Optional

# ============================================================================
# KONFIGURATION: Synonyme & Typos
# ============================================================================

# Aus deinem domain_rules.yaml + erkannte Probleme aus dem Graph
SYNONYM_MAPPINGS = {
    # === Wargaming Handbook Varianten (Typos!) ===
    "Vwargaming Handbook": "Wargaming Handbook",
    "Iiiwargaming Handbook": "Wargaming Handbook",
    "IIIWargaming Handbook": "Wargaming Handbook",
    
    # === Doctrine Centre Varianten ===
    "Concepts And Doctrine Centre": "Development, Concepts And Doctrine Centre",
    
    # === Decision Making ===
    "decision-making": "Decision Making",
    "decision making": "Decision Making",
    "Decision-Making": "Decision Making",
    "decisions": "Decision Making",
    "decision": "Decision Making",
    "judgement": "Decision Making",
    "judgment": "Decision Making",
    
    # === Red Cell / Red Team ===
    "red team": "Red Cell",
    "Red Team": "Red Cell",
    "red teaming": "Red Cell",
    "Red Teaming": "Red Cell",
    "adversary": "Red Cell",
    "enemy": "Red Cell",
    
    # === Training Audience ===
    "trainee": "Training Audience",
    "trainees": "Training Audience",
    "players": "Training Audience",
    "Players": "Training Audience",
    "player": "Training Audience",
    
    # === Wargaming/Wargame Normalisierung ===
    "wargame": "Wargaming",
    "Wargame": "Wargaming",
    "war game": "Wargaming",
    "War Game": "Wargaming",
    "wargames": "Wargaming",
    "Wargames": "Wargaming",
    
    # === Simulation ===
    "simulation": "Simulation",
    "simulations": "Simulation",
    
    # === Scenario ===
    "scenario": "Scenario",
    "scenarios": "Scenario",
    
    # === Control Team ===
    "white cell": "Control Team",
    "White Cell": "Control Team",
    "game director": "Control Team",
    "Game Director": "Control Team",
    
    # === Adjudication ===
    "adjudicated": "Adjudication",
    "adjudicator": "Adjudication",
    "adjudicators": "Adjudication",
    "umpire": "Adjudication",
    "Umpire": "Adjudication",
    
    # === Analysis ===
    "analysis": "Analysis",
    "Analysis Phase": "Analysis",
    
    # === Generische lowercase -> Title Case ===
    "training": "Training",
    "exercise": "Exercise",
    "planning": "Strategic Planning",
    "sponsor": "Sponsor",
    "variants": "Variants",
}

# Zusätzliche Typo-Patterns (Regex-basiert)
TYPO_PATTERNS = [
    # Römische Ziffern am Anfang entfernen (Iii, Vii, etc.)
    (r"^[IiVvXx]+([A-Z])", r"\1"),
]

# ============================================================================
# CYPHER QUERY GENERATORS
# ============================================================================

def generate_merge_query(source_name: str, target_name: str) -> str:
    """
    Generiert APOC-basierte Merge-Query.
    Alle Relationships von source werden auf target übertragen, dann source gelöscht.
    """
    return f"""
// Merge: "{source_name}" → "{target_name}"
MATCH (source:Entity {{name: "{source_name}"}})
MATCH (target:Entity {{name: "{target_name}"}})
WHERE source <> target
CALL apoc.refactor.mergeNodes([target, source], {{
    properties: "discard",
    mergeRels: true
}})
YIELD node
RETURN node.name AS merged_node;
"""


def generate_merge_query_no_apoc(source_name: str, target_name: str) -> str:
    """
    Fallback ohne APOC: Relationships manuell übertragen.
    """
    return f"""
// Merge (ohne APOC): "{source_name}" → "{target_name}"
// Step 1: Eingehende Relationships übertragen
MATCH (source:Entity {{name: "{source_name}"}})<-[r]-(other)
MATCH (target:Entity {{name: "{target_name}"}})
WHERE source <> target AND other <> target
CALL {{
    WITH r, other, target
    WITH type(r) AS relType, other, target
    MERGE (other)-[newRel:TRANSFERRED]->(target)
    // Note: Relationship-Typ kann nicht dynamisch gesetzt werden ohne APOC
}}
RETURN count(*) AS transferred_in;

// Step 2: Ausgehende Relationships übertragen
MATCH (source:Entity {{name: "{source_name}"}})-[r]->(other)
MATCH (target:Entity {{name: "{target_name}"}})
WHERE source <> target AND other <> target
CALL {{
    WITH r, other, target
    MERGE (target)-[:TRANSFERRED]->(other)
}}
RETURN count(*) AS transferred_out;

// Step 3: Source löschen
MATCH (source:Entity {{name: "{source_name}"}})
DETACH DELETE source;
"""


def generate_rename_query(old_name: str, new_name: str) -> str:
    """Einfaches Umbenennen eines Nodes."""
    return f"""
// Rename: "{old_name}" → "{new_name}"
MATCH (n:Entity {{name: "{old_name}"}})
SET n.name = "{new_name}",
    n.id = "{new_name}",
    n.caption = "{new_name}",
    n.title = "{new_name}"
RETURN n.name AS renamed;
"""


def generate_case_fix_query(lowercase_name: str) -> str:
    """Title Case für lowercase Nodes."""
    title_case = lowercase_name.title()
    return generate_rename_query(lowercase_name, title_case)


def generate_find_duplicates_query() -> str:
    """Query zum Finden potenzieller Duplikate."""
    return """
// Finde potenzielle Duplikate (case-insensitive)
MATCH (n:Entity)
WITH toLower(n.name) AS lowername, collect(n) AS nodes, count(*) AS cnt
WHERE cnt > 1
RETURN lowername, [n IN nodes | n.name] AS variants, cnt
ORDER BY cnt DESC;
"""


def generate_find_orphans_query() -> str:
    """Findet isolierte Knoten."""
    return """
// Finde Orphan-Nodes (keine Verbindungen)
MATCH (n:Entity)
WHERE NOT (n)--()
RETURN n.name AS orphan
ORDER BY n.name;
"""


def generate_find_weak_nodes_query() -> str:
    """Findet schwach verbundene Knoten (degree 1)."""
    return """
// Finde schwach verbundene Nodes (nur 1 Verbindung)
MATCH (n:Entity)
WITH n, size((n)--()) AS degree
WHERE degree = 1
RETURN n.name AS weak_node, degree
ORDER BY n.name;
"""


def generate_statistics_query() -> str:
    """Graph-Statistiken vor/nach Cleanup."""
    return """
// Graph Statistiken
MATCH (n:Entity)
WITH count(n) AS total_nodes
MATCH ()-[r]->()
WITH total_nodes, count(r) AS total_edges
MATCH (n:Entity)
WITH total_nodes, total_edges, n, size((n)--()) AS degree
RETURN 
    total_nodes,
    total_edges,
    avg(degree) AS avg_degree,
    max(degree) AS max_degree,
    min(degree) AS min_degree,
    count(CASE WHEN degree <= 1 THEN 1 END) AS weak_nodes,
    count(CASE WHEN degree = 0 THEN 1 END) AS orphan_nodes;
"""


# ============================================================================
# HAUPTLOGIK
# ============================================================================

class GraphCleanup:
    def __init__(self, use_apoc: bool = True):
        self.use_apoc = use_apoc
        self.queries = []
        self.stats = {
            "merges": 0,
            "renames": 0,
            "case_fixes": 0,
        }
    
    def add_synonym_merges(self):
        """Fügt Merge-Queries für alle Synonyme hinzu."""
        print("\n📦 Generiere Synonym-Merges...")
        
        # Gruppiere nach Ziel
        targets = defaultdict(list)
        for source, target in SYNONYM_MAPPINGS.items():
            if source != target:  # Keine Self-Merges
                targets[target].append(source)
        
        for target, sources in targets.items():
            for source in sources:
                if self.use_apoc:
                    self.queries.append(generate_merge_query(source, target))
                else:
                    self.queries.append(generate_merge_query_no_apoc(source, target))
                self.stats["merges"] += 1
                print(f"   → '{source}' ➔ '{target}'")
    
    def add_case_fixes(self, lowercase_entities: list[str]):
        """Fügt Case-Fix-Queries hinzu."""
        print("\n🔤 Generiere Case-Fixes...")
        
        for name in lowercase_entities:
            if name and name[0].islower():
                self.queries.append(generate_case_fix_query(name))
                self.stats["case_fixes"] += 1
                print(f"   → '{name}' ➔ '{name.title()}'")
    
    def add_analysis_queries(self):
        """Fügt Analyse-Queries hinzu (am Ende)."""
        self.queries.append("\n// " + "=" * 60)
        self.queries.append("// ANALYSE QUERIES (nach Cleanup ausführen)")
        self.queries.append("// " + "=" * 60)
        self.queries.append(generate_statistics_query())
        self.queries.append(generate_find_duplicates_query())
        self.queries.append(generate_find_orphans_query())
        self.queries.append(generate_find_weak_nodes_query())
    
    def generate_all(self, lowercase_entities: Optional[list[str]] = None) -> str:
        """Generiert alle Cleanup-Queries."""
        
        self.queries.append("// " + "=" * 60)
        self.queries.append("// WARGAMING KNOWLEDGE GRAPH CLEANUP")
        self.queries.append("// Generated by neo4j_graph_cleanup.py")
        self.queries.append("// " + "=" * 60)
        self.queries.append("")
        self.queries.append("// PRE-CLEANUP STATISTICS")
        self.queries.append(generate_statistics_query())
        self.queries.append("")
        
        # Synonym Merges
        self.queries.append("// " + "-" * 60)
        self.queries.append("// SYNONYM MERGES")
        self.queries.append("// " + "-" * 60)
        self.add_synonym_merges()
        
        # Case Fixes
        if lowercase_entities:
            self.queries.append("")
            self.queries.append("// " + "-" * 60)
            self.queries.append("// CASE NORMALIZATION")
            self.queries.append("// " + "-" * 60)
            self.add_case_fixes(lowercase_entities)
        
        # Analyse am Ende
        self.queries.append("")
        self.add_analysis_queries()
        
        return "\n".join(self.queries)
    
    def print_summary(self):
        """Druckt Zusammenfassung."""
        print("\n" + "=" * 60)
        print("📊 CLEANUP SUMMARY")
        print("=" * 60)
        print(f"   Merges geplant:     {self.stats['merges']}")
        print(f"   Case-Fixes geplant: {self.stats['case_fixes']}")
        print(f"   Total Queries:      {len([q for q in self.queries if q.strip() and not q.startswith('//')])}")


def execute_on_neo4j(queries: str, uri: str, user: str, password: str):
    """Führt Queries direkt auf Neo4j aus."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("❌ neo4j Python Driver nicht installiert!")
        print("   pip install neo4j")
        return False
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Queries splitten und ausführen
    query_list = [q.strip() for q in queries.split(";") if q.strip() and not q.strip().startswith("//")]
    
    print(f"\n🚀 Führe {len(query_list)} Queries auf {uri} aus...")
    
    with driver.session() as session:
        for i, query in enumerate(query_list, 1):
            if query.strip():
                try:
                    result = session.run(query)
                    summary = result.consume()
                    print(f"   [{i}/{len(query_list)}] ✅ {summary.counters}")
                except Exception as e:
                    print(f"   [{i}/{len(query_list)}] ❌ {e}")
    
    driver.close()
    print("\n✅ Cleanup abgeschlossen!")
    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Neo4j Wargaming Graph Cleanup - Entity Resolution ohne Re-Ingest"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Nur Queries generieren, nicht ausführen"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="cleanup_queries.cypher",
        help="Output-Datei für generierte Queries"
    )
    parser.add_argument(
        "--uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j URI"
    )
    parser.add_argument(
        "--user", "-u",
        type=str,
        default="neo4j",
        help="Neo4j Username"
    )
    parser.add_argument(
        "--password", "-p",
        type=str,
        help="Neo4j Password"
    )
    parser.add_argument(
        "--no-apoc",
        action="store_true",
        help="Generiere Queries ohne APOC (Fallback)"
    )
    parser.add_argument(
        "--graph-json",
        type=str,
        help="Optional: graph_clean.json für Analyse der lowercase Entities"
    )
    
    args = parser.parse_args()
    
    # Lowercase Entities aus JSON extrahieren (optional)
    lowercase_entities = []
    if args.graph_json:
        try:
            with open(args.graph_json, 'r') as f:
                data = json.load(f)
            for item in data:
                if 'p' in item:
                    for node in [item['p'].get('start'), item['p'].get('end')]:
                        if node:
                            name = node.get('properties', {}).get('name', '')
                            if name and name[0].islower():
                                lowercase_entities.append(name)
            lowercase_entities = list(set(lowercase_entities))
            print(f"📂 {len(lowercase_entities)} lowercase Entities aus JSON geladen")
        except Exception as e:
            print(f"⚠️  Konnte JSON nicht laden: {e}")
    
    # Cleanup generieren
    cleanup = GraphCleanup(use_apoc=not args.no_apoc)
    queries = cleanup.generate_all(lowercase_entities)
    cleanup.print_summary()
    
    # Output
    if args.dry_run or not args.password:
        # Nur in Datei schreiben
        with open(args.output, 'w') as f:
            f.write(queries)
        print(f"\n💾 Queries gespeichert in: {args.output}")
        print("\n📋 Nächste Schritte:")
        print(f"   1. Queries prüfen: cat {args.output}")
        print(f"   2. In Neo4j Browser ausführen, oder:")
        print(f"   3. python {__file__} --uri <uri> --user <user> --password <pw>")
    else:
        # Direkt ausführen
        execute_on_neo4j(queries, args.uri, args.user, args.password)


if __name__ == "__main__":
    main()
