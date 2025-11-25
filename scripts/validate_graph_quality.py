#!/usr/bin/env python3
"""
Graph Quality Validation Script
=================================

Validates the quality of the knowledge graph after re-ingestion.
Checks density, connectivity, causal relationships, and multihop paths.

Usage:
    python scripts/validate_graph_quality.py

Requirements:
    - Neo4j database running
    - Valid .env configuration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import GraphDatabase
from config.settings import AppConfig
from typing import Dict, List, Any
import json


class GraphQualityValidator:
    def __init__(self, driver):
        self.driver = driver
        self.results = {}
        # Wir ermitteln erst die Größe, um die Lattenhöhe festzulegen
        self.total_nodes = 0 

    def run_all_checks(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("GRAPH QUALITY VALIDATION (DYNAMIC MODE)")
        print("="*60 + "\n")

        # Vorab Node-Count holen für dynamische Schwellwerte
        with self.driver.session() as session:
            self.total_nodes = session.run("MATCH (n:Entity) RETURN count(n) as c").single()["c"]
        
        print(f"📊 Graph Size: {self.total_nodes} nodes detected.")
        if self.total_nodes < 50:
            print("⚠️  Small Graph detected -> Adjusting thresholds downwards.\n")

        checks = [
            ("Graph Density", self.check_graph_density),
            ("Causal Relationships", self.check_causal_relationships),
            ("Entity Connectivity", self.check_entity_connectivity),
            ("AI Connectivity", self.check_ai_connectivity),
            ("Isolated Entities", self.check_isolated_entities),
            ("Multihop Paths", self.check_multihop_paths),
        ]

        for name, check_fn in checks:
            print(f"\n{'='*60}")
            print(f"CHECK: {name}")
            print(f"{'='*60}\n")
            try:
                result = check_fn()
                self.results[name] = result
                self._print_result(result)
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                self.results[name] = {"error": str(e)}

        self._print_summary()
        return self.results

    def _get_dynamic_target(self, large_target, small_target):
        """Wählt Ziel basierend auf Graphgröße"""
        return small_target if self.total_nodes < 50 else large_target

    def check_graph_density(self) -> Dict[str, Any]:
        query = """
        MATCH (n:Entity) WITH count(n) AS node_count
        MATCH ()-[r]-() WITH node_count, count(DISTINCT r) AS rel_count
        RETURN node_count, rel_count, 
               CASE WHEN node_count > 0 THEN rel_count * 1.0 / node_count ELSE 0 END AS avg_degree
        """
        with self.driver.session() as session:
            result = session.run(query).single()
            
        avg_degree = result["avg_degree"]
        # Dynamisches Ziel: 1.2 für kleine Graphen, 3.0 für große
        target_degree = self._get_dynamic_target(3.0, 1.2)
        
        rating = "DENSE" if avg_degree >= target_degree else "SPARSE"

        return {
            "nodes": result["node_count"],
            "relationships": result["rel_count"],
            "avg_degree": round(avg_degree, 2),
            "density_rating": rating,
            "target_avg_degree": target_degree,
            "passed": avg_degree >= target_degree
        }

    def check_causal_relationships(self) -> Dict[str, Any]:
        causal_types = ['LEADS_TO', 'CAUSES', 'RESULTS_IN', 'ENABLES', 'IMPROVES', 'AFFECTS', 'MITIGATES', 'REQUIRES']
        query = """
        MATCH ()-[r]-() WHERE type(r) IN $causal_types
        RETURN count(*) as count
        """
        with self.driver.session() as session:
            count = session.run(query, causal_types=causal_types).single()["count"]

        # Dynamisches Ziel: 5 für kleine Texte, 30 für große
        target_min = self._get_dynamic_target(30, 5)

        return {
            "total_causal": count,
            "target_min": target_min,
            "passed": count >= target_min
        }

    def check_entity_connectivity(self) -> Dict[str, Any]:
        """Leaf Node Percentage"""
        query = """
        MATCH (n:Entity)
        WITH n, COUNT { (n)-[]-() } AS degree
        WITH count(*) as total, sum(CASE WHEN degree = 1 THEN 1 ELSE 0 END) as leaves
        RETURN total, leaves
        """
        with self.driver.session() as session:
            res = session.run(query).single()
            
        leaf_pct = (res["leaves"] / res["total"] * 100) if res["total"] > 0 else 0
        
        # Kleine Graphen haben natürlich mehr Ränder (Leaves). 
        # Erlaube 60% bei kleinen Graphen, fordere <30% bei großen.
        target_max_pct = self._get_dynamic_target(30.0, 60.0)

        return {
            "leaf_percentage": round(leaf_pct, 1),
            "target_max_percentage": target_max_pct,
            "passed": leaf_pct <= target_max_pct
        }

    def check_ai_connectivity(self) -> Dict[str, Any]:
        query = """
        MATCH (n:Entity) 
        WHERE toLower(n.name) CONTAINS 'artificial intelligence' OR toLower(n.name) CONTAINS 'ai'
        RETURN count{(n)-[]-()} as degree, n.name as name LIMIT 1
        """
        with self.driver.session() as session:
            rec = session.run(query).single()

        if not rec: 
            return {"passed": False, "error": "AI node not found"}

        degree = rec["degree"]
        # Ziel: 2 Verbindungen für kleine Graphen reichen völlig
        target = self._get_dynamic_target(5, 2) 

        return {
            "ai_entity": rec["name"],
            "degree": degree,
            "target": target,
            "passed": degree >= target
        }

    def check_isolated_entities(self) -> Dict[str, Any]:
        # Bleibt gleich, Isolation ist immer schlecht
        query = "MATCH (n:Entity) WHERE NOT EXISTS { (n)-[]-() } RETURN count(*) as c"
        with self.driver.session() as session:
            count = session.run(query).single()["c"]
        return {"isolated": count, "passed": count == 0}

    def check_multihop_paths(self) -> Dict[str, Any]:
        # Prüft ob wir von AI irgendwo anders hinkommen (z.B. coordination)
        query = """
        MATCH (a:Entity), (b:Entity)
        WHERE toLower(a.name) CONTAINS 'artificial' AND toLower(b.name) CONTAINS 'coordination'
        MATCH p = shortestPath((a)-[*..5]-(b))
        RETURN length(p) as hops
        """
        with self.driver.session() as session:
            rec = session.run(query).single()
        
        found = rec is not None
        return {"path_found": found, "passed": found}

    def _print_result(self, res):
        status = "✅ PASSED" if res.get("passed") else "❌ FAILED"
        print(f"  Result: {json.dumps(res, indent=2)}")
        print(f"  Status: {status}")

    def _print_summary(self):
        passed = sum(1 for r in self.results.values() if r.get("passed"))
        total = len(self.results)
        print(f"\nSUMMARY: {passed}/{total} Passed")

    def export_results(self, filepath: str = "validation_results.json"):
        """Export results to JSON file"""
        output_path = project_root / filepath

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Results exported to: {output_path}")


def main():
    """Main validation workflow"""
    # Load config
    config = AppConfig.from_env()

    # Connect to Neo4j
    print(f"Connecting to Neo4j at {config.neo4j.uri}...")
    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.user, config.neo4j.password)
    )

    try:
        # Verify connection
        with driver.session() as session:
            session.run("RETURN 1")

        print("✅ Connected to Neo4j\n")

        # Run validation
        validator = GraphQualityValidator(driver)
        results = validator.run_all_checks()

        # Export results
        validator.export_results("validation_results.json")

    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        driver.close()


if __name__ == "__main__":
    main()
