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
    """Validates knowledge graph quality metrics"""

    def __init__(self, driver):
        self.driver = driver
        self.results = {}

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all validation checks"""
        print("\n" + "="*60)
        print("GRAPH QUALITY VALIDATION")
        print("="*60 + "\n")

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

        # Summary
        self._print_summary()

        return self.results

    def check_graph_density(self) -> Dict[str, Any]:
        """Check overall graph density"""
        query = """
        MATCH (n:Entity)
        WITH count(n) AS node_count
        MATCH ()-[r]-()
        WITH node_count, count(DISTINCT r) AS rel_count
        RETURN
            node_count,
            rel_count,
            rel_count * 1.0 / node_count AS avg_degree,
            CASE
                WHEN rel_count * 1.0 / node_count < 2 THEN 'VERY SPARSE'
                WHEN rel_count * 1.0 / node_count < 5 THEN 'SPARSE'
                WHEN rel_count * 1.0 / node_count < 10 THEN 'NORMAL'
                ELSE 'DENSE'
            END AS density_rating
        """

        with self.driver.session() as session:
            result = session.run(query).single()

            return {
                "nodes": result["node_count"],
                "relationships": result["rel_count"],
                "avg_degree": round(result["avg_degree"], 2),
                "density_rating": result["density_rating"],
                "target_avg_degree": 3.0,
                "target_rating": "NORMAL",
                "passed": result["avg_degree"] >= 3.0
            }

    def check_causal_relationships(self) -> Dict[str, Any]:
        """Check count of causal relationships"""
        causal_types = [
            'LEADS_TO', 'CAUSES', 'RESULTS_IN', 'PRODUCES', 'GENERATES',
            'ENABLES', 'IMPROVES', 'ENHANCES', 'FACILITATES', 'SUPPORTS'
        ]

        query = """
        MATCH ()-[r]-()
        WHERE type(r) IN $causal_types
        RETURN type(r) as rel_type, count(*) as count
        ORDER BY count DESC
        """

        with self.driver.session() as session:
            results = session.run(query, causal_types=causal_types).data()

            total_causal = sum(r["count"] for r in results)

            return {
                "total_causal_relationships": total_causal,
                "causal_types_found": len(results),
                "breakdown": results,
                "target_min": 30,
                "passed": total_causal >= 30
            }

    def check_entity_connectivity(self) -> Dict[str, Any]:
        """Check entity connectivity distribution"""
        query = """
        MATCH (n:Entity)
        WITH n, COUNT { (n)-[]-() } AS degree
        WITH degree, count(*) as entity_count
        ORDER BY degree
        RETURN degree, entity_count
        """

        with self.driver.session() as session:
            results = session.run(query).data()

            total_entities = sum(r["entity_count"] for r in results)
            leaf_nodes = next(
                (r["entity_count"] for r in results if r["degree"] == 1),
                0
            )

            leaf_percentage = (leaf_nodes / total_entities * 100) if total_entities > 0 else 0

            return {
                "total_entities": total_entities,
                "leaf_nodes": leaf_nodes,
                "leaf_percentage": round(leaf_percentage, 1),
                "distribution": results,
                "target_leaf_percentage": 30.0,
                "passed": leaf_percentage < 30.0
            }

    def check_ai_connectivity(self) -> Dict[str, Any]:
        """Check AI entity connectivity"""
        query = """
        MATCH (ai:Entity)
        WHERE toLower(ai.name) CONTAINS 'artificial intelligence'
           OR toLower(ai.name) = 'ai'
        WITH ai, COUNT { (ai)-[]-() } AS degree
        RETURN ai.name as name, ai.id as id, degree
        ORDER BY degree DESC
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query).single()

            if result:
                return {
                    "ai_entity": result["name"],
                    "ai_id": result["id"],
                    "degree": result["degree"],
                    "target_min_degree": 5,
                    "passed": result["degree"] >= 5
                }
            else:
                return {
                    "ai_entity": None,
                    "error": "No AI entity found in graph",
                    "passed": False
                }

    def check_isolated_entities(self) -> Dict[str, Any]:
        """Check for isolated entities with no relationships"""
        query = """
        MATCH (n:Entity)
        WHERE NOT EXISTS { (n)-[]-() }
        RETURN count(*) AS isolated_count
        """

        with self.driver.session() as session:
            result = session.run(query).single()

            return {
                "isolated_count": result["isolated_count"],
                "target": 0,
                "passed": result["isolated_count"] == 0
            }

    def check_multihop_paths(self) -> Dict[str, Any]:
        """Check for multihop paths from AI to coordination"""
        query = """
        MATCH (ai:Entity)
        WHERE toLower(ai.name) CONTAINS 'artificial intelligence'
           OR toLower(ai.name) = 'ai'
        WITH ai
        MATCH (coord:Entity)
        WHERE toLower(coord.name) CONTAINS 'coordination'
        WITH ai, coord
        MATCH path = (ai)-[*1..5]-(coord)
        RETURN
            length(path) as hops,
            [n IN nodes(path) | n.name] AS entities,
            [r IN relationships(path) | type(r)] AS relationships
        ORDER BY hops
        LIMIT 10
        """

        with self.driver.session() as session:
            results = session.run(query).data()

            return {
                "paths_found": len(results),
                "paths": results[:5],  # First 5 paths
                "target_min_paths": 2,
                "passed": len(results) >= 2
            }

    def _print_result(self, result: Dict[str, Any]):
        """Pretty print validation result"""
        if "error" in result:
            print(f"❌ ERROR: {result['error']}")
            return

        # Print all key metrics
        for key, value in result.items():
            if key in ["passed", "breakdown", "distribution", "paths"]:
                continue

            if isinstance(value, (int, float, str)):
                print(f"  {key}: {value}")

        # Print pass/fail
        if "passed" in result:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"\n  Status: {status}")

    def _print_summary(self):
        """Print overall summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60 + "\n")

        total_checks = len(self.results)
        passed_checks = sum(
            1 for r in self.results.values()
            if isinstance(r, dict) and r.get("passed", False)
        )

        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Pass Rate: {passed_checks/total_checks*100:.1f}%")

        if passed_checks == total_checks:
            print("\n🎉 ALL CHECKS PASSED! Graph quality is excellent.")
        elif passed_checks >= total_checks * 0.7:
            print("\n⚠️  MOST CHECKS PASSED. Graph quality is acceptable but can be improved.")
        else:
            print("\n❌ MANY CHECKS FAILED. Graph quality needs improvement.")
            print("   → Consider re-ingesting with improved extraction configuration.")

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
