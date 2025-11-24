#!/usr/bin/env python3
"""
Model Comparison Script for Knowledge Graph Extraction
========================================================

Tests different LLM models for KG extraction quality.
Compares relationship extraction performance.

Usage:
    python scripts/compare_extraction_models.py

Requirements:
    - Ollama running with test models
    - Valid .env configuration
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_index.llms.ollama import Ollama
from ollama import Client
from src.extractors.kg_extractor import KnowledgeGraphExtractor
from config.settings import AppConfig


class ModelComparer:
    """Compare extraction quality across different models"""

    # Test text with rich relationship potential
    TEST_TEXT = """
    NATO uses artificial intelligence in wargaming exercises to improve coordination
    between allied units. AI-powered scenario design enables realistic testing of
    command structures. Virtual reality technology enhances immersive training
    experiences for military personnel. These wargaming methodologies support strategic
    planning by identifying gaps in coordination protocols. Advanced simulation tools
    facilitate decision-making under complex operational scenarios.
    """

    # Models to test
    MODELS_TO_TEST = [
        {
            "name": "mistral-small",
            "model_id": "mistral-small3.2:24b-instruct-2506-q8_0",
            "description": "Fast, good for simple extraction"
        },
        {
            "name": "qwen3-14b",
            "model_id": "qwen3:14b",
            "description": "Balanced speed/quality"
        },
        {
            "name": "qwen3-32b",
            "model_id": "qwen3:32b",
            "description": "High quality, slower"
        },
    ]

    def __init__(self, config: AppConfig):
        self.config = config
        self.results = {}

    def test_model(self, model_info: Dict[str, str]) -> Dict[str, Any]:
        """Test a single model's extraction performance"""
        print(f"\n{'='*60}")
        print(f"Testing: {model_info['name']}")
        print(f"Model ID: {model_info['model_id']}")
        print(f"{'='*60}\n")

        try:
            # Initialize LLM
            ollama_client = Client(
                host=self.config.ollama.host,
                headers={"Authorization": f"Bearer {self.config.ollama.api_key}"}
            )

            llm = Ollama(
                model=model_info['model_id'],
                base_url=self.config.ollama.host,
                request_timeout=180,
                client=ollama_client,
            )

            # Create extractor
            extractor = KnowledgeGraphExtractor(llm=llm)

            # Extract triplets
            print("🔄 Extracting triplets...")
            triplets = extractor.extract_triplets_from_text(
                self.TEST_TEXT,
                verbose=False
            )

            # Analyze results
            analysis = self._analyze_triplets(triplets)

            print(f"\n📊 Results:")
            print(f"   Total Relationships: {analysis['total_relationships']}")
            print(f"   Unique Entities: {analysis['unique_entities']}")
            print(f"   Avg Relationships per Entity: {analysis['avg_relationships_per_entity']:.2f}")
            print(f"   Causal Relationships: {analysis['causal_count']}")
            print(f"   Relationship Types: {analysis['relationship_types']}")

            # Sample triplets
            print(f"\n🔍 Sample Triplets (first 5):")
            for i, t in enumerate(triplets[:5], 1):
                print(f"   {i}. {t}")

            return {
                "model": model_info['name'],
                "model_id": model_info['model_id'],
                "description": model_info['description'],
                "success": True,
                "triplets": [str(t) for t in triplets],
                "analysis": analysis
            }

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return {
                "model": model_info['name'],
                "model_id": model_info['model_id'],
                "description": model_info['description'],
                "success": False,
                "error": str(e)
            }

    def _analyze_triplets(self, triplets: List) -> Dict[str, Any]:
        """Analyze triplet quality"""
        entities = set()
        relationship_types = set()
        causal_types = {
            'LEADS_TO', 'CAUSES', 'RESULTS_IN', 'PRODUCES', 'GENERATES',
            'ENABLES', 'IMPROVES', 'ENHANCES', 'FACILITATES', 'SUPPORTS'
        }
        causal_count = 0

        for t in triplets:
            entities.add(t.subject)
            entities.add(t.object)
            relationship_types.add(t.predicate)

            if t.predicate.upper() in causal_types:
                causal_count += 1

        return {
            "total_relationships": len(triplets),
            "unique_entities": len(entities),
            "relationship_types": len(relationship_types),
            "avg_relationships_per_entity": len(triplets) / len(entities) if entities else 0,
            "causal_count": causal_count,
            "causal_percentage": (causal_count / len(triplets) * 100) if triplets else 0,
            "relationship_type_list": list(relationship_types)
        }

    def run_comparison(self) -> Dict[str, Any]:
        """Run comparison across all models"""
        print("\n" + "="*60)
        print("🧪 MODEL COMPARISON TEST")
        print("="*60)

        print("\n📝 Test Text:")
        print(f"   {self.TEST_TEXT[:100]}...")
        print(f"   (Total length: {len(self.TEST_TEXT)} chars)")

        # Test each model
        for model_info in self.MODELS_TO_TEST:
            result = self.test_model(model_info)
            self.results[model_info['name']] = result

        # Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print comparison summary"""
        print("\n" + "="*60)
        print("📊 COMPARISON SUMMARY")
        print("="*60)

        # Extract successful results
        successful = {k: v for k, v in self.results.items() if v.get('success')}

        if not successful:
            print("\n❌ No successful extractions!")
            return

        # Sort by total relationships
        sorted_results = sorted(
            successful.items(),
            key=lambda x: x[1]['analysis']['total_relationships'],
            reverse=True
        )

        print("\n🏆 Rankings (by total relationships):")
        for i, (name, result) in enumerate(sorted_results, 1):
            analysis = result['analysis']
            print(f"\n{i}. {result['model']} ({result['model_id']})")
            print(f"   Total Relationships: {analysis['total_relationships']}")
            print(f"   Causal Relationships: {analysis['causal_count']} ({analysis['causal_percentage']:.1f}%)")
            print(f"   Relationship Types: {analysis['relationship_types']}")
            print(f"   Avg per Entity: {analysis['avg_relationships_per_entity']:.2f}")

        # Recommendation
        print("\n" + "="*60)
        print("💡 RECOMMENDATION")
        print("="*60)

        winner = sorted_results[0]
        winner_analysis = winner[1]['analysis']

        if winner_analysis['total_relationships'] >= 15 and winner_analysis['causal_count'] >= 5:
            print(f"\n✅ {winner[0]} delivers EXCELLENT extraction quality!")
            print(f"   Recommended for production use.")
        elif winner_analysis['total_relationships'] >= 10:
            print(f"\n⚠️  {winner[0]} delivers ACCEPTABLE quality.")
            print(f"   Consider using for speed, but quality could be improved.")
        else:
            print(f"\n❌ ALL models show POOR extraction quality!")
            print(f"   Issues:")
            print(f"   - Total relationships too low (target: >15)")
            print(f"   - Causal relationships too low (target: >5)")
            print(f"   Consider:")
            print(f"   1. Testing stronger models (e.g., qwen3:70b)")
            print(f"   2. Improving extraction prompts")
            print(f"   3. Using richer test text")

    def export_results(self, filepath: str = "model_comparison_results.json"):
        """Export results to JSON"""
        output_path = project_root / filepath

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Results exported to: {output_path}")


def main():
    """Main comparison workflow"""
    # Load config
    config = AppConfig.from_env()

    print(f"\n🔧 Configuration:")
    print(f"   Ollama Host: {config.ollama.host}")

    # Run comparison
    comparer = ModelComparer(config)
    results = comparer.run_comparison()

    # Export results
    comparer.export_results()

    print("\n" + "="*60)
    print("✅ Comparison complete!")
    print("="*60)


if __name__ == "__main__":
    main()
