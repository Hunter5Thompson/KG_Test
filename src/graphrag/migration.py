"""
Neo4j Schema Migration für GraphRAG
Fügt Document-Linking und Indizes hinzu
"""
from neo4j import GraphDatabase
from typing import Dict
from config.settings import Neo4jConfig


class GraphRAGMigration:
    """Migriert bestehenden KG für GraphRAG-Optimierung"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def run_migration(self, dry_run: bool = True):
        """Führe Migration durch"""
        print("=" * 60)
        print("GraphRAG Schema Migration")
        print("=" * 60)
        
        # 1. Pre-Migration Analysis
        print("\n📊 Pre-Migration Status:")
        stats = self.get_current_stats()
        self._print_stats(stats)
        
        if dry_run:
            print("\n⚠️  DRY RUN MODE")
            print("\nGeplante Änderungen:")
            print("  1. Vector Index auf Entity.embedding")
            print("  2. Fulltext Index auf Entity.id")
            print("  3. Property Index auf Entity.id")
            return
        
        # 2. Create Indexes
        print("\n🔧 Erstelle Indizes...")
        self.create_vector_index()
        self.create_fulltext_index()
        self.create_property_indexes()
        
        # 3. Post-Migration
        print("\n📊 Post-Migration Status:")
        stats_after = self.get_current_stats()
        self._print_stats(stats_after)
        
        print("\n✅ Migration erfolgreich!")
    
    def get_current_stats(self) -> Dict:
        """Sammle Graph-Statistiken"""
        with self.driver.session(database=self.database) as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            entity_stats = session.run("""
                MATCH (e:Entity)
                RETURN count(e) as entity_count,
                       count(e.embedding) as entities_with_embeddings
            """).single()
            
            rel_types = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """).data()
            
            isolated = session.run("""
                MATCH (e:Entity)
                WHERE NOT (e)-[]->() AND NOT ()-[]->(e)
                RETURN count(e) as isolated_count
            """).single()["isolated_count"]
            
            indexes = session.run("SHOW INDEXES").data()
            
            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "entities": entity_stats["entity_count"] if entity_stats else 0,
                "entities_with_embeddings": entity_stats["entities_with_embeddings"] if entity_stats else 0,
                "isolated_entities": isolated,
                "top_relationship_types": rel_types,
                "existing_indexes": len(indexes)
            }
    
    def create_vector_index(self):
        """Erstelle Vector Index für Entity Embeddings"""
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.embedding IS NOT NULL
                    RETURN size(e.embedding) as dim
                    LIMIT 1
                """).single()
                
                if not result:
                    print("   ⚠️  Keine Embeddings - Index übersprungen")
                    return
                
                dim = result["dim"]
                print(f"   Embedding Dimension: {dim}")
                
                session.run(f"""
                    CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
                    FOR (e:Entity)
                    ON e.embedding
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                print("   ✅ Vector Index: entity_vector_index")
            except Exception as e:
                print(f"   ⚠️  Vector Index Fehler: {e}")
    
    def create_fulltext_index(self):
        """Erstelle Fulltext Index"""
        with self.driver.session(database=self.database) as session:
            try:
                session.run("""
                    CREATE FULLTEXT INDEX entity_fulltext_index IF NOT EXISTS
                    FOR (e:Entity)
                    ON EACH [e.id]
                """)
                print("   ✅ Fulltext Index: entity_fulltext_index")
            except Exception as e:
                print(f"   ⚠️  Fulltext Error: {e}")
    
    def create_property_indexes(self):
        """Erstelle Property Indizes"""
        with self.driver.session(database=self.database) as session:
            try:
                session.run("""
                    CREATE INDEX entity_id_index IF NOT EXISTS
                    FOR (e:Entity)
                    ON (e.id)
                """)
                print("   ✅ Property Index: entity_id_index")
            except Exception as e:
                print(f"   ⚠️  Property Error: {e}")
    
    def validate_graph_quality(self) -> Dict:
        """Prüfe Graph-Qualität"""
        with self.driver.session(database=self.database) as session:
            density = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]->()
                WITH e, count(r) as out_degree
                RETURN 
                    avg(out_degree) as avg_out_degree,
                    max(out_degree) as max_out_degree
            """).single()
            
            hubs = session.run("""
                MATCH (e:Entity)-[r]->()
                WITH e, count(r) as degree
                WHERE degree > 5
                RETURN e.id as entity, degree
                ORDER BY degree DESC
                LIMIT 10
            """).data()
            
            return {
                "avg_relationships_per_entity": density["avg_out_degree"],
                "max_relationships": density["max_out_degree"],
                "hub_entities": hubs
            }
    
    def _print_stats(self, stats: Dict):
        """Pretty-print Statistiken"""
        print(f"   Nodes: {stats['total_nodes']}")
        print(f"   Relationships: {stats['total_relationships']}")
        print(f"   Entities: {stats['entities']}")
        print(f"   Mit Embeddings: {stats['entities_with_embeddings']}")
        print(f"   Isolated: {stats['isolated_entities']}")
        print(f"   Indizes: {stats['existing_indexes']}")
        
        if stats['top_relationship_types']:
            print(f"\n   Top Relations:")
            for rel in stats['top_relationship_types'][:5]:
                print(f"     • {rel['rel_type']}: {rel['count']}")
    
    def close(self):
        self.driver.close()


def main():
    import argparse
    from config.settings import AppConfig
    
    parser = argparse.ArgumentParser(description="GraphRAG Migration")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    
    config = AppConfig.from_env()
    migration = GraphRAGMigration(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password
    )
    
    try:
        if args.validate:
            print("\n🔍 Quality Check:")
            quality = migration.validate_graph_quality()
            print(f"   Avg Relations/Entity: {quality['avg_relationships_per_entity']:.2f}")
            print(f"   Max Relations: {quality['max_relationships']}")
            if quality['hub_entities']:
                print(f"\n   Top Hubs:")
                for hub in quality['hub_entities']:
                    print(f"     • {hub['entity']}: {hub['degree']}")
        else:
            migration.run_migration(dry_run=args.dry_run)
    finally:
        migration.close()


if __name__ == "__main__":
    main()