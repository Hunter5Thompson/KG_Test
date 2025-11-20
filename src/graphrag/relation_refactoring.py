"""
Neo4j Relation Type Refactoring
Konvertiert RELATION {type: "..."} → Native Relation Types
"""
from neo4j import GraphDatabase
from typing import Dict, List
from tqdm import tqdm


class RelationRefactoring:
    """
    Refactored RELATION Relationships zu nativen Types
    
    WARNUNG: Dies ist eine destructive Operation!
    Backup empfohlen vor Ausführung.
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def analyze_relations(self) -> Dict:
        """Analysiere aktuelle Relation-Struktur"""
        with self.driver.session(database=self.database) as session:
            # Alle Relation-Types
            types = session.run("""
                MATCH ()-[r:RELATION]->()
                RETURN r.type as rel_type, count(*) as count
                ORDER BY count DESC
            """).data()
            
            # Relations ohne Type-Property
            no_type = session.run("""
                MATCH ()-[r:RELATION]->()
                WHERE r.type IS NULL
                RETURN count(r) as count
            """).single()["count"]
            
            # Total RELATION count
            total = session.run("""
                MATCH ()-[r:RELATION]->()
                RETURN count(r) as count
            """).single()["count"]
            
            return {
                "total_relations": total,
                "relations_without_type": no_type,
                "relation_types": types
            }
    
    def run_refactoring(self, dry_run: bool = True, batch_size: int = 100):
        """
        Führe Refactoring durch
        
        Args:
            dry_run: Wenn True, keine Änderungen
            batch_size: Relations pro Batch (für große Graphen)
        """
        print("=" * 60)
        print("Relation Type Refactoring")
        print("=" * 60)
        
        # 1. Analyse
        print("\n📊 Analyse:")
        analysis = self.analyze_relations()
        
        print(f"   Total RELATION: {analysis['total_relations']}")
        print(f"   Ohne Type-Property: {analysis['relations_without_type']}")
        print(f"   Unique Types: {len(analysis['relation_types'])}")
        
        if analysis['relation_types']:
            print(f"\n   Top 10 Types:")
            for rel in analysis['relation_types'][:10]:
                print(f"     • {rel['rel_type']}: {rel['count']}")
        
        if dry_run:
            print("\n⚠️  DRY RUN MODE - Keine Änderungen")
            print("\nGeplante Änderungen:")
            print(f"  1. Konvertiere {analysis['total_relations']} RELATION Edges")
            print(f"  2. Erstelle {len(analysis['relation_types'])} native Relation-Types")
            print(f"  3. Lösche alte RELATION Edges")
            
            if analysis['relations_without_type'] > 0:
                print(f"\n⚠️  WARNUNG: {analysis['relations_without_type']} Relations haben kein r.type Property!")
                print("     Diese würden übersprungen werden.")
            
            return
        
        # 2. Backup-Warnung
        print("\n⚠️  WICHTIG: Backup empfohlen!")
        print("   Diese Operation löscht alle [:RELATION] Edges.")
        response = input("\n   Fortfahren? (yes/no): ")
        
        if response.lower() != "yes":
            print("   Abgebrochen.")
            return
        
        # 3. Refactoring
        print("\n🔧 Starte Refactoring...")
        
        with self.driver.session(database=self.database) as session:
            # Get all relation types
            rel_types = [r['rel_type'] for r in analysis['relation_types'] if r['rel_type']]
            
            total_converted = 0
            for rel_type in tqdm(rel_types, desc="Converting"):
                # Batch-Processing für große Graphen
                converted = self._convert_relation_type(session, rel_type, batch_size)
                total_converted += converted
            
            print(f"\n✅ Konvertiert: {total_converted} Relations")
            
            # 4. Cleanup - Lösche alte RELATION Edges
            print("\n🧹 Cleanup alte RELATION Edges...")
            deleted = session.run("""
                MATCH ()-[r:RELATION]->()
                DELETE r
                RETURN count(r) as deleted
            """).single()["deleted"]
            
            print(f"   Gelöscht: {deleted} alte Relations")
        
        # 5. Verification
        print("\n📊 Verification:")
        verification = self._verify_refactoring()
        print(f"   Neue Relations: {verification['new_relations']}")
        print(f"   Verbleibende RELATION: {verification['remaining_old_relations']}")
        
        if verification['remaining_old_relations'] == 0:
            print("\n✅ Refactoring erfolgreich!")
        else:
            print(f"\n⚠️  {verification['remaining_old_relations']} RELATION Edges verblieben")
    
    def _convert_relation_type(self, session, rel_type: str, batch_size: int) -> int:
        """
        Konvertiere einen Relation-Type
        
        Returns:
            Anzahl konvertierter Relations
        """
        # Sanitize relation type für Cypher
        safe_rel_type = self._sanitize_relation_type(rel_type)
        
        # Batch-Processing
        offset = 0
        total = 0
        
        while True:
            result = session.run(f"""
                MATCH (a)-[old:RELATION]->(b)
                WHERE old.type = $rel_type
                WITH a, b, old
                SKIP $offset LIMIT $batch_size
                
                CREATE (a)-[new:`{safe_rel_type}`]->(b)
                SET new = old
                REMOVE new.type
                
                RETURN count(new) as created
            """, rel_type=rel_type, offset=offset, batch_size=batch_size).single()
            
            created = result["created"]
            total += created
            
            if created < batch_size:
                break
            
            offset += batch_size
        
        return total
    
    def _sanitize_relation_type(self, rel_type: str) -> str:
        """
        Sanitize Relation-Type für Cypher
        
        Neo4j Relation-Types müssen valid identifiers sein:
        - Keine Spaces → Underscores
        - Keine Sonderzeichen
        - UPPERCASE für Konvention
        """
        # Replace spaces and special chars
        sanitized = rel_type.replace(" ", "_").replace("-", "_")
        
        # Remove invalid characters
        sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
        
        # Uppercase (Neo4j Konvention)
        sanitized = sanitized.upper()
        
        return sanitized
    
    def _verify_refactoring(self) -> Dict:
        """Verifiziere Refactoring"""
        with self.driver.session(database=self.database) as session:
            # Count new relations (all non-RELATION types)
            new_rels = session.run("""
                MATCH ()-[r]->()
                WHERE NOT type(r) = 'RELATION'
                RETURN count(r) as count
            """).single()["count"]
            
            # Count remaining RELATION
            old_rels = session.run("""
                MATCH ()-[r:RELATION]->()
                RETURN count(r) as count
            """).single()["count"]
            
            return {
                "new_relations": new_rels,
                "remaining_old_relations": old_rels
            }
    
    def close(self):
        self.driver.close()


def main():
    import argparse
    from config.settings import AppConfig
    
    parser = argparse.ArgumentParser(description="Relation Type Refactoring")
    parser.add_argument("--dry-run", action="store_true", help="Analyse ohne Änderungen")
    parser.add_argument("--batch-size", type=int, default=100, help="Relations per batch")
    args = parser.parse_args()
    
    config = AppConfig.from_env()
    
    refactoring = RelationRefactoring(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password
    )
    
    try:
        refactoring.run_refactoring(dry_run=args.dry_run, batch_size=args.batch_size)
    finally:
        refactoring.close()


if __name__ == "__main__":
    main()