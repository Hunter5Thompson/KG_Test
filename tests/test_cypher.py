# KORREKT
cypher_query = """
    MATCH (n:Entity) 
    WHERE n.content CONTAINS 'Chapter 7' 
    RETURN n.name AS title, n.id AS id // <--- 'id' hier hinzugefügt
"""
records = session.run(cypher_query)
for r in records:
    print(r['id']) # <-- Funktioniert jetzt