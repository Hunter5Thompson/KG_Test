# clean_embeddings.py
import json
import sys

def remove_embeddings(data):
    """Rekursiv alle 'embedding' Keys entfernen"""
    if isinstance(data, dict):
        # Entferne embedding wenn vorhanden
        if 'embedding' in data:
            del data['embedding']
        # Auch in properties schauen (Neo4j nested structure)
        if 'properties' in data and isinstance(data['properties'], dict):
            if 'embedding' in data['properties']:
                del data['properties']['embedding']
        # Rekursiv durch alle Values
        for value in data.values():
            remove_embeddings(value)
    elif isinstance(data, list):
        # Rekursiv durch alle List-Items
        for item in data:
            remove_embeddings(item)
    return data

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "graph_export.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "graph_clean.json"
    
    print(f"📖 Reading: {input_file}")
    
    # ✅ FIX: utf-8-sig handelt BOM automatisch
    try:
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        # Fallback: Versuche andere Encodings
        print("   ⚠️  UTF-8 failed, trying latin-1...")
        with open(input_file, 'r', encoding='latin-1') as f:
            data = json.load(f)
    
    print("🧹 Removing embeddings...")
    clean_data = remove_embeddings(data)
    
    print(f"💾 Writing: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)
    
    # Statistiken
    import os
    size_before = os.path.getsize(input_file) / 1024 / 1024
    size_after = os.path.getsize(output_file) / 1024 / 1024
    saved = size_before - size_after
    
    print(f"\n✅ Done!")
    print(f"   Before: {size_before:.2f} MB")
    print(f"   After:  {size_after:.2f} MB")
    print(f"   Saved:  {saved:.2f} MB ({(saved/size_before*100):.1f}%)")

if __name__ == "__main__":
    main()