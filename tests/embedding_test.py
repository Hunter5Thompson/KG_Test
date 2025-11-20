"""Test /api/embed endpoint"""
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OLLAMA_API_KEY")
BASE_URL = os.getenv("OLLAMA_HOST", "http://test.ki-plattform.apps.gisamgmt.global/").rstrip('/')
MODEL = "qwen3-embedding:4b-q8_0"

endpoint = f"{BASE_URL}/api/embed"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print(f"🔍 Testing: {endpoint}\n")

# Test verschiedene Payload-Formate
payloads = [
    {"model": MODEL, "input": "test"},
    {"model": MODEL, "prompt": "test"},
    {"model": MODEL, "text": "test"},
]

client = httpx.Client(headers=headers, timeout=10.0)

for i, payload in enumerate(payloads, 1):
    print(f"Test {i}: {payload}")
    
    try:
        response = client.post(endpoint, json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS!")
            print(f"Response keys: {list(data.keys())}")
            
            # Finde das Embedding
            if "embedding" in data:
                print(f"📊 Embedding dimension: {len(data['embedding'])}")
                print(f"✨ First 5 values: {data['embedding'][:5]}")
                print(f"\n🎯 USE THIS PAYLOAD FORMAT: {payload}\n")
                break
            elif "embeddings" in data:
                emb = data["embeddings"]
                if isinstance(emb, list) and len(emb) > 0:
                    first_emb = emb[0] if isinstance(emb[0], list) else emb
                    print(f"📊 Embedding dimension: {len(first_emb)}")
                    print(f"✨ First 5 values: {first_emb[:5]}")
                    print(f"\n🎯 USE THIS PAYLOAD FORMAT: {payload}\n")
                    break
        else:
            print(f"❌ Failed: {response.text[:200]}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 60 + "\n")

client.close()