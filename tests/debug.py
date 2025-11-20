#!/usr/bin/env python3
"""
Debug /api/chat Response Structure
Zeigt die genaue Response-Struktur deines Ollama-Servers
"""
import httpx
import json
from config.settings import AppConfig


def debug_chat_response():
    """Test /api/chat und zeige komplette Response"""
    config = AppConfig.from_env()
    
    base_url = config.ollama.host.rstrip('/')
    api_key = config.ollama.api_key
    model = config.ollama.llm_model
    
    print("=" * 60)
    print("Debug /api/chat Response Structure")
    print("=" * 60)
    
    print(f"\n📋 Config:")
    print(f"   Base URL: {base_url}")
    print(f"   Model: {model}")
    print(f"   API Key: {api_key[:10]}...{api_key[-5:]}")
    
    # Build request
    endpoint = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'Hello' in one word."}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 10
        }
    }
    
    print(f"\n🔍 Testing: {endpoint}")
    print(f"\n📤 Request Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        
        response = client.post(endpoint, json=payload)
        
        print(f"\n📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n📊 Response Structure:")
            print(f"   Top-level keys: {list(data.keys())}")
            
            print(f"\n📄 Full Response:")
            print(json.dumps(data, indent=2))
            
            # Analyze message field
            if "message" in data:
                msg = data["message"]
                print(f"\n🔍 'message' field analysis:")
                print(f"   Type: {type(msg)}")
                
                if isinstance(msg, dict):
                    print(f"   Keys: {list(msg.keys())}")
                    print(f"\n   Full message content:")
                    print(json.dumps(msg, indent=4))
                    
                    # Try to find text
                    print(f"\n🎯 Looking for text content:")
                    for key in ["content", "text", "message", "response"]:
                        if key in msg:
                            value = msg[key]
                            print(f"   ✅ Found in '{key}': {value}")
                            break
                    else:
                        print(f"   ⚠️  No standard text key found!")
                        print(f"   Available keys: {list(msg.keys())}")
                
                elif isinstance(msg, str):
                    print(f"   Value: {msg}")
            
            print("\n" + "=" * 60)
            print("✅ Debug complete!")
            
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.close()


if __name__ == "__main__":
    debug_chat_response()