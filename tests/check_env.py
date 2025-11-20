"""Check what's actually loaded from .env"""
from dotenv import load_dotenv
import os

load_dotenv()

print("🔍 Checking environment variables:\n")

vars_to_check = [
    "NEO4J_URI",
    "NEO4J_USER", 
    "NEO4J_PASSWORD",
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "OLLAMA_API_KEY"
]

for var in vars_to_check:
    value = os.getenv(var)
    if value:
        # Mask sensitive values
        if "PASSWORD" in var or "KEY" in var:
            display = f"{value[:10]}..." if len(value) > 10 else "***"
        else:
            display = value
        print(f"✅ {var:20s} = {display}")
    else:
        print(f"❌ {var:20s} = NOT SET")

print("\n" + "="*60)
print("💡 If OLLAMA_API_KEY shows 'NOT SET', check your .env file!")
print("="*60)