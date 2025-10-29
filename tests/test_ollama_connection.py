"""Quick test of Ollama connection from agent code."""
import os
import sys
from ollama import Client

print(f"Python version: {sys.version}")
print(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST', 'not set')}")
print(f"OLLAMA_MODEL: {os.environ.get('OLLAMA_MODEL', 'not set')}")
print()

# Initialize client
client = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
print(f"Client initialized with host: {client._client.base_url if hasattr(client, '_client') else 'unknown'}")
print()

# Test 1: List models
print("Test 1: Listing models...")
try:
    result = client.list()
    models = result.get('models', [])
    print(f"✓ Found {len(models)} models")
    for model in models:
        model_obj = model if isinstance(model, dict) else model.__dict__
        name = model_obj.get('model', model_obj.get('name', 'unknown'))
        print(f"  - {name}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print()

# Test 2: Simple chat
print("Test 2: Simple chat request...")
try:
    response = client.chat(
        model=os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"),
        messages=[
            {"role": "user", "content": "Say 'test successful' and nothing else"}
        ],
        options={"temperature": 0.0}
    )
    content = response["message"]["content"]
    print(f"✓ Response: {content[:100]}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("All tests passed! Ollama connection is working.")
