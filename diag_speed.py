import os
import time

from ollama import chat

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
prompt = "Write a one-line poem about electrons."

print(f"Testing model: {MODEL}")
t0 = time.time()
resp = chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
elapsed = time.time() - t0

print("=== Time elapsed:", round(elapsed, 2), "seconds ===")
print(resp["message"]["content"])
