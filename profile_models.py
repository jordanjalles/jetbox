#!/usr/bin/env python3
"""Profile different Ollama models for speed vs quality."""
import time
from ollama import chat, list as list_models

def test_model(model: str, prompt: str, tools: list = None, num_tests: int = 3):
    """Test a model multiple times and report stats."""
    times = []
    responses = []

    for i in range(num_tests):
        try:
            start = time.perf_counter()
            resp = chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                options={"temperature": 0.2},
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            responses.append(resp)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        "avg": avg_time,
        "min": min_time,
        "max": max_time,
        "responses": responses,
    }

def main():
    print("="*60)
    print("OLLAMA MODEL SPEED COMPARISON")
    print("="*60)

    # Get available models
    try:
        models_response = list_models()
        available = [m.get('model', m.get('name', '')) for m in models_response.get('models', [])]
        print(f"\nAvailable models: {', '.join(available[:5])}")
    except Exception as e:
        print(f"Could not list models via API: {e}")
        # Fallback to known models
        available = ["gpt-oss:20b", "qwen2.5-coder:7b", "qwen2.5-coder:3b", "llama3.2:3b"]

    # Test models we know exist
    test_models = [
        "qwen2.5-coder:3b",  # Fastest small model
        "qwen2.5-coder:7b",  # Medium speed, good quality
        "llama3.2:3b",       # Alternative small model
        "gpt-oss:20b",       # Slower but highest quality
    ]

    # Filter to only available models
    test_models = [m for m in test_models if any(m in a for a in available)]
    if not test_models:
        test_models = available[:4]  # Use first 4 available

    if not test_models:
        print("No test models available!")
        return

    print(f"\nTesting models: {', '.join(test_models)}")

    # Simple tool spec
    tools = [{
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            }
        }
    }]

    # Test 1: Simple question (no tools)
    print("\n" + "="*60)
    print("TEST 1: Simple question (no tools)")
    print("="*60)
    prompt = "What is 2+2? Answer with just the number."
    results = {}

    for model in test_models:
        print(f"\n{model}:")
        result = test_model(model, prompt, tools=None, num_tests=3)
        if result:
            results[model] = result
            print(f"  Avg: {result['avg']:.1f}ms | Min: {result['min']:.1f}ms | Max: {result['max']:.1f}ms")
            print(f"  Response: {result['responses'][0]['message']['content'][:100]}")

    # Test 2: Tool calling task
    print("\n" + "="*60)
    print("TEST 2: Tool calling (write file)")
    print("="*60)
    prompt = "Write 'hello world' to test.txt"
    tool_results = {}

    for model in test_models:
        print(f"\n{model}:")
        result = test_model(model, prompt, tools=tools, num_tests=2)
        if result:
            tool_results[model] = result
            print(f"  Avg: {result['avg']:.1f}ms | Min: {result['min']:.1f}ms | Max: {result['max']:.1f}ms")
            msg = result['responses'][0]['message']
            has_tools = 'tool_calls' in msg and msg['tool_calls']
            print(f"  Tool call: {'✓ Yes' if has_tools else '✗ No'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Speed Ranking (faster is better)")
    print("="*60)

    print("\nSimple questions:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg'])
    for i, (model, res) in enumerate(sorted_results, 1):
        print(f"  {i}. {model:20s} - {res['avg']:6.1f}ms avg")

    print("\nTool calling:")
    sorted_tool = sorted(tool_results.items(), key=lambda x: x[1]['avg'])
    for i, (model, res) in enumerate(sorted_tool, 1):
        print(f"  {i}. {model:20s} - {res['avg']:6.1f}ms avg")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if sorted_results:
        fastest = sorted_results[0]
        print(f"\nFastest model: {fastest[0]}")
        print(f"  - Average latency: {fastest[1]['avg']:.1f}ms")
        print(f"  - Best for: Quick iterations, simple tasks")

    if sorted_tool:
        best_tools = sorted_tool[0]
        print(f"\nFastest with tools: {best_tools[0]}")
        print(f"  - Average latency: {best_tools[1]['avg']:.1f}ms")
        print(f"  - Best for: Agent loops with tool calling")

    # Calculate potential speedup
    if len(sorted_tool) > 1:
        slowest = sorted_tool[-1]
        speedup = slowest[1]['avg'] / best_tools[1]['avg']
        savings_per_round = slowest[1]['avg'] - best_tools[1]['avg']
        print(f"\nPotential speedup: {speedup:.1f}x faster")
        print(f"  - Savings per round: {savings_per_round:.1f}ms")
        print(f"  - For 10 rounds: {savings_per_round * 10 / 1000:.1f}s saved")

if __name__ == "__main__":
    main()
