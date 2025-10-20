#!/usr/bin/env python3
"""Profile agent performance to identify bottlenecks."""
import time
import json
from pathlib import Path
from typing import Any
import subprocess

# Timing decorator
def timeit(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            print(f"⏱ {name}: {elapsed:.2f}ms")
            return result
        return wrapper
    return decorator

# Profile different operations
@timeit("File read (1KB)")
def profile_file_read_small():
    Path("test_small.txt").write_text("x" * 1024)
    return Path("test_small.txt").read_text()

@timeit("File read (100KB)")
def profile_file_read_medium():
    Path("test_medium.txt").write_text("x" * 102400)
    return Path("test_medium.txt").read_text()

@timeit("File write (1KB)")
def profile_file_write_small():
    Path("test_write.txt").write_text("x" * 1024)

@timeit("File write (100KB)")
def profile_file_write_medium():
    Path("test_write.txt").write_text("x" * 102400)

@timeit("JSON serialize (state.json)")
def profile_json_serialize():
    data = {
        "goal": {"description": "test", "tasks": [
            {"description": f"task_{i}", "subtasks": [
                {"description": f"subtask_{j}", "actions": []}
                for j in range(5)
            ]} for i in range(10)
        ]},
        "current_task_idx": 0,
        "loop_counts": {f"action_{i}": i for i in range(100)},
    }
    return json.dumps(data, indent=2)

@timeit("JSON deserialize (state.json)")
def profile_json_deserialize():
    data = profile_json_serialize()
    return json.loads(data)

@timeit("Subprocess: python --version")
def profile_subprocess_python():
    return subprocess.run(["python", "--version"], capture_output=True, text=True)

@timeit("Subprocess: pytest --collect-only")
def profile_subprocess_pytest_collect():
    return subprocess.run(["pytest", "--collect-only", "-q"], capture_output=True, text=True, timeout=5)

@timeit("Subprocess: ruff check (dry)")
def profile_subprocess_ruff():
    return subprocess.run(["ruff", "check", "agent.py"], capture_output=True, text=True, timeout=5)

@timeit("Path.exists() x100")
def profile_path_exists():
    p = Path("agent.py")
    for _ in range(100):
        p.exists()

@timeit("Import ollama")
def profile_import_ollama():
    import ollama
    return ollama

@timeit("Ollama chat (1 token, no tools)")
def profile_ollama_minimal():
    from ollama import chat
    start = time.perf_counter()
    resp = chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "Hi"}],
        options={"num_predict": 1},
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  └─ Actual chat time: {elapsed:.2f}ms")
    return resp

@timeit("Ollama chat (simple question, no tools)")
def profile_ollama_simple():
    from ollama import chat
    start = time.perf_counter()
    resp = chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        options={"temperature": 0.2},
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  └─ Actual chat time: {elapsed:.2f}ms")
    print(f"  └─ Response length: {len(resp['message']['content'])} chars")
    return resp

@timeit("Ollama chat (with tools, simple)")
def profile_ollama_with_tools():
    from ollama import chat
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    }]
    start = time.perf_counter()
    resp = chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "Call test_tool"}],
        tools=tools,
        options={"temperature": 0.2},
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  └─ Actual chat time: {elapsed:.2f}ms")
    return resp

def profile_agent_round():
    """Profile a full agent round from agent_enhanced.py."""
    print("\n" + "="*60)
    print("AGENT ROUND BREAKDOWN")
    print("="*60)

    # Simulate a typical round
    timings = {}

    # 1. Probe state
    start = time.perf_counter()
    from agent_enhanced import probe_state
    state = probe_state()
    timings["probe_state"] = (time.perf_counter() - start) * 1000

    # 2. Context manager operations
    start = time.perf_counter()
    from context_manager import ContextManager
    ctx = ContextManager()
    ctx.load_or_init("Test goal")
    timings["context_load"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    ctx.update_probe_state(state)
    timings["context_update"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    compact = ctx.get_compact_context()
    timings["context_compact"] = (time.perf_counter() - start) * 1000

    # 3. Message construction
    start = time.perf_counter()
    messages = [
        {"role": "system", "content": "Test system prompt"},
        {"role": "user", "content": compact},
    ]
    timings["message_build"] = (time.perf_counter() - start) * 1000

    # 4. LLM call (simulated with minimal request)
    start = time.perf_counter()
    from ollama import chat
    from agent_enhanced import tool_specs
    resp = chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": "Write file test.py"}],
        tools=tool_specs(),
        options={"temperature": 0.2, "num_predict": 50},
    )
    timings["llm_call"] = (time.perf_counter() - start) * 1000

    # 5. Tool execution (write_file)
    start = time.perf_counter()
    Path("test_tool.py").write_text("# test\n")
    timings["tool_execute"] = (time.perf_counter() - start) * 1000

    # 6. State save
    start = time.perf_counter()
    ctx._save_state()
    timings["state_save"] = (time.perf_counter() - start) * 1000

    # Print breakdown
    print("\nTIMING BREAKDOWN (one round):")
    total = sum(timings.values())
    for name, ms in sorted(timings.items(), key=lambda x: -x[1]):
        pct = (ms / total) * 100
        print(f"  {name:20s}: {ms:7.2f}ms ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total:7.2f}ms")

    return timings

def main():
    print("="*60)
    print("JETBOX AGENT PERFORMANCE PROFILER")
    print("="*60)

    print("\n--- File I/O Operations ---")
    profile_file_read_small()
    profile_file_read_medium()
    profile_file_write_small()
    profile_file_write_medium()

    print("\n--- JSON Operations ---")
    profile_json_serialize()
    profile_json_deserialize()

    print("\n--- Path Operations ---")
    profile_path_exists()

    print("\n--- Subprocess Operations ---")
    profile_subprocess_python()
    try:
        profile_subprocess_pytest_collect()
    except Exception as e:
        print(f"  └─ Error: {e}")
    try:
        profile_subprocess_ruff()
    except Exception as e:
        print(f"  └─ Error: {e}")

    print("\n--- Ollama Operations ---")
    profile_import_ollama()

    print("\nTesting Ollama latency (may take a few seconds)...")
    try:
        profile_ollama_minimal()
        profile_ollama_simple()
        profile_ollama_with_tools()
    except Exception as e:
        print(f"  └─ Ollama error: {e}")
        print("  └─ Make sure Ollama is running and gpt-oss:20b is available")

    print("\n--- Full Agent Round ---")
    try:
        profile_agent_round()
    except Exception as e:
        print(f"Error profiling agent round: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    for f in ["test_small.txt", "test_medium.txt", "test_write.txt", "test_tool.py"]:
        Path(f).unlink(missing_ok=True)

    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
