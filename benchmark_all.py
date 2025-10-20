#!/usr/bin/env python3
"""Comprehensive benchmark comparing all agent versions."""
import subprocess
import time
import json
from pathlib import Path
import shutil

def cleanup():
    """Remove test artifacts."""
    for path in ["mathx", "tests", "pyproject.toml", ".agent_context",
                 "agent.log", "agent_enhanced.log", "agent_fast.log"]:
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

def run_agent(script: str, model: str, goal: str, timeout: int = 120) -> dict:
    """Run an agent and measure performance."""
    print(f"\n{'='*60}")
    print(f"Testing: {script} with {model}")
    print(f"{'='*60}")

    cleanup()
    time.sleep(1)  # Let filesystem settle

    env = {"OLLAMA_MODEL": model}
    start = time.time()

    try:
        result = subprocess.run(
            ["python", script, goal],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**env}
        )
        elapsed = time.time() - start

        # Parse log file for round timings
        log_file = script.replace(".py", ".log")
        rounds = []
        if Path(log_file).exists():
            log_content = Path(log_file).read_text()
            for line in log_content.split("\n"):
                if "Round" in line and "ms" in line:
                    rounds.append(line)

        return {
            "script": script,
            "model": model,
            "success": result.returncode == 0,
            "elapsed": elapsed,
            "stdout": result.stdout[:500],
            "stderr": result.stderr[:500] if result.stderr else "",
            "rounds": rounds[:10],  # First 10 rounds
            "files_created": {
                "mathx": Path("mathx/__init__.py").exists(),
                "tests": Path("tests/test_mathx.py").exists(),
                "pyproject": Path("pyproject.toml").exists(),
            }
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "script": script,
            "model": model,
            "success": False,
            "elapsed": elapsed,
            "error": f"Timeout after {timeout}s",
            "rounds": [],
            "files_created": {}
        }

    except Exception as e:
        return {
            "script": script,
            "model": model,
            "success": False,
            "error": str(e),
            "rounds": [],
            "files_created": {}
        }

def main():
    goal = "Create mathx package with add(a,b) and multiply(a,b), add tests, run ruff and pytest."

    tests = [
        # Fast model comparisons
        ("agent_fast.py", "llama3.2:3b"),
        ("agent_fast.py", "qwen2.5-coder:3b"),
        ("agent_fast.py", "qwen2.5-coder:7b"),

        # Quality model
        ("agent_fast.py", "gpt-oss:20b"),

        # Baseline for comparison
        ("agent_enhanced.py", "gpt-oss:20b"),
    ]

    results = []

    print(f"\n{'='*60}")
    print("JETBOX AGENT PERFORMANCE BENCHMARK")
    print(f"{'='*60}")
    print(f"\nGoal: {goal}")
    print(f"\nRunning {len(tests)} test configurations...")

    for i, (script, model) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Testing {script} with {model}...")
        result = run_agent(script, model, goal, timeout=60)
        results.append(result)

        # Quick summary
        status = "✓" if result.get("success") else "✗"
        elapsed = result.get("elapsed", 0)
        files = result.get("files_created", {})
        file_count = sum(1 for v in files.values() if v)

        print(f"  {status} Completed in {elapsed:.1f}s")
        print(f"  Files created: {file_count}/3")

    # Final summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Script':<20} {'Model':<18} {'Time':>8} {'Files':>6} {'Status'}")
    print("-" * 60)

    for r in results:
        script = r['script'].replace('.py', '')
        model = r['model']
        elapsed = r.get('elapsed', 0)
        files = r.get('files_created', {})
        file_count = sum(1 for v in files.values() if v)
        status = "✓" if r.get('success') else "✗"

        print(f"{script:<20} {model:<18} {elapsed:7.1f}s {file_count:>3}/3 {status}")

    # Speed ranking
    print(f"\n{'='*60}")
    print("SPEED RANKING (fastest first)")
    print(f"{'='*60}")

    sorted_results = sorted(results, key=lambda x: x.get('elapsed', 999))
    for i, r in enumerate(sorted_results, 1):
        script = r['script'].replace('.py', '')
        model = r['model']
        elapsed = r.get('elapsed', 0)
        print(f"  {i}. {script:<15} {model:<18} {elapsed:6.1f}s")

    # Save detailed results
    output_file = Path("benchmark_results.json")
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to: {output_file}")

    cleanup()

if __name__ == "__main__":
    main()
