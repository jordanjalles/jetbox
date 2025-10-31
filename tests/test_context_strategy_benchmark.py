"""
Benchmark test comparing context management strategies.

Compares HierarchicalStrategy vs AppendUntilFullStrategy on:
- Token usage (context size)
- Rounds to completion
- Success rate
- Speed (LLM calls, wall time)
"""
import tempfile
from pathlib import Path
import json
import time

from task_executor_agent import TaskExecutorAgent
from context_strategies import HierarchicalStrategy, AppendUntilFullStrategy


# Test tasks of varying complexity
BENCHMARK_TASKS = [
    {
        "name": "simple_file",
        "goal": "Create hello.py with print('Hello, World!')",
        "max_rounds": 10,
        "expected_files": ["hello.py"],
    },
    {
        "name": "two_functions",
        "goal": "Create math_utils.py with add(a, b) and multiply(a, b) functions",
        "max_rounds": 15,
        "expected_files": ["math_utils.py"],
    },
    {
        "name": "with_tests",
        "goal": "Create calculator.py with add(a, b) and subtract(a, b). Write tests.",
        "max_rounds": 20,
        "expected_files": ["calculator.py"],
    },
    {
        "name": "multi_file_web",
        "goal": "Create a simple web page: index.html with CSS (style.css) and JS (app.js)",
        "max_rounds": 25,
        "expected_files": ["index.html", "style.css", "app.js"],
    },
    {
        "name": "package_structure",
        "goal": "Create a Python package 'shapes' with Circle and Rectangle classes. Add __init__.py.",
        "max_rounds": 25,
        "expected_files": ["shapes/__init__.py"],
    },
]


def run_task_with_strategy(task, strategy, workspace):
    """
    Run a task with a given strategy and collect metrics.

    Args:
        task: Task configuration dict
        strategy: ContextStrategy instance
        workspace: Workspace path

    Returns:
        Dict with metrics
    """
    start_time = time.time()

    agent = TaskExecutorAgent(
        workspace=workspace,
        goal=task["goal"],
        max_rounds=task["max_rounds"],
        context_strategy=strategy,
    )

    result = agent.run()

    wall_time = time.time() - start_time

    # Collect metrics
    workspace_dir = agent.workspace_manager.workspace_dir

    # Check file creation
    files_created = []
    for expected in task["expected_files"]:
        path = workspace_dir / expected
        if path.exists():
            files_created.append(expected)

    # Estimate total context tokens used
    total_tokens = 0
    if hasattr(agent, 'state') and hasattr(agent.state, 'messages'):
        for msg in agent.state.messages:
            total_tokens += strategy.estimate_context_size([msg])

    # Get final context size
    final_context = agent.build_context()
    final_context_tokens = strategy.estimate_context_size(final_context)

    return {
        "task": task["name"],
        "strategy": strategy.get_name(),
        "status": result.get("status", "unknown"),
        "success": result.get("status") == "success",
        "rounds": agent.state.total_rounds,
        "files_created": len(files_created),
        "files_expected": len(task["expected_files"]),
        "wall_time": round(wall_time, 2),
        "total_messages": len(agent.state.messages) if hasattr(agent, 'state') else 0,
        "total_tokens_est": total_tokens,
        "final_context_tokens": final_context_tokens,
        "final_context_messages": len(final_context),
    }


def run_benchmark():
    """
    Run full benchmark comparing strategies.

    Returns:
        Dict with results and summary
    """
    print("\n" + "="*70)
    print("CONTEXT STRATEGY BENCHMARK")
    print("="*70)

    strategies = [
        ("hierarchical", HierarchicalStrategy(history_keep=12)),
        ("append_until_full", AppendUntilFullStrategy(max_tokens=8000, recent_keep=20)),
    ]

    all_results = []

    for strategy_name, strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Testing Strategy: {strategy_name}")
        print(f"{'='*70}")

        for task in BENCHMARK_TASKS:
            print(f"\n  Task: {task['name']}")
            print(f"  Goal: {task['goal']}")

            with tempfile.TemporaryDirectory() as tmpdir:
                workspace = Path(tmpdir)

                try:
                    metrics = run_task_with_strategy(task, strategy, workspace)
                    all_results.append(metrics)

                    # Print immediate results
                    status_icon = "✓" if metrics["success"] else "✗"
                    print(f"  {status_icon} Status: {metrics['status']}")
                    print(f"    Rounds: {metrics['rounds']}/{task['max_rounds']}")
                    print(f"    Files: {metrics['files_created']}/{metrics['files_expected']}")
                    print(f"    Time: {metrics['wall_time']}s")
                    print(f"    Context tokens: {metrics['final_context_tokens']}")
                    print(f"    Messages: {metrics['total_messages']}")

                except Exception as e:
                    print(f"  ✗ FAILED WITH EXCEPTION: {e}")
                    all_results.append({
                        "task": task["name"],
                        "strategy": strategy_name,
                        "status": "error",
                        "success": False,
                        "error": str(e),
                    })

    # Generate summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    for strategy_name, _ in strategies:
        strategy_results = [r for r in all_results if r.get("strategy") == strategy_name]

        successes = sum(1 for r in strategy_results if r.get("success"))
        total = len(strategy_results)
        avg_rounds = sum(r.get("rounds", 0) for r in strategy_results) / total if total > 0 else 0
        avg_time = sum(r.get("wall_time", 0) for r in strategy_results) / total if total > 0 else 0
        avg_context = sum(r.get("final_context_tokens", 0) for r in strategy_results) / total if total > 0 else 0
        avg_messages = sum(r.get("total_messages", 0) for r in strategy_results) / total if total > 0 else 0

        print(f"\n{strategy_name.upper()}")
        print(f"  Success rate:       {successes}/{total} ({successes/total*100:.1f}%)")
        print(f"  Avg rounds:         {avg_rounds:.1f}")
        print(f"  Avg wall time:      {avg_time:.1f}s")
        print(f"  Avg context tokens: {avg_context:.0f}")
        print(f"  Avg messages:       {avg_messages:.1f}")

    # Comparison table
    print("\n" + "="*70)
    print("PER-TASK COMPARISON")
    print("="*70)

    for task in BENCHMARK_TASKS:
        print(f"\n{task['name']}:")

        hierarchical = next((r for r in all_results if r["task"] == task["name"] and r["strategy"] == "hierarchical"), None)
        append = next((r for r in all_results if r["task"] == task["name"] and r["strategy"] == "append_until_full"), None)

        if hierarchical and append:
            print(f"  Success:  H={'✓' if hierarchical.get('success') else '✗'}  A={'✓' if append.get('success') else '✗'}")
            print(f"  Rounds:   H={hierarchical.get('rounds', '?')}  A={append.get('rounds', '?')}")
            print(f"  Time:     H={hierarchical.get('wall_time', '?')}s  A={append.get('wall_time', '?')}s")
            print(f"  Context:  H={hierarchical.get('final_context_tokens', '?')}  A={append.get('final_context_tokens', '?')} tokens")
            print(f"  Messages: H={hierarchical.get('total_messages', '?')}  A={append.get('total_messages', '?')}")

            # Winner analysis
            if hierarchical.get('success') and append.get('success'):
                if hierarchical['rounds'] < append['rounds']:
                    print(f"  🏆 Hierarchical completed in fewer rounds")
                elif append['rounds'] < hierarchical['rounds']:
                    print(f"  🏆 Append completed in fewer rounds")

                if hierarchical['final_context_tokens'] < append['final_context_tokens']:
                    print(f"  💾 Hierarchical uses less context ({hierarchical['final_context_tokens']} vs {append['final_context_tokens']})")
                else:
                    print(f"  💾 Append uses less context ({append['final_context_tokens']} vs {hierarchical['final_context_tokens']})")

    return {
        "results": all_results,
        "summary": {
            strategy_name: {
                "total": len([r for r in all_results if r.get("strategy") == strategy_name]),
                "successes": sum(1 for r in all_results if r.get("strategy") == strategy_name and r.get("success")),
                "avg_rounds": sum(r.get("rounds", 0) for r in all_results if r.get("strategy") == strategy_name) / len([r for r in all_results if r.get("strategy") == strategy_name]),
                "avg_context_tokens": sum(r.get("final_context_tokens", 0) for r in all_results if r.get("strategy") == strategy_name) / len([r for r in all_results if r.get("strategy") == strategy_name]),
            }
            for strategy_name, _ in strategies
        }
    }


if __name__ == "__main__":
    results = run_benchmark()

    # Save results
    output_file = Path("context_strategy_benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
