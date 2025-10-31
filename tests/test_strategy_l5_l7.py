"""
Test both context strategies on L5-L7 harder tasks.
"""
import tempfile
from pathlib import Path
import json
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_executor_agent import TaskExecutorAgent
from context_strategies import HierarchicalStrategy, AppendUntilFullStrategy
from semantic_validator import validate_workspace


# L5-L7 tasks (subset for benchmarking)
L5_L7_TASKS = [
    {
        "name": "L5_blog_system",
        "goal": """Create blog system with models and manager:
1. Post class with: title, content, author, created_at, comments list
2. Comment class with: author, content, created_at
3. BlogManager class with:
   - add_post(title, content, author) -> returns Post
   - add_comment(post_id, author, content) -> adds Comment to Post
   - get_post(post_id) -> returns Post or None
   - list_posts() -> returns all posts
   - save_to_json(filepath) and load_from_json(filepath)
4. Write tests for CRUD operations and persistence""",
        "validator": "blog_system",
        "max_rounds": 40,
    },
    {
        "name": "L6_observer",
        "goal": """Create observer pattern: Subject class with attach/detach/notify methods, Observer base class, and at least 2 concrete observer implementations. Include tests.""",
        "validator": "observer_pattern",
        "max_rounds": 30,
    },
    {
        "name": "L7_rate_limiter",
        "goal": """Create rate limiter with token bucket algorithm (start simple, no Redis needed):
1. RateLimiter class with:
   - __init__(capacity, refill_rate) - capacity=max tokens, refill_rate=tokens per second
   - _tokens (current tokens), _last_refill (timestamp)
   - _refill() method: calculate tokens to add based on time elapsed
   - allow_request() method: returns True if token available, False otherwise
   - When allowing request, consume 1 token
2. Token refill logic: tokens = min(capacity, current_tokens + (time_elapsed * refill_rate))
3. Write tests verifying: basic allow/deny, refill over time, capacity limits""",
        "validator": "rate_limiter",
        "max_rounds": 40,
    },
]


def run_task_with_strategy(task, strategy, workspace):
    """Run a task with a given strategy and collect metrics."""
    start_time = time.time()

    agent = TaskExecutorAgent(
        workspace=workspace,
        goal=task["goal"],
        max_rounds=task["max_rounds"],
        context_strategy=strategy,
    )

    result = agent.run()

    wall_time = time.time() - start_time

    # Validate workspace semantically
    workspace_dir = agent.workspace_manager.workspace_dir
    validation = validate_workspace(workspace_dir, task["validator"])

    # Calculate passed/failed counts from validation results
    validation_passed = 0
    validation_failed = 0
    if "found" in validation:
        for symbol_type, symbols in validation["found"].items():
            validation_passed += len(symbols)
    if "missing" in validation:
        for symbol_type, symbols in validation["missing"].items():
            validation_failed += len(symbols)

    return {
        "task": task["name"],
        "strategy": strategy.get_name(),
        "status": result.get("status", "unknown"),
        "success": validation["success"],
        "validation_passed": validation_passed,
        "validation_failed": validation_failed,
        "validation_found": validation.get("found", {}),
        "validation_missing": validation.get("missing", {}),
        "rounds": agent.state.total_rounds,
        "wall_time": round(wall_time, 2),
        "total_messages": len(agent.state.messages) if hasattr(agent, 'state') else 0,
        "final_context_tokens": strategy.estimate_context_size(agent.build_context()),
    }


def main():
    """Run L5-L7 benchmark comparing strategies."""
    print("\n" + "="*70)
    print("L5-L7 STRATEGY BENCHMARK")
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

        for task in L5_L7_TASKS:
            print(f"\n  Task: {task['name']}")
            print(f"  Goal: {task['goal'][:80]}...")

            with tempfile.TemporaryDirectory() as tmpdir:
                workspace = Path(tmpdir)

                try:
                    metrics = run_task_with_strategy(task, strategy, workspace)
                    all_results.append(metrics)

                    # Print immediate results
                    status_icon = "✓" if metrics["success"] else "✗"
                    print(f"  {status_icon} Status: {metrics['status']}")
                    print(f"    Success: {metrics['success']}")
                    print(f"    Validation: {metrics['validation_passed']} passed, {metrics['validation_failed']} failed")
                    print(f"    Rounds: {metrics['rounds']}/{task['max_rounds']}")
                    print(f"    Time: {metrics['wall_time']}s")
                    print(f"    Context: {metrics['final_context_tokens']} tokens")
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

    # Generate summary
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

        print(f"\n{strategy_name.upper()}")
        print(f"  Success rate:       {successes}/{total} ({successes/total*100:.1f}%)")
        print(f"  Avg rounds:         {avg_rounds:.1f}")
        print(f"  Avg wall time:      {avg_time:.1f}s")
        print(f"  Avg context tokens: {avg_context:.0f}")

    # Per-task comparison
    print("\n" + "="*70)
    print("PER-TASK COMPARISON")
    print("="*70)

    for task in L5_L7_TASKS:
        print(f"\n{task['name']}:")

        hierarchical = next((r for r in all_results if r["task"] == task["name"] and r["strategy"] == "hierarchical"), None)
        append = next((r for r in all_results if r["task"] == task["name"] and r["strategy"] == "append_until_full"), None)

        if hierarchical and append:
            print(f"  Success:  H={'✓' if hierarchical.get('success') else '✗'}  A={'✓' if append.get('success') else '✗'}")
            print(f"  Rounds:   H={hierarchical.get('rounds', '?')}  A={append.get('rounds', '?')}")
            print(f"  Time:     H={hierarchical.get('wall_time', '?')}s  A={append.get('wall_time', '?')}s")
            print(f"  Context:  H={hierarchical.get('final_context_tokens', '?')}  A={append.get('final_context_tokens', '?')} tokens")

            # Winner analysis
            if hierarchical.get('success') and append.get('success'):
                if hierarchical['rounds'] < append['rounds']:
                    print(f"  🏆 Hierarchical: {hierarchical['rounds']} vs {append['rounds']} rounds")
                elif append['rounds'] < hierarchical['rounds']:
                    print(f"  🏆 Append: {append['rounds']} vs {hierarchical['rounds']} rounds")
                else:
                    print(f"  🤝 Tied at {append['rounds']} rounds")

    # Save results
    output_file = Path("l5_l7_strategy_results.json")
    with open(output_file, 'w') as f:
        json.dump({"results": all_results}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
