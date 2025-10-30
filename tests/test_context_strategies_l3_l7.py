#!/usr/bin/env python3
"""
Comprehensive context strategy evaluation: L3-L7 tasks with 5-minute timeouts.

Includes partial credit scoring for progress and jetbox notes quality.
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


# L3-L7 task definitions
ALL_TASKS = [
    # L3: Basic file operations
    {
        "name": "L3_calculator",
        "level": 3,
        "goal": "Create calculator package with add, subtract, multiply, divide functions. Write tests for all operations including edge cases (division by zero).",
        "validator": "calculator",
        "timeout": 300,  # 5 minutes
    },
    {
        "name": "L3_file_processor",
        "level": 3,
        "goal": "Create FileProcessor class with read_lines(filepath), write_lines(filepath, lines), count_words(filepath) methods. Add tests.",
        "validator": "file_processor",
        "timeout": 300,
    },

    # L4: Classes with state
    {
        "name": "L4_todo_list",
        "level": 4,
        "goal": "Create TodoList class with add_task(title), complete_task(id), list_tasks(), remove_task(id). Tasks have id, title, completed status. Include tests.",
        "validator": "todo_list",
        "timeout": 300,
    },
    {
        "name": "L4_stack",
        "level": 4,
        "goal": "Create Stack class with push(item), pop(), peek(), is_empty(), size(). Raise exceptions for pop/peek on empty stack. Write tests.",
        "validator": "stack",
        "timeout": 300,
    },

    # L5: Multi-class systems
    {
        "name": "L5_blog_system",
        "level": 5,
        "goal": """Create blog system with models and manager:
1. Post class: title, content, author, created_at, comments list
2. Comment class: author, content, created_at
3. BlogManager class with add_post, add_comment, get_post, list_posts, save_to_json, load_from_json
4. Write tests for CRUD and persistence""",
        "validator": "blog_system",
        "timeout": 300,
    },
    {
        "name": "L5_inventory",
        "level": 5,
        "goal": """Create inventory system:
1. Item class: id, name, quantity, price
2. Inventory class with add_item, remove_item, update_quantity, get_item, list_items, total_value
3. Write tests for all operations""",
        "validator": "inventory_system",
        "timeout": 300,
    },

    # L6: Design patterns
    {
        "name": "L6_observer",
        "level": 6,
        "goal": "Create observer pattern: Subject class with attach/detach/notify, Observer base class, 2+ concrete observers. Include tests.",
        "validator": "observer_pattern",
        "timeout": 300,
    },
    {
        "name": "L6_factory",
        "level": 6,
        "goal": "Create factory pattern: Shape base class, Circle/Square/Triangle concrete classes, ShapeFactory with create_shape(type). Include tests.",
        "validator": "factory_pattern",
        "timeout": 300,
    },

    # L7: Algorithms and complex logic
    {
        "name": "L7_rate_limiter",
        "level": 7,
        "goal": """Create rate limiter with token bucket:
1. RateLimiter(capacity, refill_rate) class
2. allow_request() method (True if token available)
3. Token refill logic based on time elapsed
4. Tests for allow/deny, refill, capacity""",
        "validator": "rate_limiter",
        "timeout": 300,
    },
    {
        "name": "L7_lru_cache",
        "level": 7,
        "goal": """Create LRU cache:
1. LRUCache(capacity) class
2. get(key) and put(key, value) methods
3. Evict least recently used when at capacity
4. Tests for basic ops, eviction, capacity""",
        "validator": "lru_cache",
        "timeout": 300,
    },
]


def calculate_partial_credit(result, workspace_dir):
    """
    Calculate partial credit score (0.0 - 1.0) based on:
    - Validation results (0.5 weight)
    - Jetbox notes quality (0.3 weight)
    - Task completion status (0.2 weight)

    Returns:
        dict with score and breakdown
    """
    scores = {}

    # 1. Validation score (0.5 weight)
    validation_passed = result.get("validation_passed", 0)
    validation_failed = result.get("validation_failed", 0)
    total_validations = validation_passed + validation_failed

    if total_validations > 0:
        validation_score = validation_passed / total_validations
    else:
        validation_score = 0.0
    scores["validation"] = validation_score * 0.5

    # 2. Jetbox notes quality (0.3 weight)
    jetbox_file = workspace_dir / "jetboxnotes.md"
    notes_score = 0.0
    if jetbox_file.exists():
        content = jetbox_file.read_text()
        # Quality indicators
        has_content = len(content) > 50
        has_bullets = "-" in content or "*" in content
        has_progress = any(word in content.lower() for word in ["completed", "created", "implemented", "added"])
        has_structure = "##" in content or len(content.split("\n")) > 3

        notes_score = sum([has_content, has_bullets, has_progress, has_structure]) / 4.0
    scores["jetbox_notes"] = notes_score * 0.3

    # 3. Task completion status (0.2 weight)
    status = result.get("status", "unknown")
    if status == "success":
        completion_score = 1.0
    elif status == "timeout":
        # Timeout with progress gets partial credit
        completion_score = 0.5 if validation_passed > 0 else 0.3
    elif status == "failed":
        completion_score = 0.2 if validation_passed > 0 else 0.0
    else:
        completion_score = 0.0
    scores["completion"] = completion_score * 0.2

    # Total score
    total_score = sum(scores.values())

    return {
        "total_score": round(total_score, 3),
        "breakdown": scores,
        "rating": "excellent" if total_score >= 0.8 else "good" if total_score >= 0.6 else "partial" if total_score >= 0.3 else "poor"
    }


def run_task_with_strategy(task, strategy, workspace):
    """Run a task with strategy, collect metrics, and calculate partial credit."""
    start_time = time.time()

    print(f"\n  Task: {task['name']}")
    print(f"  Goal: {task['goal'][:80]}...")

    agent = TaskExecutorAgent(
        workspace=workspace,
        goal=task["goal"],
        context_strategy=strategy,
        timeout=task["timeout"],  # 5-minute timeout
    )

    result = agent.run(max_rounds=256)  # High round limit, rely on timeout

    wall_time = time.time() - start_time

    # Validate workspace semantically
    workspace_dir = agent.workspace_manager.workspace_dir
    validation = validate_workspace(workspace_dir, task["validator"])

    # Calculate passed/failed counts
    validation_passed = 0
    validation_failed = 0
    if "found" in validation:
        for symbol_type, symbols in validation["found"].items():
            validation_passed += len(symbols)
    if "missing" in validation:
        for symbol_type, symbols in validation["missing"].items():
            validation_failed += len(symbols)

    # Calculate total rounds from agent state (source of truth)
    # NOTE: Using state.total_rounds instead of summing subtask rounds
    # because pre-decomposition rounds won't have an active subtask
    total_rounds = agent.state.total_rounds

    # Build result
    task_result = {
        "task": task["name"],
        "level": task["level"],
        "strategy": strategy.get_name(),
        "status": result.get("status", "unknown"),
        "success": validation["success"],
        "validation_passed": validation_passed,
        "validation_failed": validation_failed,
        "validation_found": validation.get("found", {}),
        "validation_missing": validation.get("missing", {}),
        "rounds": total_rounds,
        "wall_time": round(wall_time, 2),
        "timeout_limit": task["timeout"],
    }

    # Calculate partial credit
    partial_credit = calculate_partial_credit(task_result, workspace_dir)
    task_result["partial_credit"] = partial_credit

    print(f"  ✓ Status: {task_result['status']}")
    print(f"  ✓ Validation: {validation_passed}/{validation_passed + validation_failed} passed")
    print(f"  ✓ Score: {partial_credit['total_score']} ({partial_credit['rating']})")
    print(f"  ✓ Time: {wall_time:.1f}s / {task['timeout']}s")

    return task_result


def main():
    """Run comprehensive L3-L7 evaluation."""
    print("="*70)
    print("CONTEXT STRATEGY EVALUATION: L3-L7")
    print("5-minute timeouts per task | Partial credit scoring")
    print("="*70)

    strategies = [
        HierarchicalStrategy(),
        AppendUntilFullStrategy(),
    ]

    results = []

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Testing Strategy: {strategy.get_name()}")
        print(f"{'='*70}")

        for task in ALL_TASKS:
            with tempfile.TemporaryDirectory() as tmpdir:
                workspace = Path(tmpdir)

                try:
                    result = run_task_with_strategy(task, strategy, workspace)
                    results.append(result)
                except Exception as e:
                    print(f"  ✗ Exception: {e}")
                    results.append({
                        "task": task["name"],
                        "level": task["level"],
                        "strategy": strategy.get_name(),
                        "status": "exception",
                        "success": False,
                        "validation_passed": 0,
                        "validation_failed": 0,
                        "rounds": 0,
                        "wall_time": 0,
                        "partial_credit": {"total_score": 0.0, "rating": "poor"},
                        "error": str(e),
                    })

    # Save results
    output_file = "l3_l7_context_strategy_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print_summary(results)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")


def print_summary(results):
    """Print comparative summary with partial credit."""
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")

    strategies = list(set(r["strategy"] for r in results))

    for strategy in strategies:
        strat_results = [r for r in results if r["strategy"] == strategy]

        # Calculate stats
        total_tasks = len(strat_results)
        full_success = sum(1 for r in strat_results if r["success"])
        avg_score = sum(r["partial_credit"]["total_score"] for r in strat_results) / total_tasks
        avg_time = sum(r["wall_time"] for r in strat_results) / total_tasks
        avg_rounds = sum(r["rounds"] for r in strat_results) / total_tasks

        # By level
        by_level = {}
        for level in [3, 4, 5, 6, 7]:
            level_results = [r for r in strat_results if r["level"] == level]
            if level_results:
                by_level[level] = {
                    "count": len(level_results),
                    "success": sum(1 for r in level_results if r["success"]),
                    "avg_score": sum(r["partial_credit"]["total_score"] for r in level_results) / len(level_results),
                }

        print(f"\n{strategy.upper()}")
        print(f"  Overall:")
        print(f"    Full success: {full_success}/{total_tasks} ({full_success/total_tasks*100:.1f}%)")
        print(f"    Avg score:    {avg_score:.3f}")
        print(f"    Avg time:     {avg_time:.1f}s")
        print(f"    Avg rounds:   {avg_rounds:.1f}")

        print(f"  By level:")
        for level, stats in sorted(by_level.items()):
            print(f"    L{level}: {stats['success']}/{stats['count']} success, {stats['avg_score']:.3f} avg score")


if __name__ == "__main__":
    main()
