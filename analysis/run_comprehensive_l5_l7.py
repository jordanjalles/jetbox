#!/usr/bin/env python3
"""
Comprehensive L5-L7 Evaluation: 3 tasks per level, 3 runs each = 27 total tests.

Tests the agent's performance on advanced integration, architecture, and expert-level tasks.
"""
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from task_executor_agent import TaskExecutorAgent


# 3 representative tasks per level
TASKS = {
    # ===== LEVEL 5: INTEGRATION =====
    "L5_blog_system": {
        "level": "L5",
        "name": "blog_system",
        "goal": "Create blog system: Post model, Comment model, BlogManager with CRUD operations, persistence to JSON",
        "expected_files": ["blog.py", "models.py", "storage.py"],
        "validation": lambda ws: (
            (ws / "blog.py").exists()
            and (ws / "models.py").exists()
            and (ws / "storage.py").exists()
            and "BlogManager" in (ws / "blog.py").read_text()
        )
    },
    "L5_todo_app": {
        "level": "L5",
        "name": "todo_app",
        "goal": "Create todo app: Todo model, Category model, TodoManager with filtering, sorting, and JSON persistence",
        "expected_files": ["todo.py", "models.py", "manager.py"],
        "validation": lambda ws: (
            (ws / "todo.py").exists()
            and (ws / "models.py").exists()
            and (ws / "manager.py").exists()
            and "TodoManager" in (ws / "todo.py").read_text() or "TodoManager" in (ws / "manager.py").read_text()
        )
    },
    "L5_inventory": {
        "level": "L5",
        "name": "inventory_system",
        "goal": "Create inventory system: Product model, Inventory class with add/remove/search, low-stock alerts, CSV export",
        "expected_files": ["inventory.py", "product.py", "alerts.py"],
        "validation": lambda ws: (
            (ws / "inventory.py").exists()
            and (ws / "product.py").exists()
            and (ws / "alerts.py").exists()
            and "Inventory" in (ws / "inventory.py").read_text()
        )
    },

    # ===== LEVEL 6: ARCHITECTURE =====
    "L6_observer": {
        "level": "L6",
        "name": "observer_pattern",
        "goal": "Create observer pattern: Subject, Observer classes, event system with subscribe/unsubscribe/notify",
        "expected_files": ["observer.py", "subject.py", "events.py"],
        "validation": lambda ws: (
            (ws / "observer.py").exists()
            and (ws / "subject.py").exists()
            and (ws / "events.py").exists()
            and "Observer" in (ws / "observer.py").read_text()
            and "Subject" in (ws / "subject.py").read_text()
        )
    },
    "L6_factory": {
        "level": "L6",
        "name": "factory_pattern",
        "goal": "Create factory pattern: Product interface, ConcreteProducts, Factory class with create_product method",
        "expected_files": ["factory.py", "products.py"],
        "validation": lambda ws: (
            (ws / "factory.py").exists()
            and (ws / "products.py").exists()
            and "Factory" in (ws / "factory.py").read_text()
            and "create" in (ws / "factory.py").read_text().lower()
        )
    },
    "L6_dependency_injection": {
        "level": "L6",
        "name": "dependency_injection",
        "goal": "Create DI container: register services, resolve dependencies, singleton/transient lifetimes",
        "expected_files": ["container.py", "services.py"],
        "validation": lambda ws: (
            (ws / "container.py").exists()
            and (ws / "services.py").exists()
            and "Container" in (ws / "container.py").read_text()
            and ("register" in (ws / "container.py").read_text() or "resolve" in (ws / "container.py").read_text())
        )
    },

    # ===== LEVEL 7: EXPERT =====
    "L7_rate_limiter": {
        "level": "L7",
        "name": "rate_limiter",
        "goal": "Create rate limiter: token bucket algorithm, sliding window, distributed support, Redis backend",
        "expected_files": ["rate_limiter.py", "algorithms.py", "backends.py"],
        "validation": lambda ws: (
            (ws / "rate_limiter.py").exists()
            and (ws / "algorithms.py").exists()
            and (ws / "backends.py").exists()
            and "RateLimiter" in (ws / "rate_limiter.py").read_text()
        )
    },
    "L7_connection_pool": {
        "level": "L7",
        "name": "connection_pool",
        "goal": "Create connection pool: acquire/release connections, max pool size, timeout handling, health checks",
        "expected_files": ["pool.py", "connection.py", "health.py"],
        "validation": lambda ws: (
            (ws / "pool.py").exists()
            and (ws / "connection.py").exists()
            and (ws / "health.py").exists()
            and "ConnectionPool" in (ws / "pool.py").read_text() or "Pool" in (ws / "pool.py").read_text()
        )
    },
    "L7_circuit_breaker": {
        "level": "L7",
        "name": "circuit_breaker",
        "goal": "Create circuit breaker: failure detection, half-open state, automatic recovery, metrics tracking",
        "expected_files": ["circuit_breaker.py", "states.py", "metrics.py"],
        "validation": lambda ws: (
            (ws / "circuit_breaker.py").exists()
            and (ws / "states.py").exists()
            and (ws / "metrics.py").exists()
            and "CircuitBreaker" in (ws / "circuit_breaker.py").read_text()
        )
    },
}


def run_task(task_id: str, task_config: dict, run_number: int, max_rounds: int = 30):
    """
    Run a single task.

    Args:
        task_id: Task identifier
        task_config: Task configuration dict
        run_number: Which run this is (1-3)
        max_rounds: Maximum rounds before timeout

    Returns:
        Result dict with success, duration, files_ok, validation_ok
    """
    level = task_config["level"]
    name = task_config["name"]

    print(f"\n{'='*70}")
    print(f"{level}: {name} (Run {run_number}/3)")
    print(f"{'='*70}")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        agent = TaskExecutorAgent(
            workspace=workspace,
            goal=task_config["goal"],
            max_rounds=max_rounds,
            model="gpt-oss:20b",
            temperature=0.2
        )

        # Run agent
        start = time.time()
        try:
            result = agent.run()
            duration = time.time() - start

            # Check files
            actual_workspace = agent.workspace_manager.workspace_dir
            files_ok = all(
                (actual_workspace / f).exists()
                for f in task_config["expected_files"]
            )

            # Run validation
            validation_ok = False
            if files_ok:
                try:
                    validation_ok = task_config["validation"](actual_workspace)
                except Exception as e:
                    print(f"Validation error: {e}")
                    validation_ok = False

            success = files_ok and validation_ok

            return {
                "level": level,
                "task_id": task_id,
                "name": name,
                "run": run_number,
                "success": success,
                "duration": duration,
                "files_ok": files_ok,
                "validation_ok": validation_ok,
                "rounds": agent.state.total_rounds,
            }

        except Exception as e:
            duration = time.time() - start
            print(f"Error: {e}")
            return {
                "level": level,
                "task_id": task_id,
                "name": name,
                "run": run_number,
                "success": False,
                "duration": duration,
                "files_ok": False,
                "validation_ok": False,
                "error": str(e),
                "rounds": agent.state.total_rounds if hasattr(agent.state, 'total_rounds') else 0,
            }


def main():
    """Run comprehensive L5-L7 evaluation."""
    print("="*70)
    print("COMPREHENSIVE EVALUATION: L5-L7")
    print("3 tasks per level, 3 runs each = 27 total tests")
    print("="*70)
    print()

    all_results = []

    # Run each task 3 times
    task_ids = [
        # L5
        "L5_blog_system", "L5_todo_app", "L5_inventory",
        # L6
        "L6_observer", "L6_factory", "L6_dependency_injection",
        # L7
        "L7_rate_limiter", "L7_connection_pool", "L7_circuit_breaker",
    ]

    for task_id in task_ids:
        task_config = TASKS[task_id]

        for run_num in range(1, 4):
            result = run_task(task_id, task_config, run_num)
            all_results.append(result)

            # Show result
            status = "✓" if result["success"] else "✗"
            print(f"{status} {task_config['level']} {task_config['name']} run {run_num}: {result['duration']:.1f}s, {result['rounds']} rounds")

    # Calculate statistics by level
    print("\n" + "="*70)
    print("SUMMARY BY LEVEL")
    print("="*70)

    for level in ["L5", "L6", "L7"]:
        level_results = [r for r in all_results if r["level"] == level]

        successes = sum(1 for r in level_results if r["success"])
        avg_duration = sum(r["duration"] for r in level_results) / len(level_results)
        avg_rounds = sum(r["rounds"] for r in level_results) / len(level_results)

        print(f"\n{level}: {successes}/{len(level_results)} passed (avg {avg_duration:.1f}s, {avg_rounds:.1f} rounds)")

        # Group by task
        tasks_in_level = {}
        for r in level_results:
            task_name = r["name"]
            if task_name not in tasks_in_level:
                tasks_in_level[task_name] = []
            tasks_in_level[task_name].append(r)

        for task_name, task_results in sorted(tasks_in_level.items()):
            task_successes = sum(1 for r in task_results if r["success"])
            task_avg_duration = sum(r["duration"] for r in task_results) / len(task_results)
            task_avg_rounds = sum(r["rounds"] for r in task_results) / len(task_results)
            print(f"  {task_name}: {task_successes}/3 (avg {task_avg_duration:.1f}s, {task_avg_rounds:.1f} rounds)")

    # Overall stats
    total_success = sum(1 for r in all_results if r["success"])
    total_tasks = len(all_results)
    pass_rate = (total_success / total_tasks) * 100

    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"Total passed: {total_success}/{total_tasks} ({pass_rate:.1f}%)")
    print(f"Average duration: {sum(r['duration'] for r in all_results) / len(all_results):.1f}s")
    print(f"Average rounds: {sum(r['rounds'] for r in all_results) / len(all_results):.1f}")

    # Save results
    output_file = Path("comprehensive_l5_l7_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-oss:20b",
            "task_count": total_tasks,
            "results": all_results,
            "summary": {
                "total_passed": total_success,
                "total_tasks": total_tasks,
                "pass_rate": pass_rate,
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
