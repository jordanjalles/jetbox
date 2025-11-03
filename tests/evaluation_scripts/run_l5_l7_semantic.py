#!/usr/bin/env python3
"""
Re-run L5-L7 evaluation with semantic validation.

Uses semantic_validator.py to check for functionality instead of exact file names.
"""
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from task_executor_agent import TaskExecutorAgent
from semantic_validator import validate_workspace


# Same tasks as before, but with semantic validation
TASKS = {
    # ===== LEVEL 5: INTEGRATION =====
    "L5_blog_system": {
        "level": "L5",
        "name": "blog_system",
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
        "validator_task": "blog_system",
    },
    "L5_todo_app": {
        "level": "L5",
        "name": "todo_app",
        "goal": """Create todo app with filtering and sorting:
1. Todo class with: title, description, completed (bool), due_date, category
2. Category class with: name, description
3. TodoManager class with:
   - add_todo(title, description, due_date, category) -> Todo
   - add_category(name, description) -> Category
   - filter_todos(completed=None, category=None, due_before=None) -> filtered list
   - sort_todos(todos, key='due_date', reverse=False) -> sorted list
   - save_to_json(filepath) and load_from_json(filepath)
4. Write tests for filtering, sorting, and persistence""",
        "validator_task": "todo_app",
    },
    "L5_inventory": {
        "level": "L5",
        "name": "inventory_system",
        "goal": """Create inventory management system:
1. Product class with: id, name, quantity, price, min_stock
2. Inventory class with:
   - add_product(name, quantity, price, min_stock) -> Product
   - remove_product(product_id) -> removes product
   - update_quantity(product_id, quantity) -> updates stock
   - search(name_query) -> returns matching products
   - get_low_stock() -> returns products below min_stock
   - export_to_csv(filepath) -> saves inventory to CSV
3. Write tests for add/remove/search and low-stock alerts""",
        "validator_task": "inventory_system",
    },

    # ===== LEVEL 6: ARCHITECTURE =====
    "L6_observer": {
        "level": "L6",
        "name": "observer_pattern",
        "goal": """Create observer pattern: Subject class with attach/detach/notify methods, Observer base class, and at least 2 concrete observer implementations. Include tests.""",
        "validator_task": "observer_pattern",
    },
    "L6_factory": {
        "level": "L6",
        "name": "factory_pattern",
        "goal": """Create factory pattern: Product base class, at least 3 concrete product types, and Factory class with create_product(type) method. Include tests.""",
        "validator_task": "factory_pattern",
    },
    "L6_dependency_injection": {
        "level": "L6",
        "name": "dependency_injection",
        "goal": """Create dependency injection container: DIContainer class with register(name, factory, lifetime) and resolve(name) methods. Support singleton and transient lifetimes. Include tests.""",
        "validator_task": "dependency_injection",
    },

    # ===== LEVEL 7: EXPERT =====
    "L7_rate_limiter": {
        "level": "L7",
        "name": "rate_limiter",
        "goal": """Create rate limiter with token bucket algorithm (start simple, no Redis needed):
1. RateLimiter class with:
   - __init__(capacity, refill_rate) - capacity=max tokens, refill_rate=tokens per second
   - _tokens (current tokens), _last_refill (timestamp)
   - _refill() method: calculate tokens to add based on time elapsed
   - allow_request() method: returns True if token available, False otherwise
   - When allowing request, consume 1 token
2. Token refill logic: tokens = min(capacity, current_tokens + (time_elapsed * refill_rate))
3. Example: capacity=10, refill_rate=1 means 10 requests burst, then 1 per second
4. Write tests showing:
   - Burst of requests (up to capacity)
   - Rate limiting after burst (only refill_rate allowed)
   - Token refill over time""",
        "validator_task": "rate_limiter",
    },
    "L7_connection_pool": {
        "level": "L7",
        "name": "connection_pool",
        "goal": """Create connection pool for managing reusable connections:
1. Connection class: simple object with id and is_healthy() method
2. ConnectionPool class with:
   - __init__(max_size) - maximum connections in pool
   - _available queue (connections ready to use)
   - _in_use set (connections currently borrowed)
   - acquire(timeout=None) method: get connection from pool or create new (up to max_size)
   - release(connection) method: return connection to pool
   - _create_connection() method: create new Connection instance
3. acquire() should:
   - Return available connection if exists
   - Create new if pool not full
   - Wait/timeout if pool exhausted
4. release() should mark connection as available
5. Write tests showing:
   - Acquire/release cycle
   - Pool size limits
   - Multiple acquires exhausting pool""",
        "validator_task": "connection_pool",
    },
    "L7_circuit_breaker": {
        "level": "L7",
        "name": "circuit_breaker",
        "goal": """Create circuit breaker for fault tolerance:
1. CircuitBreaker class with:
   - States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
   - __init__(failure_threshold, timeout) - threshold=failures to open, timeout=seconds before retry
   - _state, _failure_count, _last_failure_time
   - call(func) method: wraps function calls with circuit breaker logic
2. State transitions:
   - CLOSED: normal operation, count failures
   - CLOSED -> OPEN: when failure_count >= failure_threshold
   - OPEN: reject all calls immediately (fast fail)
   - OPEN -> HALF_OPEN: after timeout seconds elapsed
   - HALF_OPEN: allow 1 test call
   - HALF_OPEN -> CLOSED: if test succeeds, reset failure_count
   - HALF_OPEN -> OPEN: if test fails, stay open
3. Write tests showing:
   - Normal operation (closed state)
   - Opening after failures
   - Fast-fail during open
   - Recovery testing (half-open)""",
        "validator_task": "circuit_breaker",
    },
}


def run_task(task_id: str, task_config: dict, run_number: int, max_rounds: int = 30):
    """Run a single task with semantic validation."""
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

            # Get actual workspace
            actual_workspace = agent.workspace_manager.workspace_dir

            # Semantic validation
            validation_result = validate_workspace(
                actual_workspace,
                task_config["validator_task"]
            )

            validation_ok = validation_result["success"]
            files_created = validation_result["summary"]["file_names"]

            return {
                "level": level,
                "task_id": task_id,
                "name": name,
                "run": run_number,
                "success": validation_ok,
                "duration": duration,
                "validation_ok": validation_ok,
                "rounds": agent.state.total_rounds,
                "files_created": files_created,
                "classes_found": validation_result["found"]["classes"],
                "classes_missing": validation_result["missing"]["classes"],
                "functions_found": validation_result["found"]["functions"],
                "functions_missing": validation_result["missing"]["functions"],
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
                "validation_ok": False,
                "error": str(e),
                "rounds": agent.state.total_rounds if hasattr(agent.state, 'total_rounds') else 0,
            }


def main():
    """Run L5-L7 evaluation with semantic validation."""
    print("="*70)
    print("L5-L7 EVALUATION WITH SEMANTIC VALIDATION")
    print("Checking for functionality, not file names")
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

            # Show what was found/missing
            if "classes_found" in result:
                if result["classes_found"]:
                    print(f"   Found: {', '.join(result['classes_found'])}")
                if result["classes_missing"]:
                    print(f"   Missing: {', '.join(result['classes_missing'])}")

    # Calculate statistics
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

    # Comparison to strict validation
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Strict file validation: 0/27 (0%)")
    print(f"Semantic validation: {total_success}/{total_tasks} ({pass_rate:.1f}%)")
    print(f"Improvement: +{pass_rate:.1f} percentage points")

    # Save results
    output_file = Path("l5_l7_semantic_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-oss:20b",
            "validation_type": "semantic",
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
