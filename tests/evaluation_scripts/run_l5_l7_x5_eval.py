#!/usr/bin/env python3
"""
L5-L7 Evaluation Runner (5x each)

Runs 5 instances of L5, L6, and L7 complexity tasks to verify orchestrator reliability.
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# Test definitions
TESTS = {
    "L5": [
        "Create a Flask REST API with CRUD endpoints for a User model (fields: id, name, email). Use in-memory storage. Include POST /users, GET /users, GET /users/<id>, PUT /users/<id>, DELETE /users/<id>. Write pytest tests for all endpoints.",
        "Build a simple blog API with Post model (id, title, content, author). Use Flask and in-memory list storage. Include CRUD endpoints and pytest tests.",
        "Create a Flask API for a Todo list with Todo model (id, text, completed). Use in-memory storage. Include CRUD operations and unit tests with pytest.",
        "Build a Flask REST API for a Product catalog (id, name, price, description). Use in-memory list. Include all CRUD endpoints and comprehensive pytest tests.",
        "Create a Flask API for managing Books (id, title, author, year). Use in-memory storage. Include CRUD endpoints, error handling, and pytest tests.",
    ],
    "L6": [
        "Build a Flask blog API with User and Post models. Include user registration/login with JWT tokens, post CRUD operations, and SQLite database. Write pytest tests for auth and posts.",
        "Create a Flask e-commerce API with Product and Order models. Include JWT authentication, product catalog, order management, SQLite storage, and unit tests.",
        "Build a Flask forum API with User, Thread, and Reply models. Include authentication with JWT, thread/reply CRUD, SQLite database, and comprehensive tests.",
        "Create a Flask library API with User and Book models. Include JWT auth, book checkout/return, due dates, SQLite storage, and pytest tests.",
        "Build a Flask ticket system with User and Ticket models. Include JWT authentication, ticket creation/assignment, status tracking, SQLite, and tests.",
    ],
    "L7": [
        "Build a task management system with: 1) User authentication (register/login), 2) Projects with multiple tasks, 3) Task assignment to users, 4) Status tracking (todo/in-progress/done), 5) SQLite database, 6) Flask REST API, 7) Unit tests for all endpoints. Make it production-ready.",
        "Create a blogging platform with: 1) User authentication (JWT), 2) Posts with rich content, 3) Comments and replies, 4) Like/upvote system, 5) Categories and tags, 6) SQLite database, 7) Flask REST API, 8) Comprehensive unit tests. Make it production-ready.",
        "Build an inventory management system with: 1) User authentication, 2) Products with categories, 3) Stock tracking and updates, 4) Order management, 5) Supplier management, 6) SQLite database, 7) Flask REST API, 8) Complete test suite. Production-ready.",
        "Create a customer support system with: 1) User authentication (agents and customers), 2) Ticket creation and assignment, 3) Comments and status updates, 4) Priority levels, 5) Department routing, 6) SQLite database, 7) Flask REST API, 8) Full unit tests. Production-ready.",
        "Build a project collaboration tool with: 1) User authentication, 2) Workspaces and projects, 3) Tasks with assignments, 4) File attachments, 5) Activity logs, 6) SQLite database, 7) Flask REST API, 8) Comprehensive tests. Production-ready.",
    ]
}

TIMEOUT = 300  # 5 minutes per test

def run_test(level, test_num, goal, output_dir):
    """Run a single test and return result."""
    test_id = f"{level}_run{test_num}"
    log_file = output_dir / f"{test_id}.log"

    print(f"\n{'='*70}")
    print(f"Running {test_id}")
    print(f"Goal: {goal[:80]}...")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Run orchestrator
        cmd = [
            "timeout", str(TIMEOUT),
            "python", "orchestrator_main.py",
            goal,
            "--once"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT + 10  # Extra buffer
        )

        elapsed = time.time() - start_time

        # Save full log
        with open(log_file, 'w') as f:
            f.write(f"=== {test_id} ===\n")
            f.write(f"Goal: {goal}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Elapsed: {elapsed:.1f}s\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        # Analyze result
        stdout = result.stdout.lower()
        stderr = result.stderr.lower()

        # Success indicators
        success_indicators = [
            "goal complete",
            "mark_complete",
            "completed with status: success"
        ]

        # Failure indicators
        failure_indicators = [
            "error:",
            "exception:",
            "traceback",
            "failed",
            "timed out"
        ]

        has_success = any(ind in stdout for ind in success_indicators)
        has_failure = any(ind in stderr or ind in stdout for ind in failure_indicators)

        # Determine status
        if result.returncode == 124:  # timeout return code
            status = "TIMEOUT"
        elif has_success and not has_failure:
            status = "SUCCESS"
        elif has_failure:
            status = "FAILED"
        else:
            status = "UNKNOWN"

        print(f"✓ {test_id}: {status} ({elapsed:.1f}s)")

        return {
            "test_id": test_id,
            "level": level,
            "run": test_num,
            "goal": goal,
            "status": status,
            "elapsed": elapsed,
            "returncode": result.returncode,
            "log_file": str(log_file)
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ {test_id}: TIMEOUT ({elapsed:.1f}s)")

        return {
            "test_id": test_id,
            "level": level,
            "run": test_num,
            "goal": goal,
            "status": "TIMEOUT",
            "elapsed": elapsed,
            "returncode": 124,
            "log_file": str(log_file)
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ {test_id}: ERROR - {e}")

        return {
            "test_id": test_id,
            "level": level,
            "run": test_num,
            "goal": goal,
            "status": "ERROR",
            "elapsed": elapsed,
            "error": str(e),
            "log_file": str(log_file)
        }

def main():
    """Run all tests and generate report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation_results") / f"l5_l7_x5_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("L5-L7 x5 EVALUATION")
    print(f"Output directory: {output_dir}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    results = []
    start_time = time.time()

    # Run all tests
    for level in ["L5", "L6", "L7"]:
        for run_num, goal in enumerate(TESTS[level], 1):
            result = run_test(level, run_num, goal, output_dir)
            results.append(result)

            # Small delay between tests
            time.sleep(2)

    total_elapsed = time.time() - start_time

    # Save results JSON
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total_elapsed": total_elapsed,
            "results": results
        }, f, indent=2)

    # Generate summary report
    report_file = output_dir / "SUMMARY.md"
    generate_report(results, total_elapsed, report_file)

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"Results: {results_file}")
    print(f"Report: {report_file}")
    print(f"{'='*70}\n")

    # Print summary
    with open(report_file) as f:
        print(f.read())

def generate_report(results, total_elapsed, report_file):
    """Generate markdown summary report."""

    # Calculate stats
    by_level = {"L5": [], "L6": [], "L7": []}
    for r in results:
        by_level[r["level"]].append(r)

    stats = {}
    for level, level_results in by_level.items():
        total = len(level_results)
        success = sum(1 for r in level_results if r["status"] == "SUCCESS")
        failed = sum(1 for r in level_results if r["status"] == "FAILED")
        timeout = sum(1 for r in level_results if r["status"] == "TIMEOUT")
        error = sum(1 for r in level_results if r["status"] == "ERROR")
        avg_time = sum(r["elapsed"] for r in level_results) / total if total > 0 else 0

        stats[level] = {
            "total": total,
            "success": success,
            "failed": failed,
            "timeout": timeout,
            "error": error,
            "avg_time": avg_time,
            "success_rate": (success / total * 100) if total > 0 else 0
        }

    # Write report
    with open(report_file, 'w') as f:
        f.write("# L5-L7 x5 Evaluation Report\n\n")
        f.write(f"**Date**: {datetime.now().isoformat()}\n")
        f.write(f"**Total Time**: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)\n")
        f.write(f"**Total Tests**: {len(results)}\n\n")

        f.write("## Summary by Level\n\n")
        f.write("| Level | Total | Success | Failed | Timeout | Error | Success Rate | Avg Time |\n")
        f.write("|-------|-------|---------|--------|---------|-------|--------------|----------|\n")

        for level in ["L5", "L6", "L7"]:
            s = stats[level]
            f.write(f"| {level} | {s['total']} | {s['success']} | {s['failed']} | {s['timeout']} | {s['error']} | {s['success_rate']:.1f}% | {s['avg_time']:.1f}s |\n")

        # Overall stats
        total_tests = len(results)
        total_success = sum(s["success"] for s in stats.values())
        total_failed = sum(s["failed"] for s in stats.values())
        total_timeout = sum(s["timeout"] for s in stats.values())
        total_error = sum(s["error"] for s in stats.values())
        overall_rate = (total_success / total_tests * 100) if total_tests > 0 else 0

        f.write(f"| **Overall** | {total_tests} | {total_success} | {total_failed} | {total_timeout} | {total_error} | {overall_rate:.1f}% | {total_elapsed/total_tests:.1f}s |\n\n")

        f.write("## Detailed Results\n\n")

        for level in ["L5", "L6", "L7"]:
            f.write(f"### {level} Results\n\n")

            for r in by_level[level]:
                status_emoji = {
                    "SUCCESS": "✅",
                    "FAILED": "❌",
                    "TIMEOUT": "⏱️",
                    "ERROR": "💥"
                }.get(r["status"], "❓")

                f.write(f"{status_emoji} **{r['test_id']}** ({r['elapsed']:.1f}s)\n")
                f.write(f"- Goal: {r['goal'][:100]}...\n")
                f.write(f"- Status: {r['status']}\n")
                f.write(f"- Log: `{r['log_file']}`\n\n")

        f.write("## Analysis\n\n")

        if overall_rate >= 80:
            f.write("**Status**: ✅ EXCELLENT - System performing well\n\n")
        elif overall_rate >= 60:
            f.write("**Status**: ⚠️ GOOD - Some issues to investigate\n\n")
        else:
            f.write("**Status**: ❌ NEEDS ATTENTION - Significant issues detected\n\n")

        f.write("**Key Findings**:\n")

        # Identify patterns
        if total_timeout > 0:
            f.write(f"- {total_timeout} tests timed out (>{TIMEOUT}s) - may need timeout adjustment or performance optimization\n")

        if total_error > 0:
            f.write(f"- {total_error} tests had errors - check logs for exceptions\n")

        # Level-specific observations
        for level in ["L5", "L6", "L7"]:
            s = stats[level]
            if s["success_rate"] < 50:
                f.write(f"- {level} has low success rate ({s['success_rate']:.1f}%) - investigate {level}-specific issues\n")

        f.write("\n")

if __name__ == "__main__":
    main()
