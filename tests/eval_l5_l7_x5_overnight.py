#!/usr/bin/env python3
"""
L5-L7 Overnight Evaluation (5 runs per problem)

This evaluation tests delegation with:
- Empty round detection and recovery
- Tool call error feedback
- Workspace coordination
- Architect delegation guidance

Each level has 3 problems × 5 runs = 15 tests per level
Total: 45 tests (L5-L7)
Expected duration: 4-6 hours
"""
import json
import time
from pathlib import Path
from datetime import datetime
from agents.orchestrator_agent import OrchestratorAgent

# L5-L7 Test Problems
TEST_PROBLEMS = {
    "L5": [
        "Create a Flask REST API with CRUD endpoints for a User model. Use in-memory storage. Include tests.",
        "Create a simple blog API with posts and comments. Use SQLite. Include tests for all endpoints.",
        "Create a task management API with categories and priorities. Use in-memory storage. Include tests.",
    ],
    "L6": [
        "Create a Flask app with user authentication (login/logout). Use session management. Include tests.",
        "Create a multi-user chat API with rooms and messages. Use in-memory storage. Include tests.",
        "Create an e-commerce API with products, cart, and checkout. Use in-memory storage. Include tests.",
    ],
    "L7": [
        "Create a full-stack Flask app with user auth, posts, and comments. Use SQLite. Include frontend templates.",
        "Create a project management system with users, projects, tasks. Use SQLite. Include tests and docs.",
        "Create a collaborative todo app with sharing and permissions. Use SQLite. Include tests and API docs.",
    ],
}

def run_single_test(level: str, problem_idx: int, problem: str, run_idx: int) -> dict:
    """Run a single test problem."""
    print(f"\n{'='*80}")
    print(f"{level} Problem {problem_idx+1} Run {run_idx+1}")
    print(f"{'='*80}")
    print(f"Task: {problem[:70]}...")
    print()

    start_time = time.time()

    # Create workspace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_name = f"{level.lower()}_p{problem_idx+1}_run{run_idx+1}_{timestamp}"
    workspace = Path(".agent_workspaces") / workspace_name

    # Run test
    try:
        orchestrator = OrchestratorAgent(workspace=workspace)
        orchestrator.add_message({
            "role": "user",
            "content": problem
        })

        result = orchestrator.run(max_rounds=20)

        end_time = time.time()
        duration = end_time - start_time

        # Count files created
        files_created = 0
        if workspace.exists():
            files_created = len([f for f in workspace.rglob("*") if f.is_file() and not f.name.startswith(".")])

        return {
            "level": level,
            "problem_idx": problem_idx,
            "run_idx": run_idx,
            "problem": problem,
            "status": result.get("status"),
            "duration": duration,
            "files_created": files_created,
            "workspace": str(workspace),
            "summary": result.get("summary", result.get("message", result.get("reason", ""))),
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        return {
            "level": level,
            "problem_idx": problem_idx,
            "run_idx": run_idx,
            "problem": problem,
            "status": "error",
            "duration": duration,
            "files_created": 0,
            "workspace": str(workspace),
            "error": str(e),
        }

def main():
    """Run L5-L7 evaluation with 5 runs per problem."""
    print("="*80)
    print("L5-L7 OVERNIGHT EVALUATION (x5)")
    print("="*80)
    print()
    print("Testing with:")
    print("  - Empty round detection and recovery")
    print("  - Tool call error feedback")
    print("  - Workspace coordination")
    print("  - Architect delegation guidance")
    print()
    print("Levels: L5-L7")
    print("Problems per level: 3")
    print("Runs per problem: 5")
    print("Total tests: 45")
    print("Expected duration: 4-6 hours")
    print()
    print("="*80)

    overall_start = time.time()
    results = []

    # Run all tests
    for level in ["L5", "L6", "L7"]:
        problems = TEST_PROBLEMS[level]

        for problem_idx, problem in enumerate(problems):
            for run_idx in range(5):  # 5 runs per problem
                result = run_single_test(level, problem_idx, problem, run_idx)
                results.append(result)

                # Print immediate result
                status_icon = "✓" if result["status"] == "success" else "✗"
                print(f"\n{status_icon} {level} P{problem_idx+1} R{run_idx+1}: {result['status']} in {result['duration']:.1f}s ({result['files_created']} files)")

    overall_end = time.time()
    overall_duration = overall_end - overall_start

    # Save detailed results
    output_file = Path("evaluation_results") / f"l5_l7_x5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "total_duration": overall_duration,
            "results": results,
        }, f, indent=2)

    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    for level in ["L5", "L6", "L7"]:
        level_results = [r for r in results if r["level"] == level]
        successes = [r for r in level_results if r["status"] == "success"]

        success_rate = len(successes) / len(level_results) * 100 if level_results else 0
        avg_duration = sum(r["duration"] for r in level_results) / len(level_results) if level_results else 0
        avg_files = sum(r["files_created"] for r in successes) / len(successes) if successes else 0

        print(f"\n{level}:")
        print(f"  Success Rate: {success_rate:.1f}% ({len(successes)}/{len(level_results)})")
        print(f"  Avg Duration: {avg_duration:.1f}s")
        print(f"  Avg Files:    {avg_files:.1f}")

    # Overall stats
    total_successes = [r for r in results if r["status"] == "success"]
    overall_success_rate = len(total_successes) / len(results) * 100

    print("\nOVERALL:")
    print(f"  Success Rate: {overall_success_rate:.1f}% ({len(total_successes)}/{len(results)})")
    print(f"  Total Time:   {overall_duration/3600:.1f} hours")
    print(f"  Avg Test:     {overall_duration/len(results):.1f}s")

    # Time analysis
    print("\nTIME BREAKDOWN:")
    durations = [r["duration"] for r in results]
    print(f"  Min:    {min(durations):.1f}s")
    print(f"  Max:    {max(durations):.1f}s")
    print(f"  Median: {sorted(durations)[len(durations)//2]:.1f}s")

    # Check for long-running tests (old problem: 40 minutes)
    long_tests = [r for r in results if r["duration"] > 300]  # > 5 minutes
    if long_tests:
        print(f"\n  ⚠️  {len(long_tests)} tests took >5 minutes:")
        for r in long_tests:
            print(f"    - {r['level']} P{r['problem_idx']+1} R{r['run_idx']+1}: {r['duration']:.1f}s ({r['status']})")
    else:
        print("\n  ✓ No tests exceeded 5 minutes")

    # Empty round and error feedback analysis
    print("\nEMPTY ROUND DETECTION:")
    print("  Check logs for:")
    print("    - '[loop_detection] ⚠️  Empty round #N'")
    print("    - '🚨 CRITICAL: You have not called ANY tools'")
    print("    - 'ERROR: Your last response had a malformed tool call'")

    print(f"\nDetailed results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
