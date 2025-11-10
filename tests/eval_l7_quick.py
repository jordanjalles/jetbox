#!/usr/bin/env python3
"""
L7 Quick Evaluation (5 runs per problem)

Test L7 delegation fixes:
- Orchestrator delegation nudge after 2 empty rounds
- ChatbotBehavior excluded in autonomous mode
- Context compaction with emergency mode
- Invalid parameter validation with spec feedback

L7: 3 problems × 5 runs = 15 tests
Expected duration: 1-2 hours
"""
import json
import time
from pathlib import Path
from datetime import datetime
from agents.orchestrator_agent import OrchestratorAgent

# L7 Test Problems
TEST_PROBLEMS = [
    "Create a full-stack Flask app with user auth, posts, and comments. Use SQLite. Include frontend templates.",
    "Create a project management system with users, projects, tasks. Use SQLite. Include tests and docs.",
    "Create a collaborative todo app with sharing and permissions. Use SQLite. Include tests and API docs.",
]

def run_single_test(problem_idx: int, problem: str, run_idx: int) -> dict:
    """Run a single test problem."""
    print(f"\n{'='*80}")
    print(f"L7 Problem {problem_idx+1} Run {run_idx+1}")
    print(f"{'='*80}")
    print(f"Task: {problem}")
    print()

    start_time = time.time()

    # Create workspace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_name = f"l7_p{problem_idx+1}_run{run_idx+1}_{timestamp}"
    workspace = Path(".agent_workspaces") / workspace_name

    # Run test with goal string (autonomous mode - excludes ChatbotBehavior)
    try:
        orchestrator = OrchestratorAgent(
            workspace=workspace,
            exclude_behaviors=["ChatbotBehavior"]  # Explicit autonomous mode
        )
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
    """Run L7 evaluation with 5 runs per problem."""
    print("="*80)
    print("L7 QUICK EVALUATION (x5) - Testing Fixes")
    print("="*80)
    print()
    print("Testing fixes:")
    print("  ✓ Orchestrator delegation nudge after 2 empty rounds")
    print("  ✓ ChatbotBehavior excluded in autonomous mode")
    print("  ✓ Context compaction emergency mode (>100%)")
    print("  ✓ Invalid parameter validation with spec feedback")
    print("  ✓ Workspace reuse on retry")
    print()
    print("Problems: 3")
    print("Runs per problem: 5")
    print("Total tests: 15")
    print("Expected duration: 1-2 hours")
    print()
    print("="*80)

    overall_start = time.time()
    results = []

    # Run all tests
    for problem_idx, problem in enumerate(TEST_PROBLEMS):
        for run_idx in range(5):  # 5 runs per problem
            result = run_single_test(problem_idx, problem, run_idx)
            results.append(result)

            # Print immediate result
            status_icon = "✓" if result["status"] == "success" else "✗"
            print(f"\n{status_icon} L7 P{problem_idx+1} R{run_idx+1}: {result['status']} in {result['duration']:.1f}s ({result['files_created']} files)")

    overall_end = time.time()
    overall_duration = overall_end - overall_start

    # Save detailed results
    output_file = Path("evaluation_results") / f"l7_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "level": "L7",
            "total_tests": len(results),
            "total_duration": overall_duration,
            "results": results,
        }, f, indent=2)

    # Generate summary report
    print("\n" + "="*80)
    print("L7 QUICK EVALUATION SUMMARY")
    print("="*80)

    successes = [r for r in results if r["status"] == "success"]
    success_rate = len(successes) / len(results) * 100 if results else 0
    avg_duration = sum(r["duration"] for r in results) / len(results) if results else 0
    avg_files = sum(r["files_created"] for r in successes) / len(successes) if successes else 0

    print("\nL7 Results:")
    print(f"  Success Rate: {success_rate:.1f}% ({len(successes)}/{len(results)})")
    print(f"  Avg Duration: {avg_duration:.1f}s ({avg_duration/60:.1f}m)")
    print(f"  Avg Files:    {avg_files:.1f}")
    print(f"  Total Time:   {overall_duration/3600:.1f} hours")

    # Compare to baseline (0% success from overnight eval)
    print("\nIMPROVEMENT:")
    print("  Baseline (overnight eval):  0.0% (0/15)")
    print(f"  Current (with fixes):       {success_rate:.1f}% ({len(successes)}/{len(results)})")

    if success_rate > 0:
        print(f"  ✓ SUCCESS: L7 tests now working! ({len(successes)} successes)")
    else:
        print("  ✗ Still failing - need more investigation")

    # Time analysis
    print("\nTIME BREAKDOWN:")
    durations = [r["duration"] for r in results]
    print(f"  Min:    {min(durations):.1f}s")
    print(f"  Max:    {max(durations):.1f}s")
    print(f"  Median: {sorted(durations)[len(durations)//2]:.1f}s")

    # Check for timeouts (old problem: all L7 timed out)
    timeout_tests = [r for r in results if r["status"] == "timeout"]
    if timeout_tests:
        print(f"\n  ⚠️  {len(timeout_tests)}/{len(results)} tests timed out:")
        for r in timeout_tests:
            print(f"    - P{r['problem_idx']+1} R{r['run_idx']+1}: {r['duration']:.1f}s")
    else:
        print("\n  ✓ No timeouts!")

    # Per-problem breakdown
    print("\nPER-PROBLEM BREAKDOWN:")
    for problem_idx in range(3):
        problem_results = [r for r in results if r["problem_idx"] == problem_idx]
        problem_successes = [r for r in problem_results if r["status"] == "success"]
        problem_rate = len(problem_successes) / len(problem_results) * 100 if problem_results else 0

        print(f"\n  Problem {problem_idx+1}: {problem_rate:.1f}% ({len(problem_successes)}/5)")
        print(f"    \"{TEST_PROBLEMS[problem_idx][:60]}...\"")

    print(f"\nDetailed results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
