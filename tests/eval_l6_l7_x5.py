#!/usr/bin/env python3
"""
L6-L7 Evaluation x5 runs per problem

Tests orchestrator delegation with complex multi-component projects.
Each problem gets 5 runs with 10-minute timeout per run.
"""
import json
import time
from pathlib import Path
from datetime import datetime
from orchestrator_agent import OrchestratorAgent

# L6-L7 Test Problems (orchestrator-level complexity)
TEST_PROBLEMS = {
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

def run_single_test(level: str, problem_idx: int, problem: str, run_idx: int, timeout_seconds: int = 600) -> dict:
    """Run a single test problem with timeout."""
    print(f"\n{'='*80}")
    print(f"{level} Problem {problem_idx+1} Run {run_idx+1}")
    print(f"{'='*80}")
    print(f"Task: {problem[:70]}...")
    print(f"Timeout: {timeout_seconds}s ({timeout_seconds//60} minutes)")
    print()

    start_time = time.time()

    # Create workspace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_name = f"{level.lower()}_p{problem_idx+1}_run{run_idx+1}_{timestamp}"
    workspace = Path(".agent_workspaces") / workspace_name

    # Run test with timeout
    from threading import Thread
    from queue import Queue

    result_queue = Queue()

    def run_orchestrator():
        try:
            orchestrator = OrchestratorAgent(workspace=workspace)
            orchestrator.add_message({
                "role": "user",
                "content": problem
            })
            result = orchestrator.run(max_rounds=20)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", e))

    thread = Thread(target=run_orchestrator, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    end_time = time.time()
    duration = end_time - start_time

    # Check if thread completed
    if thread.is_alive():
        print(f"⚠️  TIMEOUT: Test exceeded {timeout_seconds}s - thread still running")
        # Count files created
        files_created = 0
        if workspace.exists():
            files_created = len([f for f in workspace.rglob("*") if f.is_file() and f.name != "workspace_task_notes.md"])

        return {
            "level": level,
            "problem_idx": problem_idx,
            "run_idx": run_idx,
            "problem": problem,
            "status": "timeout",
            "duration": duration,
            "files_created": files_created,
            "workspace": str(workspace),
            "error": f"Test exceeded {timeout_seconds}s timeout",
        }

    # Thread completed - get result
    if not result_queue.empty():
        status, result = result_queue.get()

        # Count files created
        files_created = 0
        if workspace.exists():
            files_created = len([f for f in workspace.rglob("*") if f.is_file() and f.name != "workspace_task_notes.md"])

        if status == "success":
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
        else:  # error
            return {
                "level": level,
                "problem_idx": problem_idx,
                "run_idx": run_idx,
                "problem": problem,
                "status": "error",
                "duration": duration,
                "files_created": 0,
                "workspace": str(workspace),
                "error": str(result),
            }
    else:
        # Thread died without result
        return {
            "level": level,
            "problem_idx": problem_idx,
            "run_idx": run_idx,
            "problem": problem,
            "status": "error",
            "duration": duration,
            "files_created": 0,
            "workspace": str(workspace),
            "error": "Thread completed but no result available",
        }

def main():
    """Run L6-L7 evaluation."""
    print("="*80)
    print("L6-L7 EVALUATION - 5 RUNS PER PROBLEM")
    print("="*80)
    print()

    all_results = []
    start_time = time.time()

    # Run tests
    for level in ["L6", "L7"]:
        problems = TEST_PROBLEMS[level]
        for prob_idx, problem in enumerate(problems):
            for run_idx in range(5):  # 5 runs per problem
                result = run_single_test(level, prob_idx, problem, run_idx, timeout_seconds=600)
                all_results.append(result)

                # Print result
                status_icon = "✓" if result["status"] == "success" else "✗"
                print(f"{status_icon} {level} P{prob_idx+1} R{run_idx+1}: {result['status']} in {result['duration']:.1f}s ({result['files_created']} files)")

    # Save results
    output_file = Path("evaluation_results") / f"l6_l7_x5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "total_duration": time.time() - start_time,
            "results": all_results,
        }, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    for level in ["L6", "L7"]:
        level_results = [r for r in all_results if r["level"] == level]
        success_count = sum(1 for r in level_results if r["status"] == "success")
        timeout_count = sum(1 for r in level_results if r["status"] == "timeout")
        error_count = sum(1 for r in level_results if r["status"] == "error")
        total = len(level_results)

        print(f"\n{level}: {success_count}/{total} success ({success_count/total*100:.1f}%)")
        print(f"  Success: {success_count}, Timeout: {timeout_count}, Error: {error_count}")

        # Success rate by problem
        for prob_idx in range(len(TEST_PROBLEMS[level])):
            prob_results = [r for r in level_results if r["problem_idx"] == prob_idx]
            prob_success = sum(1 for r in prob_results if r["status"] == "success")
            print(f"  Problem {prob_idx+1}: {prob_success}/5 success")

    print(f"\nResults saved to: {output_file}")
    print(f"Total duration: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
