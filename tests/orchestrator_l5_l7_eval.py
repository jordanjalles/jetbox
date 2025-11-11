#!/usr/bin/env python3
"""
Orchestrator L5-L7 Evaluation

Tests orchestrator with complex multi-component problems:
- L5: Integration (5 tasks) - Multi-component systems
- L6: Architecture (5 tasks) - Design patterns
- L7: Expert (4 tasks) - Production-grade systems

Total: 14 tasks testing orchestrator delegation and coordination
"""
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.evaluation_suite_extended import get_extended_tasks


def run_task_with_orchestrator(task, timeout_minutes=15):
    """
    Run a single task with orchestrator team.

    Args:
        task: TaskDefinition to run
        timeout_minutes: Timeout in minutes

    Returns:
        dict with results
    """
    workspace = tempfile.mkdtemp(prefix=f"orch_L{task.level}_{task.name}_")
    workspace_path = Path(workspace)

    print(f"\n{'='*80}")
    print(f"Running: L{task.level} - {task.name}")
    print(f"Description: {task.description}")
    print(f"Workspace: {workspace}")
    print(f"{'='*80}")

    start_time = datetime.now()

    try:
        cmd = [
            sys.executable, "/workspace/agent.py",
            "--team", "default",  # Orchestrator team
            "--workspace", workspace,
            task.goal
        ]

        print(f"\nCommand: {' '.join(cmd)}")
        print(f"Timeout: {timeout_minutes} minutes")
        print(f"Starting at: {start_time.strftime('%H:%M:%S')}\n")

        result = subprocess.run(
            cmd,
            cwd="/workspace",
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Check if agent completed successfully
        agent_success = result.returncode == 0

        # Check if expected files were created
        files_created = []
        files_missing = []
        for expected_file in task.expected_files:
            file_path = workspace_path / expected_file
            if file_path.exists():
                files_created.append(expected_file)
            else:
                files_missing.append(expected_file)

        files_success = len(files_missing) == 0

        # Run validation commands
        validation_results = []
        validation_success = True

        for val_cmd in task.validation_commands:
            try:
                val_result = subprocess.run(
                    val_cmd,
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                val_passed = val_result.returncode == 0
                validation_results.append({
                    "command": " ".join(val_cmd),
                    "passed": val_passed,
                    "output": val_result.stdout if val_passed else val_result.stderr
                })
                if not val_passed:
                    validation_success = False
            except Exception as e:
                validation_results.append({
                    "command": " ".join(val_cmd),
                    "passed": False,
                    "error": str(e)
                })
                validation_success = False

        overall_success = agent_success and files_success and validation_success

        # Print results
        status_emoji = "✅" if overall_success else "❌"
        print(f"\n{status_emoji} L{task.level} - {task.name}: {'SUCCESS' if overall_success else 'FAILED'}")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Agent exit: {'✓' if agent_success else '✗'} (code {result.returncode})")
        print(f"   Files: {'✓' if files_success else '✗'} ({len(files_created)}/{len(task.expected_files)})")
        print(f"   Validation: {'✓' if validation_success else '✗'} ({sum(1 for v in validation_results if v['passed'])}/{len(validation_results)})")

        if files_missing:
            print(f"   Missing files: {', '.join(files_missing)}")

        if not validation_success:
            print(f"   Failed validations:")
            for val in validation_results:
                if not val["passed"]:
                    print(f"      - {val['command']}")
                    if "error" in val:
                        print(f"        Error: {val['error']}")

        return {
            "task": task.name,
            "level": task.level,
            "success": overall_success,
            "duration": duration,
            "agent_success": agent_success,
            "files_success": files_success,
            "validation_success": validation_success,
            "files_created": files_created,
            "files_missing": files_missing,
            "validation_results": validation_results,
            "workspace": workspace,
            "stdout_lines": len(result.stdout.splitlines()),
            "stderr_lines": len(result.stderr.splitlines())
        }

    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n❌ L{task.level} - {task.name}: TIMEOUT after {timeout_minutes} minutes")

        return {
            "task": task.name,
            "level": task.level,
            "success": False,
            "duration": duration,
            "agent_success": False,
            "files_success": False,
            "validation_success": False,
            "timeout": True,
            "workspace": workspace
        }

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n❌ L{task.level} - {task.name}: ERROR - {e}")

        return {
            "task": task.name,
            "level": task.level,
            "success": False,
            "duration": duration,
            "agent_success": False,
            "files_success": False,
            "validation_success": False,
            "error": str(e),
            "workspace": workspace
        }


def print_summary(results):
    """Print evaluation summary."""
    print("\n" + "="*80)
    print("ORCHESTRATOR L5-L7 EVALUATION SUMMARY")
    print("="*80)

    # Group by level
    by_level = {}
    for result in results:
        level = result["level"]
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(result)

    # Print per-level summary
    total_success = 0
    total_tasks = 0

    for level in sorted(by_level.keys()):
        level_results = by_level[level]
        level_success = sum(1 for r in level_results if r["success"])
        level_total = len(level_results)
        level_success_rate = (level_success / level_total * 100) if level_total > 0 else 0

        total_success += level_success
        total_tasks += level_total

        print(f"\nLevel {level}:")
        print(f"  Success: {level_success}/{level_total} ({level_success_rate:.1f}%)")
        print(f"  Tasks:")

        for result in level_results:
            status = "✅" if result["success"] else "❌"
            duration = result["duration"]
            print(f"    {status} {result['task']} ({duration:.1f}s)")

    # Overall summary
    overall_success_rate = (total_success / total_tasks * 100) if total_tasks > 0 else 0

    print(f"\n{'='*80}")
    print(f"OVERALL: {total_success}/{total_tasks} ({overall_success_rate:.1f}%) tasks successful")
    print(f"{'='*80}\n")

    return overall_success_rate


def save_results(results, output_file):
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(results),
            "successful_tasks": sum(1 for r in results if r["success"]),
            "results": results
        }, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    """Run orchestrator L5-L7 evaluation."""
    print("="*80)
    print("ORCHESTRATOR L5-L7 EVALUATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Team: default (orchestrator → architect + task_executor)")
    print("="*80)

    # Get L5-L7 tasks
    all_tasks = get_extended_tasks()
    l5_l7_tasks = [t for t in all_tasks if t.level >= 5]

    print(f"\nTasks to run:")
    for level in [5, 6, 7]:
        level_tasks = [t for t in l5_l7_tasks if t.level == level]
        print(f"  L{level}: {len(level_tasks)} tasks")
        for task in level_tasks:
            print(f"    - {task.name}: {task.description}")

    print(f"\nTotal: {len(l5_l7_tasks)} tasks")

    # Confirm (skip if not running interactively)
    if sys.stdin.isatty():
        response = input("\nProceed with evaluation? (y/n): ")
        if response.lower() != 'y':
            print("Evaluation cancelled.")
            return
    else:
        print("\nRunning in non-interactive mode, proceeding automatically...")

    # Run tasks
    results = []

    for i, task in enumerate(l5_l7_tasks, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Task {i}/{len(l5_l7_tasks)}")
        print(f"{'#'*80}")

        # Timeout based on level
        timeout = {5: 15, 6: 20, 7: 25}.get(task.level, 15)

        result = run_task_with_orchestrator(task, timeout_minutes=timeout)
        results.append(result)

        # Save incremental results
        save_results(results, f"evaluation_results/orchestrator_l5_l7_incremental.json")

    # Print summary
    success_rate = print_summary(results)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"evaluation_results/orchestrator_l5_l7_{timestamp}.json")

    print(f"\nEvaluation complete!")
    print(f"Success rate: {success_rate:.1f}%")

    return 0 if success_rate >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
