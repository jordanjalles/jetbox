#!/usr/bin/env python3
"""Overnight autonomous workload orchestrator.

Runs ~8 hours of autonomous agent testing and HRM-JEPA development.
Generates comprehensive analysis and improvement recommendations.
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Import test suites
from run_stress_tests import TESTS as BASE_TESTS, run_test, clean_workspace
from extended_tests import EXTENDED_TESTS
from hrm_jepa_tasks import HRM_JEPA_TASKS

# Configuration
RESULTS_DIR = Path("overnight_results")
LOGS_DIR = RESULTS_DIR / "logs"
RUNS_DIR = RESULTS_DIR / "runs"
PLOTS_DIR = RESULTS_DIR / "plots"
FAILED_WORKSPACES_DIR = RESULTS_DIR / "workspaces" / "failed"

MIN_DISK_SPACE_GB = 5
SESSION_ID = f"overnight-{datetime.now().strftime('%Y-%m-%d-%H%M')}"


def check_environment() -> bool:
    """Validate environment before starting."""
    print("=" * 70)
    print("ENVIRONMENT VALIDATION")
    print("=" * 70)

    # Check disk space
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    print(f"✓ Free disk space: {free_gb:.1f} GB")
    if free_gb < MIN_DISK_SPACE_GB:
        print(f"✗ ERROR: Need at least {MIN_DISK_SPACE_GB}GB free")
        return False

    # Check Ollama
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ Ollama is running")
            if "gpt-oss:20b" in result.stdout:
                print("✓ gpt-oss:20b model available")
            else:
                print("⚠ Warning: gpt-oss:20b not found, some tests may fail")
        else:
            print("⚠ Warning: Ollama not responding properly")
    except Exception as e:
        print(f"⚠ Warning: Could not check Ollama: {e}")

    # Check Python and pytest
    try:
        subprocess.run(["python", "--version"], check=True, capture_output=True)
        print("✓ Python available")
    except:
        print("✗ ERROR: Python not found")
        return False

    try:
        subprocess.run(["pytest", "--version"], check=True, capture_output=True)
        print("✓ Pytest available")
    except:
        print("⚠ Warning: Pytest not found, verification tests may fail")

    print()
    return True


def setup_directories() -> None:
    """Create output directories."""
    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    FAILED_WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)


def run_test_batch(
    tests: list[dict],
    repetitions: int,
    phase_name: str
) -> list[dict]:
    """Run a batch of tests multiple times."""
    print("=" * 70)
    print(f"PHASE: {phase_name}")
    print(f"Tests: {len(tests)} x {repetitions} repetitions")
    print("=" * 70)
    print()

    results = []
    total = len(tests) * repetitions
    completed = 0

    start_time = time.time()

    for test in tests:
        for rep in range(repetitions):
            completed += 1
            run_id = f"{test['id']}-run-{rep+1:04d}"

            print(f"[{completed}/{total}] Running {run_id}...")
            print(f"  {test['name']}")

            # Run test
            try:
                result = run_test(test)
                result["run_id"] = run_id
                result["repetition"] = rep + 1
                result["phase"] = phase_name

                # Save individual result
                result_file = RUNS_DIR / f"{run_id}.json"
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)

                # Save output log
                log_file = LOGS_DIR / f"{run_id}.txt"
                log_file.write_text(result.get("output", ""))

                # If failed, save workspace snapshot
                if not result["success"]:
                    workspace_pattern = f".agent_workspace/*"
                    workspaces = list(Path(".").glob(workspace_pattern))
                    if workspaces:
                        latest = max(workspaces, key=lambda p: p.stat().st_mtime)
                        dest = FAILED_WORKSPACES_DIR / run_id
                        if latest.exists():
                            shutil.copytree(latest, dest, dirs_exist_ok=True)

                results.append(result)

                status = "✓ PASS" if result["success"] else "✗ FAIL"
                print(f"  {status} - {result['duration']:.1f}s")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                result = {
                    "run_id": run_id,
                    "id": test["id"],
                    "name": test["name"],
                    "repetition": rep + 1,
                    "phase": phase_name,
                    "success": False,
                    "failure_mode": "execution_error",
                    "error": str(e),
                    "duration": 0,
                }
                results.append(result)

            print()

    elapsed = time.time() - start_time
    print(f"Phase completed in {elapsed/60:.1f} minutes")
    print()

    return results


def run_hrm_jepa_task(task: dict) -> dict:
    """Run a single HRM-JEPA development task."""
    print("=" * 70)
    print(f"HRM-JEPA TASK: {task['id']} - {task['name']}")
    print("=" * 70)
    print()
    print(f"Goal: {task['goal']}")
    print()

    start_time = time.time()

    # Format task as agent goal
    agent_task = f"{task['goal']}. {task['task']}"

    try:
        # Run agent with HRM task
        proc = subprocess.run(
            ["python", "agent.py", agent_task],
            capture_output=True,
            text=True,
            timeout=task.get("timeout", 1800),
            cwd=task.get("working_dir", ".")
        )

        duration = time.time() - start_time
        output = proc.stdout + proc.stderr

        # Check for success indicators
        success = False
        if "Goal achieved" in output or "Complete" in output:
            success = True

        # Check if expected outputs were created
        outputs_created = []
        if "working_dir" in task:
            base_dir = Path(task["working_dir"])
            for expected in task.get("expected_outputs", []):
                if (base_dir / expected).exists():
                    outputs_created.append(expected)

        result = {
            "task_id": task["id"],
            "name": task["name"],
            "category": task["category"],
            "success": success,
            "duration": duration,
            "outputs_created": outputs_created,
            "outputs_expected": task.get("expected_outputs", []),
            "output": output,
        }

        # Save result
        result_file = RUNS_DIR / f"HRM-{task['id']}.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        log_file = LOGS_DIR / f"HRM-{task['id']}.txt"
        log_file.write_text(output)

        status = "✓ SUCCESS" if success else "⚠ PARTIAL"
        print(f"\n{status} - {duration/60:.1f} minutes")
        print(f"Outputs created: {len(outputs_created)}/{len(task.get('expected_outputs', []))}")

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        result = {
            "task_id": task["id"],
            "name": task["name"],
            "category": task["category"],
            "success": False,
            "duration": duration,
            "error": "timeout",
        }
        print(f"\n✗ TIMEOUT after {duration/60:.1f} minutes")

    except Exception as e:
        duration = time.time() - start_time
        result = {
            "task_id": task["id"],
            "name": task["name"],
            "category": task["category"],
            "success": False,
            "duration": duration,
            "error": str(e),
        }
        print(f"\n✗ ERROR: {e}")

    print()
    return result


def generate_statistics(all_results: list[dict]) -> dict:
    """Generate statistical analysis of results."""
    stats = {
        "total_runs": len(all_results),
        "total_passed": sum(1 for r in all_results if r.get("success")),
        "total_failed": sum(1 for r in all_results if not r.get("success")),
        "overall_pass_rate": 0,
        "by_level": {},
        "by_test_id": {},
        "failure_modes": {},
        "performance": {
            "mean_duration": 0,
            "median_duration": 0,
            "mean_rounds": 0,
        }
    }

    if all_results:
        stats["overall_pass_rate"] = stats["total_passed"] / stats["total_runs"]

        # By level
        by_level = {}
        for r in all_results:
            level = r.get("level", 0)
            if level not in by_level:
                by_level[level] = {"total": 0, "passed": 0}
            by_level[level]["total"] += 1
            if r.get("success"):
                by_level[level]["passed"] += 1

        for level, data in by_level.items():
            stats["by_level"][level] = {
                "passed": data["passed"],
                "total": data["total"],
                "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 0
            }

        # By test ID (aggregate repetitions)
        by_test = {}
        for r in all_results:
            test_id = r.get("id", "unknown")
            if test_id not in by_test:
                by_test[test_id] = {"total": 0, "passed": 0, "durations": []}
            by_test[test_id]["total"] += 1
            if r.get("success"):
                by_test[test_id]["passed"] += 1
            by_test[test_id]["durations"].append(r.get("duration", 0))

        for test_id, data in by_test.items():
            stats["by_test_id"][test_id] = {
                "passed": data["passed"],
                "total": data["total"],
                "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 0,
                "mean_duration": sum(data["durations"]) / len(data["durations"]) if data["durations"] else 0,
            }

        # Failure modes
        for r in all_results:
            if not r.get("success"):
                mode = r.get("failure_mode", "unknown")
                stats["failure_modes"][mode] = stats["failure_modes"].get(mode, 0) + 1

        # Performance
        durations = [r.get("duration", 0) for r in all_results]
        rounds = [r.get("rounds", 0) for r in all_results if r.get("rounds")]

        if durations:
            stats["performance"]["mean_duration"] = sum(durations) / len(durations)
            stats["performance"]["median_duration"] = sorted(durations)[len(durations)//2]
        if rounds:
            stats["performance"]["mean_rounds"] = sum(rounds) / len(rounds)

    return stats


def generate_report(
    all_results: list[dict],
    hrm_results: list[dict],
    stats: dict,
    session_info: dict
) -> str:
    """Generate comprehensive markdown report."""
    report = f"""# Overnight Autonomous Testing Report

**Session ID:** {session_info['session_id']}
**Start Time:** {session_info['start_time']}
**End Time:** {session_info['end_time']}
**Duration:** {session_info['duration_hours']:.1f} hours

## Executive Summary

- **Total Test Runs:** {stats['total_runs']}
- **Passed:** {stats['total_passed']} ({stats['overall_pass_rate']*100:.1f}%)
- **Failed:** {stats['total_failed']}
- **HRM-JEPA Tasks:** {len(hrm_results)} attempted

## Results by Level

"""

    for level in sorted(stats["by_level"].keys()):
        data = stats["by_level"][level]
        report += f"- **Level {level}:** {data['passed']}/{data['total']} ({data['pass_rate']*100:.0f}%)\n"

    report += f"""

## Test Stability Analysis

Tests run multiple times to assess reliability:

"""

    # Find flaky tests (inconsistent results)
    flaky_tests = []
    consistent_pass = []
    consistent_fail = []

    for test_id, data in stats["by_test_id"].items():
        if data["total"] > 1:
            if 0 < data["pass_rate"] < 1:
                flaky_tests.append((test_id, data))
            elif data["pass_rate"] == 1:
                consistent_pass.append((test_id, data))
            else:
                consistent_fail.append((test_id, data))

    if flaky_tests:
        report += "### Flaky Tests (Inconsistent Results)\n\n"
        for test_id, data in flaky_tests:
            report += f"- **{test_id}:** {data['passed']}/{data['total']} passed ({data['pass_rate']*100:.0f}%) - INVESTIGATE\n"
        report += "\n"

    if consistent_pass:
        report += f"### Consistently Passing ({len(consistent_pass)} tests)\n\n"
        report += "All repetitions passed. Reliable tests.\n\n"

    if consistent_fail:
        report += "### Consistently Failing\n\n"
        for test_id, data in consistent_fail:
            report += f"- **{test_id}:** 0/{data['total']} passed - NEEDS FIX\n"
        report += "\n"

    report += """
## Failure Modes

"""

    if stats["failure_modes"]:
        for mode, count in sorted(stats["failure_modes"].items(), key=lambda x: -x[1]):
            report += f"- **{mode}:** {count} occurrences\n"
    else:
        report += "No failures recorded.\n"

    report += f"""

## Performance Metrics

- **Mean Duration:** {stats['performance']['mean_duration']:.1f}s
- **Median Duration:** {stats['performance']['median_duration']:.1f}s
- **Mean Rounds:** {stats['performance']['mean_rounds']:.1f}

## HRM-JEPA Development Tasks

"""

    if hrm_results:
        hrm_success = sum(1 for r in hrm_results if r.get("success"))
        report += f"**Completed:** {hrm_success}/{len(hrm_results)}\n\n"

        for r in hrm_results:
            status = "✓" if r.get("success") else "⚠" if r.get("outputs_created") else "✗"
            report += f"- {status} **{r['task_id']}:** {r['name']} ({r['duration']/60:.1f} min)\n"
            if r.get("outputs_created"):
                report += f"  - Created: {', '.join(r['outputs_created'])}\n"
        report += "\n"
    else:
        report += "No HRM-JEPA tasks completed.\n\n"

    report += """
## Top Recommendations

Based on this overnight run:

"""

    # Generate recommendations
    recommendations = []

    # Flaky test recommendations
    if flaky_tests:
        recommendations.append(f"1. **Investigate {len(flaky_tests)} flaky tests** - These show inconsistent behavior and may indicate race conditions, timing issues, or environment dependencies.")

    # Failure mode recommendations
    if "max_rounds_exceeded" in stats["failure_modes"]:
        recommendations.append("2. **Implement Phase 2 escalation** - Many tests hit round limits. Hierarchical decomposition will help.")

    if "infinite_loop" in stats["failure_modes"]:
        recommendations.append("3. **Review loop detection thresholds** - May be too aggressive or missing actual loops.")

    if "verification_failed" in stats["failure_modes"]:
        recommendations.append("4. **Fix test harness verification** - Run verify commands from workspace directory.")

    # HRM recommendations
    if len(hrm_results) < len(HRM_JEPA_TASKS):
        recommendations.append(f"5. **Continue HRM-JEPA development** - {len(HRM_JEPA_TASKS) - len(hrm_results)} tasks remaining.")

    if not recommendations:
        recommendations.append("1. **Excellent results!** - Consider adding more challenging tests.")

    for rec in recommendations:
        report += rec + "\n\n"

    report += """
## Next Steps

1. Review flaky tests and fix environmental issues
2. Implement quick wins from FAILURE_DIAGNOSIS.md
3. Continue Phase 2 implementation (hierarchical escalation)
4. Expand HRM-JEPA development based on completed tasks

---

*Generated by overnight autonomous testing system*
"""

    return report


def main() -> None:
    """Main orchestration function."""
    session_start = datetime.now()

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║            OVERNIGHT AUTONOMOUS WORKLOAD STARTING                    ║
║                Session: {SESSION_ID}                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Validate environment
    if not check_environment():
        print("✗ Environment validation failed. Aborting.")
        sys.exit(1)

    # Setup directories
    setup_directories()

    all_results = []
    hrm_results = []

    # Phase 1: Extended stress tests (L3-L5 x10)
    l3_tests = [t for t in BASE_TESTS if t["level"] == 3]
    l4_tests = [t for t in BASE_TESTS if t["level"] == 4]
    l5_tests = [t for t in BASE_TESTS if t["level"] == 5]

    if l3_tests:
        results = run_test_batch(l3_tests, repetitions=10, phase_name="L3 Extended (x10)")
        all_results.extend(results)

    if l4_tests:
        results = run_test_batch(l4_tests, repetitions=10, phase_name="L4 Extended (x10)")
        all_results.extend(results)

    if l5_tests:
        results = run_test_batch(l5_tests, repetitions=10, phase_name="L5 Extended (x10)")
        all_results.extend(results)

    # Phase 2: L6 extreme challenges (x5)
    l6_tests = [t for t in EXTENDED_TESTS if t["level"] == 6]
    if l6_tests:
        results = run_test_batch(l6_tests, repetitions=5, phase_name="L6 Extreme (x5)")
        all_results.extend(results)

    # Phase 3: HRM-JEPA tasks
    print("=" * 70)
    print("PHASE: HRM-JEPA Development")
    print("=" * 70)
    print()

    for task in HRM_JEPA_TASKS:
        result = run_hrm_jepa_task(task)
        hrm_results.append(result)

    # Phase 4: L7-L8 if time permits (x3)
    l7_tests = [t for t in EXTENDED_TESTS if t["level"] == 7]
    if l7_tests:
        results = run_test_batch(l7_tests, repetitions=3, phase_name="L7 Algorithmic (x3)")
        all_results.extend(results)

    l8_tests = [t for t in EXTENDED_TESTS if t["level"] == 8]
    if l8_tests:
        results = run_test_batch(l8_tests, repetitions=3, phase_name="L8 System Design (x3)")
        all_results.extend(results)

    # Generate statistics and report
    session_end = datetime.now()
    session_info = {
        "session_id": SESSION_ID,
        "start_time": session_start.isoformat(),
        "end_time": session_end.isoformat(),
        "duration_hours": (session_end - session_start).total_seconds() / 3600,
    }

    stats = generate_statistics(all_results)

    # Save master log
    master_log = {
        "session": session_info,
        "statistics": stats,
        "test_results_summary": {
            "total_runs": len(all_results),
            "passed": stats["total_passed"],
            "failed": stats["total_failed"],
        },
        "hrm_results_summary": {
            "total_tasks": len(hrm_results),
            "successful": sum(1 for r in hrm_results if r.get("success")),
        }
    }

    master_log_file = RESULTS_DIR / "master_log.json"
    with open(master_log_file, "w") as f:
        json.dump(master_log, f, indent=2)

    # Generate report
    report = generate_report(all_results, hrm_results, stats, session_info)
    report_file = RESULTS_DIR / f"overnight_report_{datetime.now().strftime('%Y-%m-%d')}.md"
    report_file.write_text(report)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║            OVERNIGHT AUTONOMOUS WORKLOAD COMPLETE                    ║
╚══════════════════════════════════════════════════════════════════════╝

Duration: {session_info['duration_hours']:.1f} hours
Tests Run: {len(all_results)}
Pass Rate: {stats['overall_pass_rate']*100:.1f}%

Report: {report_file}
Results: {RESULTS_DIR}/

""")


if __name__ == "__main__":
    main()
