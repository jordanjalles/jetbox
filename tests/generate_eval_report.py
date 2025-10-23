#!/usr/bin/env python3
"""Generate comprehensive analysis report from eval suite results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_results() -> list[dict[str, Any]]:
    """Load results from eval suite."""
    if not Path("eval_suite_results.json").exists():
        print("Error: eval_suite_results.json not found!")
        return []

    with open("eval_suite_results.json") as f:
        return json.load(f)


def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze results and compute statistics."""
    analysis = {
        "total_tests": 0,
        "total_passed": 0,
        "by_test": {},
        "by_level": {},
        "by_iteration": {},
        "failure_modes": {},
        "timing_stats": {},
    }

    # Aggregate by test ID
    for result in results:
        test_id = result.get("id", "unknown")
        level = result.get("level", 0)
        iteration = result.get("iteration", 0)
        success = result.get("success", False)
        duration = result.get("duration", 0)
        rounds = result.get("rounds", 0)
        failure_mode = result.get("failure_mode")

        # Overall
        analysis["total_tests"] += 1
        if success:
            analysis["total_passed"] += 1

        # By test
        if test_id not in analysis["by_test"]:
            analysis["by_test"][test_id] = {
                "name": result.get("name", ""),
                "level": level,
                "task": result.get("task", ""),
                "successes": 0,
                "failures": 0,
                "durations": [],
                "rounds": [],
                "failure_modes": [],
            }

        test_data = analysis["by_test"][test_id]
        if success:
            test_data["successes"] += 1
        else:
            test_data["failures"] += 1
            if failure_mode:
                test_data["failure_modes"].append(failure_mode)

        test_data["durations"].append(duration)
        test_data["rounds"].append(rounds)

        # By level
        if level not in analysis["by_level"]:
            analysis["by_level"][level] = {"total": 0, "passed": 0}
        analysis["by_level"][level]["total"] += 1
        if success:
            analysis["by_level"][level]["passed"] += 1

        # By iteration
        if iteration not in analysis["by_iteration"]:
            analysis["by_iteration"][iteration] = {"total": 0, "passed": 0}
        analysis["by_iteration"][iteration]["total"] += 1
        if success:
            analysis["by_iteration"][iteration]["passed"] += 1

        # Failure modes
        if not success and failure_mode:
            if failure_mode not in analysis["failure_modes"]:
                analysis["failure_modes"][failure_mode] = []
            analysis["failure_modes"][failure_mode].append({
                "test_id": test_id,
                "iteration": iteration,
                "rounds": rounds,
                "duration": duration,
            })

    # Calculate averages for each test
    for test_id, data in analysis["by_test"].items():
        n = len(data["durations"])
        data["avg_duration"] = sum(data["durations"]) / n if n > 0 else 0
        data["avg_rounds"] = sum(data["rounds"]) / n if n > 0 else 0
        data["pass_rate"] = data["successes"] / n * 100 if n > 0 else 0

    return analysis


def generate_markdown_report(analysis: dict[str, Any]) -> str:
    """Generate markdown report from analysis."""
    lines = []

    # Header
    lines.append("# Agent Evaluation Report: L3-L4-L5 Tests")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    total = analysis["total_tests"]
    passed = analysis["total_passed"]
    overall_rate = (passed / total * 100) if total > 0 else 0
    lines.append(f"- **Total Test Runs:** {total} ({len(analysis['by_test'])} tests × 5 iterations)")
    lines.append(f"- **Overall Pass Rate:** {passed}/{total} ({overall_rate:.1f}%)")
    lines.append("")

    # Performance by level
    lines.append("### Performance by Difficulty Level")
    lines.append("")
    lines.append("| Level | Description | Pass Rate | Performance |")
    lines.append("|-------|-------------|-----------|-------------|")

    level_names = {3: "Advanced", 4: "Expert", 5: "Extreme"}
    for level in sorted(analysis["by_level"].keys()):
        stats = analysis["by_level"][level]
        pct = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        performance = "🟢 Good" if pct >= 80 else "🟡 Fair" if pct >= 50 else "🔴 Poor"
        lines.append(f"| L{level} | {level_names.get(level, 'Unknown')} | "
                    f"{stats['passed']}/{stats['total']} ({pct:.0f}%) | {performance} |")

    lines.append("")

    # Detailed Results by Test
    lines.append("## Detailed Results by Test")
    lines.append("")

    for test_id in sorted(analysis["by_test"].keys()):
        data = analysis["by_test"][test_id]
        lines.append(f"### {test_id}: {data['name']}")
        lines.append("")
        lines.append(f"**Task:** {data['task']}")
        lines.append("")
        lines.append(f"**Results:** {data['successes']}/5 passed ({data['pass_rate']:.0f}%)")
        lines.append("")

        if data["successes"] > 0:
            lines.append(f"- Average duration: {data['avg_duration']:.1f}s")
            lines.append(f"- Average rounds: {data['avg_rounds']:.1f}")
            lines.append("")

        if data["failure_modes"]:
            from collections import Counter
            mode_counts = Counter(data["failure_modes"])
            lines.append("**Failure modes:**")
            for mode, count in mode_counts.most_common():
                lines.append(f"- {mode}: {count} occurrences")
            lines.append("")

        # Performance rating
        rating = "✅ Excellent" if data["pass_rate"] >= 80 else \
                 "✓ Good" if data["pass_rate"] >= 60 else \
                 "⚠ Fair" if data["pass_rate"] >= 40 else \
                 "❌ Poor"
        lines.append(f"**Rating:** {rating}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Failure Mode Analysis
    if analysis["failure_modes"]:
        lines.append("## Failure Mode Analysis")
        lines.append("")

        for mode in sorted(analysis["failure_modes"].keys()):
            instances = analysis["failure_modes"][mode]
            lines.append(f"### {mode.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"**Occurrences:** {len(instances)}")
            lines.append("")

            # Group by test
            by_test = {}
            for inst in instances:
                test = inst["test_id"]
                if test not in by_test:
                    by_test[test] = 0
                by_test[test] += 1

            lines.append("**Affected tests:**")
            for test, count in sorted(by_test.items(), key=lambda x: -x[1]):
                lines.append(f"- {test}: {count} failures")
            lines.append("")

            # Average metrics for failures
            avg_rounds = sum(i["rounds"] for i in instances) / len(instances)
            avg_duration = sum(i["duration"] for i in instances) / len(instances)
            lines.append(f"**Average rounds before failure:** {avg_rounds:.1f}")
            lines.append(f"**Average time to failure:** {avg_duration:.1f}s")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Iteration Consistency
    lines.append("## Iteration Consistency")
    lines.append("")
    lines.append("Consistency of results across 5 iterations:")
    lines.append("")
    lines.append("| Iteration | Pass Rate |")
    lines.append("|-----------|-----------|")

    for iteration in sorted(analysis["by_iteration"].keys()):
        stats = analysis["by_iteration"][iteration]
        pct = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        lines.append(f"| {iteration} | {stats['passed']}/{stats['total']} ({pct:.0f}%) |")

    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    # Based on overall performance
    if overall_rate < 50:
        lines.append("### Critical Issues")
        lines.append("")
        lines.append("The agent is failing more than 50% of tests. Priority improvements:")
        lines.append("")
        lines.append("1. **Task Decomposition:** Review how tasks are broken down into subtasks")
        lines.append("2. **Completion Detection:** Verify completion signals are working properly")
        lines.append("3. **Loop Prevention:** Check if agent is getting stuck in loops")
        lines.append("")

    # Based on failure modes
    if "max_rounds_exceeded" in analysis["failure_modes"]:
        lines.append("### Max Rounds Issues")
        lines.append("")
        max_rounds_failures = len(analysis["failure_modes"]["max_rounds_exceeded"])
        lines.append(f"**{max_rounds_failures} tests** hit the max rounds limit. Consider:")
        lines.append("")
        lines.append("- Increasing `max_per_subtask` in config")
        lines.append("- Improving decomposition to create smaller, more achievable subtasks")
        lines.append("- Adding more aggressive timeout/escalation logic")
        lines.append("")

    if "timeout" in analysis["failure_modes"]:
        lines.append("### Timeout Issues")
        lines.append("")
        timeout_failures = len(analysis["failure_modes"]["timeout"])
        lines.append(f"**{timeout_failures} tests** timed out. Consider:")
        lines.append("")
        lines.append("- Increasing test timeout limits")
        lines.append("- Optimizing LLM response times")
        lines.append("- Reducing context size to speed up inference")
        lines.append("")

    # Based on level performance
    for level, stats in analysis["by_level"].items():
        pct = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        if pct < 40:
            lines.append(f"### Level {level} Struggles")
            lines.append("")
            lines.append(f"Level {level} tests have a {pct:.0f}% pass rate. These tests may require:")
            lines.append("")
            lines.append("- More sophisticated planning strategies")
            lines.append("- Better context management")
            lines.append("- Increased depth limits for task hierarchy")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Appendix: Configuration
    lines.append("## Appendix: Agent Configuration")
    lines.append("")
    lines.append("```yaml")
    lines.append("# Configuration used during evaluation")
    lines.append("rounds:")
    lines.append("  max_per_subtask: 12")
    lines.append("  max_per_task: 128")
    lines.append("  max_global: 24")
    lines.append("")
    lines.append("hierarchy:")
    lines.append("  max_depth: 5")
    lines.append("  max_siblings: 8")
    lines.append("")
    lines.append("escalation:")
    lines.append("  strategy: force_decompose")
    lines.append("  zoom_out_target: root")
    lines.append("  max_approach_retries: 3")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main():
    """Generate the report."""
    print("Loading evaluation results...")
    results = load_results()

    if not results:
        print("No results to analyze!")
        return

    print(f"Analyzing {len(results)} test runs...")
    analysis = analyze_results(results)

    print("Generating markdown report...")
    report = generate_markdown_report(analysis)

    # Save report
    report_path = "EVAL_SUITE_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport generated: {report_path}")
    print(f"\nQuick stats:")
    print(f"  Total runs: {analysis['total_tests']}")
    print(f"  Passed: {analysis['total_passed']}")
    print(f"  Pass rate: {analysis['total_passed']/analysis['total_tests']*100:.1f}%")


if __name__ == "__main__":
    main()
