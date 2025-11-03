#!/usr/bin/env python3
"""
Runner script for the comprehensive project evaluation suite.

Executes all L5-L8 tests and generates summary reports.
"""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def run_evaluation_suite(level_filter=None, task_filter=None):
    """
    Run the full evaluation suite.

    Args:
        level_filter: Optional level filter (e.g., "L5", "L6")
        task_filter: Optional task ID filter (e.g., "L5_json_csv_converter")
    """
    print("=" * 80)
    print("JETBOX PROJECT EVALUATION SUITE")
    print("=" * 80)
    print()

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", "tests/test_project_evaluation.py", "-v", "--tb=short"]

    # Add filters if specified
    if task_filter:
        cmd.extend(["-k", task_filter])
    elif level_filter:
        cmd.extend(["-k", level_filter.lower()])

    print(f"Running: {' '.join(cmd)}")
    print()

    # Clear previous results
    results_file = Path("evaluation_results/project_eval_results.jsonl")
    if results_file.exists():
        results_file.unlink()
        print(f"Cleared previous results: {results_file}")
        print()

    # Run pytest
    start_time = datetime.now()
    proc = subprocess.run(cmd)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print()
    print("=" * 80)
    print(f"EVALUATION COMPLETED IN {duration:.1f}s")
    print("=" * 80)
    print()

    # Generate summary report
    if results_file.exists():
        generate_summary_report(results_file)
    else:
        print("No results file generated. Tests may have failed to run.")
        return 1

    return proc.returncode


def generate_summary_report(results_file: Path):
    """
    Generate summary report from results.

    Args:
        results_file: Path to JSONL results file
    """
    print("Generating summary report...")
    print()

    # Load all results
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        print("No results found.")
        return

    # Group by level
    by_level = defaultdict(list)
    for result in results:
        by_level[result["level"]].append(result)

    # Calculate statistics
    total_tasks = len(results)
    total_success = sum(1 for r in results if r["validation_result"]["success"])
    total_duration = sum(r["run_result"]["duration"] for r in results)

    # Print summary
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total tasks:       {total_tasks}")
    print(f"Successful:        {total_success} ({total_success/total_tasks*100:.1f}%)")
    print(f"Failed:            {total_tasks - total_success}")
    print(f"Total duration:    {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print(f"Average duration:  {total_duration/total_tasks:.1f}s per task")
    print()

    # Print by level
    print("=" * 80)
    print("RESULTS BY LEVEL")
    print("=" * 80)
    print()

    for level in ["L5", "L6", "L7", "L8"]:
        if level not in by_level:
            continue

        level_results = by_level[level]
        level_success = sum(1 for r in level_results if r["validation_result"]["success"])
        level_duration = sum(r["run_result"]["duration"] for r in level_results)

        print(f"{level}: {level_success}/{len(level_results)} passed ({level_success/len(level_results)*100:.1f}%)")
        print(f"  Average duration: {level_duration/len(level_results):.1f}s")
        print()

        # Show individual tasks
        for result in sorted(level_results, key=lambda r: r["task_id"]):
            success = result["validation_result"]["success"]
            duration = result["run_result"]["duration"]
            status = "✓" if success else "✗"

            print(f"  {status} {result['task_id']}: {result['name']}")
            print(f"      Duration: {duration:.1f}s")

            # Show details
            validation = result["validation_result"]
            if validation["files_created"]:
                print(f"      Files: {len(validation['files_created'])}")
            if "error" in result["run_result"]:
                print(f"      Error: {result['run_result']['error']}")

            # Show validation details
            if not success and validation.get("details"):
                details = validation["details"]
                if "symbols" in details:
                    missing_classes = details["symbols"]["missing"].get("classes", [])
                    missing_funcs = details["symbols"]["missing"].get("functions", [])
                    if missing_classes:
                        print(f"      Missing classes: {', '.join(missing_classes)}")
                    if missing_funcs:
                        print(f"      Missing functions: {', '.join(missing_funcs)}")

            print()

    # Generate markdown summary
    markdown_file = Path("evaluation_results/PROJECT_EVAL_SUMMARY.md")
    generate_markdown_summary(results, by_level, markdown_file)
    print(f"Detailed summary saved to: {markdown_file}")
    print()


def generate_markdown_summary(results: list, by_level: dict, output_file: Path):
    """
    Generate markdown summary report.

    Args:
        results: List of all results
        by_level: Results grouped by level
        output_file: Path to output markdown file
    """
    total_tasks = len(results)
    total_success = sum(1 for r in results if r["validation_result"]["success"])
    total_duration = sum(r["run_result"]["duration"] for r in results)

    lines = [
        "# Project Evaluation Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- **Total tasks:** {total_tasks}",
        f"- **Successful:** {total_success} ({total_success/total_tasks*100:.1f}%)",
        f"- **Failed:** {total_tasks - total_success}",
        f"- **Total duration:** {total_duration:.1f}s ({total_duration/60:.1f}m)",
        f"- **Average duration:** {total_duration/total_tasks:.1f}s per task",
        "",
    ]

    # Results by level
    lines.append("## Results by Level")
    lines.append("")

    for level in ["L5", "L6", "L7", "L8"]:
        if level not in by_level:
            continue

        level_results = by_level[level]
        level_success = sum(1 for r in level_results if r["validation_result"]["success"])
        level_duration = sum(r["run_result"]["duration"] for r in level_results)

        lines.append(f"### {level}")
        lines.append("")
        lines.append(f"- **Pass rate:** {level_success}/{len(level_results)} ({level_success/len(level_results)*100:.1f}%)")
        lines.append(f"- **Average duration:** {level_duration/len(level_results):.1f}s")
        lines.append("")

        # Table of tasks
        lines.append("| Task | Status | Duration | Files | Details |")
        lines.append("|------|--------|----------|-------|---------|")

        for result in sorted(level_results, key=lambda r: r["task_id"]):
            success = result["validation_result"]["success"]
            duration = result["run_result"]["duration"]
            status = "✓" if success else "✗"
            files = len(result["validation_result"]["files_created"])

            details = ""
            if not success:
                if "error" in result["run_result"]:
                    details = f"Error: {result['run_result']['error'][:30]}"
                elif result["validation_result"].get("details", {}).get("symbols"):
                    symbols = result["validation_result"]["details"]["symbols"]
                    missing = []
                    if symbols["missing"].get("classes"):
                        missing.append(f"{len(symbols['missing']['classes'])} classes")
                    if symbols["missing"].get("functions"):
                        missing.append(f"{len(symbols['missing']['functions'])} funcs")
                    details = f"Missing: {', '.join(missing)}"

            lines.append(f"| {result['name']} | {status} | {duration:.1f}s | {files} | {details} |")

        lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")

    for level in ["L5", "L6", "L7", "L8"]:
        if level not in by_level:
            continue

        lines.append(f"### {level} Tasks")
        lines.append("")

        for result in sorted(by_level[level], key=lambda r: r["task_id"]):
            success = result["validation_result"]["success"]
            status = "✓ PASS" if success else "✗ FAIL"

            lines.append(f"#### {result['task_id']}: {result['name']} - {status}")
            lines.append("")
            lines.append(f"**Description:** {result['description']}")
            lines.append("")
            lines.append(f"**Duration:** {result['run_result']['duration']:.1f}s")
            lines.append("")

            # Files created
            files = result["validation_result"]["files_created"]
            if files:
                lines.append(f"**Files created ({len(files)}):**")
                lines.append("")
                for f in files[:10]:  # Show first 10
                    lines.append(f"- `{f}`")
                if len(files) > 10:
                    lines.append(f"- ... and {len(files) - 10} more")
                lines.append("")

            # Validation details
            validation = result["validation_result"]
            lines.append(f"**Tests passed:** {'Yes' if validation['tests_passed'] else 'No'}")
            lines.append(f"**Code quality:** {'Pass' if validation['code_quality'] else 'Fail'}")
            lines.append("")

            # Symbol validation
            if "symbols" in validation.get("details", {}):
                symbols = validation["details"]["symbols"]
                if symbols["found"].get("classes"):
                    lines.append(f"**Found classes:** {', '.join(symbols['found']['classes'])}")
                if symbols["missing"].get("classes"):
                    lines.append(f"**Missing classes:** {', '.join(symbols['missing']['classes'])}")
                if symbols["found"].get("functions"):
                    lines.append(f"**Found functions:** {', '.join(symbols['found']['functions'])}")
                if symbols["missing"].get("functions"):
                    lines.append(f"**Missing functions:** {', '.join(symbols['missing']['functions'])}")
                lines.append("")

            # Errors
            if "error" in result["run_result"]:
                lines.append(f"**Error:** `{result['run_result']['error']}`")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Write to file
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run project evaluation suite")
    parser.add_argument(
        "--level",
        choices=["L5", "L6", "L7", "L8"],
        help="Filter by level (L5, L6, L7, or L8)"
    )
    parser.add_argument(
        "--task",
        help="Filter by task ID (e.g., L5_json_csv_converter)"
    )

    args = parser.parse_args()

    return run_evaluation_suite(level_filter=args.level, task_filter=args.task)


if __name__ == "__main__":
    sys.exit(main())
