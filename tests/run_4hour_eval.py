#!/usr/bin/env python3
"""
4-Hour Comprehensive Evaluation Runner

Runs extended evaluation suite with:
- 38 tasks across 7 difficulty levels
- Each task run 5 times
- Total: 190 evaluation runs
- Estimated duration: 4+ hours
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_evaluation import ComprehensiveEvaluation
from tests.evaluation_suite_extended import get_extended_tasks


def main():
    parser = argparse.ArgumentParser(description="Run 4-hour comprehensive evaluation")
    parser.add_argument("--no-prompt", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--max-level", type=int, default=7, help="Maximum difficulty level (1-7)")
    parser.add_argument("--runs-per-task", type=int, default=5, help="Number of times to run each task")
    parser.add_argument("--model", type=str, default=None, help="Ollama model to use")
    args = parser.parse_args()

    # Load extended task suite
    all_tasks = get_extended_tasks()
    tasks_to_run = [t for t in all_tasks if t.level <= args.max_level]

    total_runs = len(tasks_to_run) * args.runs_per_task
    estimated_minutes = total_runs * 1.5  # ~1.5 min per task on average
    estimated_hours = estimated_minutes / 60

    print("=" * 70)
    print("4-HOUR COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Task levels:       L1-L{args.max_level}")
    print(f"Unique tasks:      {len(tasks_to_run)}")
    print(f"Runs per task:     {args.runs_per_task}x")
    print(f"Total evaluations: {total_runs}")
    print(f"Estimated time:    {estimated_hours:.1f} hours")
    print(f"Model:             {args.model or 'default'}")
    print("=" * 70)

    if not args.no_prompt:
        response = input("\nProceed with evaluation? [y/N]: ")
        if response.lower() != 'y':
            print("Evaluation cancelled.")
            return

    # Initialize evaluator
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"evaluation_results/run_4hour_{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ComprehensiveEvaluation(
        tasks=tasks_to_run,
        model=args.model,
        output_dir=output_dir
    )

    print("\n🚀 Starting 4-hour evaluation...")
    print(f"   Output: {output_dir}")
    print(f"   Log: evaluation.log")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run evaluation with repetitions
    all_results = []
    for run_num in range(1, args.runs_per_task + 1):
        print(f"\n{'=' * 70}")
        print(f"RUN {run_num}/{args.runs_per_task}")
        print(f"{'=' * 70}\n")

        evaluator.log(f"\n{'#' * 70}")
        evaluator.log(f"STARTING RUN {run_num}/{args.runs_per_task}")
        evaluator.log(f"{'#' * 70}\n")

        results = evaluator.run_suite()
        all_results.extend(results)

        # Save incremental results
        evaluator.save_results(all_results, f"results_run{run_num}.json")

        # Print run summary
        passed = sum(1 for r in results if r.success)
        print(f"\n✅ Run {run_num} complete: {passed}/{len(results)} tasks passed")

    # Generate final comprehensive report
    evaluator.log("\n" + "=" * 70)
    evaluator.log("4-HOUR EVALUATION COMPLETE!")
    evaluator.log("=" * 70)

    # Analyze results
    total_tasks = len(all_results)
    passed = sum(1 for r in all_results if r.success)
    failed = total_tasks - passed

    # Breakdown by level
    by_level = {}
    for result in all_results:
        level = result.task.level
        if level not in by_level:
            by_level[level] = {"passed": 0, "failed": 0}
        if result.success:
            by_level[level]["passed"] += 1
        else:
            by_level[level]["failed"] += 1

    # Breakdown by failure category
    failure_categories = {}
    for result in all_results:
        if not result.success and result.failure_category:
            cat = result.failure_category
            failure_categories[cat] = failure_categories.get(cat, 0) + 1

    evaluator.log(f"\nOVERALL RESULTS:")
    evaluator.log(f"  Total evaluations: {total_tasks}")
    evaluator.log(f"  Passed: {passed} ({100*passed/total_tasks:.1f}%)")
    evaluator.log(f"  Failed: {failed} ({100*failed/total_tasks:.1f}%)")

    evaluator.log(f"\nRESULTS BY LEVEL:")
    for level in sorted(by_level.keys()):
        stats = by_level[level]
        total = stats["passed"] + stats["failed"]
        pass_rate = 100 * stats["passed"] / total if total > 0 else 0
        evaluator.log(f"  L{level}: {stats['passed']}/{total} passed ({pass_rate:.1f}%)")

    if failure_categories:
        evaluator.log(f"\nFAILURE CATEGORIES:")
        for cat, count in sorted(failure_categories.items(), key=lambda x: x[1], reverse=True):
            evaluator.log(f"  {cat}: {count}")

    evaluator.log(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    evaluator.log(f"Results saved to: {output_dir}")
    evaluator.log("=" * 70)

    # Generate detailed report
    evaluator.generate_report(all_results)

    print(f"\n{'=' * 70}")
    print(f"✅ 4-HOUR EVALUATION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Results: {output_dir}")
    print(f"Report: {output_dir}/EVALUATION_REPORT.md")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
