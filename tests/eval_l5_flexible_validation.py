"""
L5 Re-evaluation with Flexible Validation

Re-run L5 tasks to prove flexible validation improves success rate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.eval_l4_l7_context_inspection import L4L7ContextInspectionEval
from tests.evaluation_suite_extended import get_extended_tasks


def main():
    """Run L5 tasks only with flexible validation."""

    # Get only L5 tasks
    all_tasks = get_extended_tasks()
    l5_tasks = [t for t in all_tasks if t.level == 5]

    print("="*70)
    print("L5 RE-EVALUATION WITH FLEXIBLE VALIDATION")
    print("="*70)
    print(f"Tasks: {len(l5_tasks)}")
    print(f"Runs per task: 2")
    print(f"Total runs: {len(l5_tasks) * 2}")
    print("="*70)
    print()

    # Create evaluator
    eval_runner = L4L7ContextInspectionEval(
        output_dir="evaluation_results/l5_flexible_validation",
        tasks=l5_tasks
    )

    # Run evaluation
    results = eval_runner.run_evaluation(runs_per_task=2)

    # Print summary
    print("\n" + "="*70)
    print("L5 FLEXIBLE VALIDATION RESULTS")
    print("="*70)
    print(f"Total runs: {results['total_runs']}")
    print(f"Successful: {results['successful']} ({100*results['successful']//results['total_runs']}%)")
    print(f"Failed: {results['failed']} ({100*results['failed']//results['total_runs']}%)")
    print()

    # Compare to rigid validation
    print("COMPARISON:")
    print(f"  Rigid validation (previous): 0/10 (0%)")
    print(f"  Flexible validation (now): {results['successful']}/{results['total_runs']} ({100*results['successful']//results['total_runs']}%)")
    print(f"  Improvement: +{results['successful']} tasks")
    print("="*70)


if __name__ == "__main__":
    main()
