"""
Quick smoke test to verify evaluation setup before 4-hour run.

Runs ONE simple task to ensure everything works.
Should complete in <30 seconds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_evaluation import ComprehensiveEvaluator

if __name__ == "__main__":
    print("🧪 Evaluation Setup Smoke Test")
    print("="*60)
    print("Running ONE simple task to verify setup...")
    print("Should complete in <30 seconds")
    print("="*60)

    evaluator = ComprehensiveEvaluator()

    # Run only the first (simplest) task
    evaluator.suite.tasks = [evaluator.suite.tasks[0]]

    print(f"\nTest task: {evaluator.suite.tasks[0].name}")
    print(f"Goal: {evaluator.suite.tasks[0].goal}")

    evaluator.run_comprehensive_evaluation()

    print("\n" + "="*60)
    print("✅ Smoke test complete!")
    print("If you see this, the evaluation framework is working.")
    print("\nNext step: Run the full evaluation:")
    print("  python comprehensive_evaluation.py --no-prompt")
    print("="*60)
