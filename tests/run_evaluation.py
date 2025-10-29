"""
Run Evaluation Suite

Actually executes the agent on evaluation tasks and generates results.
Starts with Level 1 tasks to establish baseline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tests.evaluation_suite import EvaluationSuite

if __name__ == "__main__":
    print("🚀 Starting Jetbox Agent Evaluation")
    print("=" * 70)
    print("This will run the agent on evaluation tasks and generate reports.")
    print("Starting with Level 1 (Basic) tasks...")
    print("=" * 70)

    suite = EvaluationSuite(output_dir=Path("evaluation_results"))

    # Run Level 1 tasks only (basic tasks, should be easiest)
    suite.run_suite(max_level=1)

    print("\n✅ Evaluation complete! Check evaluation_results/ for reports.")
