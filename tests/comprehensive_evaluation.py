"""
Comprehensive 4-Hour Evaluation Run

Executes all 14 tasks across 4 difficulty levels.
Saves detailed logs, outputs, and results for later analysis.

Estimated runtime: 3-4 hours (depends on agent performance)
"""
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import traceback

sys.path.insert(0, str(Path(__file__).parent))

from tests.evaluation_suite import EvaluationSuite, TaskDefinition, TaskResult, FailureCategory
from task_executor_agent import TaskExecutorAgent


class ComprehensiveEvaluator:
    """Extended evaluator that actually runs the agent."""

    def __init__(self, output_dir: Path = None, model: str = None, tasks: list[TaskDefinition] = None):
        if tasks is None:
            # Use default evaluation suite
            self.suite = EvaluationSuite(model=model, output_dir=output_dir or Path("evaluation_results"))
            self.tasks = self.suite.tasks
            self.output_dir = self.suite.output_dir
        else:
            # Use custom task list
            self.tasks = tasks
            self.output_dir = output_dir or Path("evaluation_results")
            self.suite = None

        self.model = model
        self.output_dir.mkdir(exist_ok=True)

        # Detailed logging
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = self.output_dir / f"run_{self.run_id}"
        self.log_dir.mkdir(exist_ok=True)

        self.main_log = self.log_dir / "evaluation.log"
        self.results_json = self.log_dir / "results.json"

    def log(self, message: str):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        # Ensure log directory exists
        self.main_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.main_log, "a") as f:
            f.write(log_msg + "\n")

    def run_task_with_agent(self, task: TaskDefinition, workspace: Path) -> TaskResult:
        """Actually run the agent on a task."""
        task_log_dir = self.log_dir / f"task_{task.name}"
        task_log_dir.mkdir(exist_ok=True)

        self.log(f"\n{'='*70}")
        self.log(f"STARTING: Level {task.level} - {task.name}")
        self.log(f"Goal: {task.goal}")
        self.log(f"Timeout: {task.timeout_rounds} rounds")
        self.log(f"{'='*70}")

        start_time = time.time()
        error_msg = None
        failure_category = None
        agent_output = ""
        rounds_used = 0

        try:
            # Create task executor
            executor = TaskExecutorAgent(
                workspace=workspace,
                goal=task.goal,
                max_rounds=task.timeout_rounds,
                model=self.model
            )

            workspace_dir = executor.workspace_manager.workspace_dir
            self.log(f"Workspace: {workspace_dir}")

            # Save initial state
            with open(task_log_dir / "task_definition.json", "w") as f:
                json.dump({
                    "name": task.name,
                    "level": task.level,
                    "description": task.description,
                    "goal": task.goal,
                    "expected_files": task.expected_files,
                    "tags": task.tags,
                    "timeout_rounds": task.timeout_rounds
                }, f, indent=2)

            # RUN THE AGENT
            self.log("🤖 Starting agent execution...")

            # Actually run the agent!
            try:
                agent_result = executor.run()
                # Get rounds used from status display if available
                if hasattr(executor, 'status_display') and executor.status_display:
                    rounds_used = len(executor.status_display.stats.llm_call_times)
                else:
                    rounds_used = task.timeout_rounds
                self.log(f"✅ Agent execution completed")
                self.log(f"   Result: {agent_result.get('status', 'unknown')}")
            except Exception as e:
                self.log(f"⚠️  Agent execution error: {e}")
                # Try to get rounds from status display even after error
                if hasattr(executor, 'status_display') and executor.status_display:
                    rounds_used = len(executor.status_display.stats.llm_call_times)
                else:
                    rounds_used = task.timeout_rounds
                # Continue to check what files were created despite error

            # Check what files exist
            files_created = []
            all_files_exist = True

            for expected_file in task.expected_files:
                file_path = workspace_dir / expected_file
                if file_path.exists():
                    files_created.append(expected_file)
                    # Copy file to task log
                    (task_log_dir / expected_file.replace("/", "_")).write_text(
                        file_path.read_text()
                    )

            all_files_exist = len(files_created) == len(task.expected_files)

            # Log files created
            self.log(f"📁 Files created: {files_created if files_created else 'None'}")
            if not all_files_exist:
                missing = set(task.expected_files) - set(files_created)
                self.log(f"❌ Missing files: {missing}")

            # Run validation commands
            validation_passed = False
            validation_output = ""

            if all_files_exist:
                import os
                import subprocess

                original_cwd = os.getcwd()
                try:
                    os.chdir(workspace_dir)
                    self.log("🧪 Running validation commands...")

                    for i, cmd in enumerate(task.validation_commands):
                        self.log(f"   Command {i+1}: {' '.join(cmd)}")
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        validation_output += f"=== Command {i+1} ===\n"
                        validation_output += f"Command: {' '.join(cmd)}\n"
                        validation_output += f"Exit code: {result.returncode}\n"
                        validation_output += f"Stdout:\n{result.stdout}\n"
                        if result.stderr:
                            validation_output += f"Stderr:\n{result.stderr}\n"
                        validation_output += "\n"

                        if result.returncode == 0:
                            validation_passed = True
                            self.log(f"   ✅ Command {i+1} passed")
                        else:
                            self.log(f"   ❌ Command {i+1} failed")

                finally:
                    os.chdir(original_cwd)
            else:
                failure_category = FailureCategory.MISSING_FILES
                validation_output = f"Missing files: {set(task.expected_files) - set(files_created)}"
                self.log(f"⚠️  Skipping validation - missing files")

            # Save validation output
            (task_log_dir / "validation_output.txt").write_text(validation_output)

            # Categorize failures
            success = validation_passed and all_files_exist

            if not success and not failure_category:
                if "SyntaxError" in validation_output or "IndentationError" in validation_output:
                    failure_category = FailureCategory.SYNTAX_ERROR
                elif "ImportError" in validation_output or "ModuleNotFoundError" in validation_output:
                    failure_category = FailureCategory.IMPORT_ERROR
                elif "AssertionError" in validation_output:
                    failure_category = FailureCategory.TEST_FAILURE
                elif rounds_used >= task.timeout_rounds:
                    failure_category = FailureCategory.TIMEOUT
                else:
                    failure_category = FailureCategory.INCOMPLETE

            duration = time.time() - start_time

            # Log result
            status = "✅ PASS" if success else "❌ FAIL"
            self.log(f"\n{status} - {task.name}")
            self.log(f"   Duration: {duration:.1f}s")
            self.log(f"   Rounds used: {rounds_used}/{task.timeout_rounds}")
            if not success:
                self.log(f"   Failure: {failure_category}")

            result = TaskResult(
                task=task,
                success=success,
                duration=duration,
                rounds_used=rounds_used,
                files_created=files_created,
                validation_passed=validation_passed,
                validation_output=validation_output,
                failure_category=failure_category,
                error_message=error_msg,
                agent_output=agent_output
            )

            # Save task result
            with open(task_log_dir / "result.json", "w") as f:
                json.dump({
                    "task": task.name,
                    "level": task.level,
                    "success": result.success,
                    "duration": result.duration,
                    "rounds_used": result.rounds_used,
                    "files_created": result.files_created,
                    "validation_passed": result.validation_passed,
                    "failure_category": result.failure_category,
                    "error_message": result.error_message
                }, f, indent=2)

            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            error_trace = traceback.format_exc()
            failure_category = FailureCategory.UNKNOWN

            self.log(f"❌ ERROR in {task.name}: {error_msg}")
            self.log(f"   Traceback:\n{error_trace}")

            # Save error
            (task_log_dir / "error.txt").write_text(error_trace)

            return TaskResult(
                task=task,
                success=False,
                duration=duration,
                rounds_used=rounds_used,
                files_created=[],
                validation_passed=False,
                validation_output="",
                failure_category=failure_category,
                error_message=error_msg,
                agent_output=agent_output
            )

    def run_comprehensive_evaluation(self):
        """Run all tasks and generate comprehensive results."""
        self.log("\n" + "="*70)
        self.log("COMPREHENSIVE JETBOX EVALUATION")
        self.log(f"Run ID: {self.run_id}")
        self.log(f"Output directory: {self.log_dir}")
        self.log(f"Model: {self.suite.model or 'default'}")
        self.log("="*70)

        start_time = time.time()
        results = []

        # Run all tasks
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            for i, task in enumerate(self.tasks):
                self.log(f"\n{'#'*70}")
                self.log(f"Progress: Task {i+1}/{len(self.tasks)}")
                self.log(f"{'#'*70}")

                result = self.run_task_with_agent(task, workspace)
                results.append(result)

                # Save incremental progress
                self._save_progress(results, time.time() - start_time)

        # Generate final report
        total_duration = time.time() - start_time
        self._generate_final_report(results, total_duration)

        self.log("\n" + "="*70)
        self.log(f"✅ EVALUATION COMPLETE!")
        self.log(f"Total duration: {total_duration/60:.1f} minutes")
        self.log(f"Results saved to: {self.log_dir}")
        self.log("="*70)

    def _save_progress(self, results: list[TaskResult], elapsed_time: float):
        """Save incremental progress."""
        progress = {
            "run_id": self.run_id,
            "elapsed_time": elapsed_time,
            "elapsed_minutes": elapsed_time / 60,
            "tasks_completed": len(results),
            "tasks_total": len(self.tasks),
            "results": [
                {
                    "task": r.task.name,
                    "level": r.task.level,
                    "success": r.success,
                    "duration": r.duration,
                    "rounds_used": r.rounds_used,
                    "failure_category": r.failure_category
                }
                for r in results
            ]
        }

        with open(self.results_json, "w") as f:
            json.dump(progress, f, indent=2)

    def _generate_final_report(self, results: list[TaskResult], total_duration: float):
        """Generate comprehensive final report."""
        report_file = self.log_dir / "EVALUATION_REPORT.md"

        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed

        with open(report_file, "w") as f:
            f.write(f"# Jetbox Agent Comprehensive Evaluation Report\n\n")
            f.write(f"**Run ID**: {self.run_id}\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Duration**: {total_duration/60:.1f} minutes\n\n")
            f.write(f"**Model**: {self.model or 'default'}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Tasks**: {total}\n")
            f.write(f"- **Passed**: {passed} ({100*passed/total:.1f}%)\n")
            f.write(f"- **Failed**: {failed} ({100*failed/total:.1f}%)\n")
            f.write(f"- **Average Duration**: {sum(r.duration for r in results)/total:.1f}s per task\n\n")

            # Results by Level
            f.write("## Results by Difficulty Level\n\n")
            for level in range(1, 5):
                level_results = [r for r in results if r.task.level == level]
                if level_results:
                    level_passed = sum(1 for r in level_results if r.success)
                    level_total = len(level_results)
                    pct = 100 * level_passed / level_total
                    f.write(f"### Level {level}\n")
                    f.write(f"- **Score**: {level_passed}/{level_total} ({pct:.1f}%)\n")
                    f.write(f"- **Avg Duration**: {sum(r.duration for r in level_results)/level_total:.1f}s\n")
                    f.write(f"- **Avg Rounds**: {sum(r.rounds_used for r in level_results)/level_total:.1f}\n\n")

            # Failure Analysis
            f.write("## Failure Analysis\n\n")
            failure_counts = {}
            for result in results:
                if not result.success and result.failure_category:
                    failure_counts[result.failure_category] = failure_counts.get(result.failure_category, 0) + 1

            if failure_counts:
                f.write("### Failure Categories\n\n")
                for category, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = 100 * count / failed if failed > 0 else 0
                    f.write(f"- **{category}**: {count} failures ({pct:.1f}%)\n")
                f.write("\n")

            # Detailed Results
            f.write("## Detailed Task Results\n\n")
            for result in results:
                status = "✅ PASS" if result.success else "❌ FAIL"
                f.write(f"### {status} Level {result.task.level}: {result.task.name}\n\n")
                f.write(f"**Goal**: {result.task.goal}\n\n")
                f.write(f"**Duration**: {result.duration:.2f}s | **Rounds**: {result.rounds_used}/{result.task.timeout_rounds}\n\n")

                if result.files_created:
                    f.write(f"**Files Created**: {', '.join(result.files_created)}\n\n")

                if not result.success:
                    f.write(f"**Failure Category**: {result.failure_category}\n\n")
                    if result.error_message:
                        f.write(f"**Error**: {result.error_message}\n\n")

                f.write("---\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            if passed / total >= 0.8:
                f.write("- ✅ Agent performs well on basic tasks\n")
            elif passed / total >= 0.5:
                f.write("- ⚠️  Agent struggles with intermediate complexity\n")
            else:
                f.write("- ❌ Agent needs fundamental improvements\n")

            if failure_counts:
                top_failure = max(failure_counts.items(), key=lambda x: x[1])
                f.write(f"- 🎯 Focus on reducing **{top_failure[0]}** failures\n")

            f.write("\n")

        self.log(f"📊 Report generated: {report_file}")

    def save_results(self, results: list[TaskResult], filename: str = "results.json"):
        """Save results to JSON file."""
        results_file = self.log_dir / filename
        results_data = []
        for r in results:
            results_data.append({
                "task": r.task.name,
                "level": r.task.level,
                "success": r.success,
                "duration": r.duration,
                "rounds_used": r.rounds_used,
                "files_created": r.files_created,
                "validation_passed": r.validation_passed,
                "failure_category": r.failure_category,
                "error_message": r.error_message,
            })

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        self.log(f"💾 Results saved: {results_file}")

    def generate_report(self, results: list[TaskResult]):
        """Generate comprehensive report from results."""
        total_duration = sum(r.duration for r in results)
        self._generate_final_report(results, total_duration)

    def run_suite(self) -> list[TaskResult]:
        """Run all tasks once and return results (for use by external runners)."""
        results = []
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            for i, task in enumerate(self.tasks):
                result = self.run_task_with_agent(task, workspace)
                results.append(result)

        return results


# Alias for backwards compatibility
ComprehensiveEvaluation = ComprehensiveEvaluator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive 4-hour evaluation")
    parser.add_argument("--no-prompt", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--model", type=str, help="Ollama model to use")
    parser.add_argument("--max-level", type=int, default=4, help="Maximum difficulty level to run (1-4)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 COMPREHENSIVE 4-HOUR EVALUATION")
    print("="*70)
    print("\nThis will run ALL 14 tasks across 4 difficulty levels.")
    print("Estimated duration: 3-4 hours")
    print("\nResults will be saved to: evaluation_results/run_YYYYMMDD_HHMMSS/")
    print("\nYou can safely interrupt and resume - progress is saved incrementally.")
    print("="*70)

    if not args.no_prompt:
        input("\nPress Enter to start, or Ctrl+C to cancel...")

    evaluator = ComprehensiveEvaluator(model=args.model)

    # Filter tasks by level if needed
    if args.max_level < 4:
        original_tasks = evaluator.suite.tasks
        evaluator.suite.tasks = [t for t in original_tasks if t.level <= args.max_level]
        print(f"\n📊 Running {len(evaluator.suite.tasks)} tasks (levels 1-{args.max_level})")

    evaluator.run_comprehensive_evaluation()
