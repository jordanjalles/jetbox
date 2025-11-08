"""
L4-L7 Task Evaluation with Context Inspection

Runs L4-L7 tasks with 2 runs each, saving context inspection reports separately
for failure analysis.

Features:
- Each run gets unique context inspection snapshot directory
- Reports saved to evaluation_results/context_analysis/
- Check-in points after first 2 problems
- Detailed logging for debugging failures
"""
import sys
from pathlib import Path
import tempfile
import shutil
import subprocess
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.evaluation_suite_extended import get_extended_tasks
from tests.evaluation_suite import TaskResult, FailureCategory


class ContextInspectionEvaluator:
    """Evaluator that saves context inspection reports for each run."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"evaluation_results/context_analysis_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Separate directories for successful vs failed runs
        self.success_dir = self.output_dir / "successful_runs"
        self.failure_dir = self.output_dir / "failed_runs"
        self.success_dir.mkdir(exist_ok=True)
        self.failure_dir.mkdir(exist_ok=True)

        self.results = []

    def run_task_with_inspection(
        self,
        task,
        run_number: int
    ) -> tuple[bool, str, Path | None]:
        """
        Run a single task with context inspection enabled.

        Returns:
            (success, output, inspection_dir)
        """
        # Create unique workspace for this run
        workspace = tempfile.mkdtemp(prefix=f"eval_L{task.level}_{task.name}_run{run_number}_")
        workspace_path = Path(workspace)

        print(f"\n{'='*70}")
        print(f"Running: L{task.level} - {task.name} (Run {run_number})")
        print(f"Workspace: {workspace}")
        print(f"{'='*70}\n")

        try:
            # Run agent with context inspection
            # Use subprocess to capture all output
            import sys
            cmd = [
                sys.executable, "/workspace/agent.py",
                "--team", "eval_with_inspection",
                "--workspace", workspace,
                task.goal
            ]

            # Run from /workspace directory
            # Capture stdout and stderr
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd="/workspace",
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            duration = time.time() - start_time

            # Combine output
            output = result.stdout + "\n" + result.stderr

            # Check if task succeeded
            success = result.returncode == 0

            # Try flexible validation for L5/L6/L7 tasks with validators
            from tests.flexible_validation import VALIDATORS

            files_exist = False
            validation_passed = False
            validation_output = ""

            if task.name in VALIDATORS:
                # Use flexible validation (tests functionality not file structure)
                # Currently supports: 5 L5 tasks, 5 L6 tasks, 4 L7 tasks (14 total)
                try:
                    flex_success, flex_message = VALIDATORS[task.name](workspace_path)
                    files_exist = flex_success  # Any files created
                    validation_passed = flex_success
                    validation_output = f"Flexible validation: {flex_message}"
                except Exception as e:
                    validation_output = f"Flexible validation error: {e}"
                    files_exist = False
                    validation_passed = False
            else:
                # Use rigid validation for tasks without flexible validator
                files_exist = all(
                    (workspace_path / f).exists()
                    for f in task.expected_files
                )

                # Run validation commands if files exist
                if files_exist:
                    try:
                        for val_cmd in task.validation_commands:
                            val_result = subprocess.run(
                                val_cmd,
                                cwd=workspace,
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            validation_output += val_result.stdout + "\n" + val_result.stderr
                            if val_result.returncode != 0:
                                validation_passed = False
                                break
                        else:
                            validation_passed = True
                    except Exception as e:
                        validation_output = f"Validation error: {e}"
                        validation_passed = False

            # Overall success = agent succeeded, files exist, validation passed
            overall_success = success and files_exist and validation_passed

            # Copy context inspection directory if it exists
            # Context inspector saves to /workspace/.context_inspection (where agent.py runs)
            context_inspection_dir = Path("/workspace/.context_inspection")
            saved_inspection_dir = None

            if context_inspection_dir.exists():
                # Save to success or failure directory
                target_dir = self.success_dir if overall_success else self.failure_dir

                inspection_target = target_dir / f"L{task.level}_{task.name}_run{run_number}_inspection"
                shutil.copytree(context_inspection_dir, inspection_target)
                saved_inspection_dir = inspection_target

                print(f"✓ Context inspection saved to: {inspection_target}")

                # Clean up for next run to avoid overwriting
                shutil.rmtree(context_inspection_dir)
                print(f"✓ Cleaned up {context_inspection_dir} for next run")

                # Also save agent logs from the nested workspace
                # Agent creates workspace in /workspace/.agent_workspaces/
                nested_workspace = Path("/workspace/.agent_workspaces")

                # Find the most recent workspace matching this task
                import glob
                workspace_pattern = f"workspace-{workspace.replace('/', '-')}-*"
                matching_workspaces = sorted(
                    glob.glob(str(nested_workspace / workspace_pattern)),
                    key=lambda p: Path(p).stat().st_mtime,
                    reverse=True
                )

                if matching_workspaces:
                    agent_workspace = Path(matching_workspaces[0])

                    log_files = [
                        ".agent_context/state.json",
                        ".agent_context/stats.json",
                    ]

                    for log_file in log_files:
                        src = agent_workspace / log_file
                        if src.exists():
                            dst = target_dir / f"L{task.level}_{task.name}_run{run_number}_{Path(log_file).name}"
                            shutil.copy2(src, dst)

            # Save summary
            summary = {
                "task": f"L{task.level} - {task.name}",
                "run": run_number,
                "success": overall_success,
                "duration": duration,
                "files_exist": files_exist,
                "validation_passed": validation_passed,
                "inspection_dir": str(saved_inspection_dir) if saved_inspection_dir else None,
                "workspace": workspace
            }

            self.results.append(summary)

            # Print status
            status = "✓ SUCCESS" if overall_success else "✗ FAILED"
            print(f"\n{status} - L{task.level} {task.name} Run {run_number}")
            print(f"Duration: {duration:.1f}s")
            print(f"Files exist: {files_exist}, Validation: {validation_passed}")

            return overall_success, output, saved_inspection_dir

        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT - L{task.level} {task.name} Run {run_number}")
            return False, "TIMEOUT", None
        except Exception as e:
            print(f"✗ ERROR - L{task.level} {task.name} Run {run_number}: {e}")
            return False, str(e), None
        finally:
            # Keep workspace for debugging failures
            pass

    def run_evaluation(self, levels=[4, 5, 6, 7], runs_per_task=2, check_in_after=2):
        """Run evaluation for specified levels."""
        all_tasks = get_extended_tasks()

        # Filter to specified levels
        tasks_to_run = [t for t in all_tasks if t.level in levels]

        print(f"\n{'='*70}")
        print(f"L4-L7 Context Inspection Evaluation")
        print(f"{'='*70}")
        print(f"Tasks per level: {len([t for t in tasks_to_run if t.level == 4])} (L4), "
              f"{len([t for t in tasks_to_run if t.level == 5])} (L5), "
              f"{len([t for t in tasks_to_run if t.level == 6])} (L6), "
              f"{len([t for t in tasks_to_run if t.level == 7])} (L7)")
        print(f"Runs per task: {runs_per_task}")
        print(f"Total runs: {len(tasks_to_run) * runs_per_task}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")

        problem_count = 0

        for task in tasks_to_run:
            for run_num in range(1, runs_per_task + 1):
                problem_count += 1

                # Run task with inspection
                success, output, inspection_dir = self.run_task_with_inspection(
                    task, run_num
                )

                # Check-in point
                if problem_count == check_in_after:
                    print(f"\n{'='*70}")
                    print(f"CHECK-IN: First {check_in_after} problems complete")
                    print(f"{'='*70}")
                    self.print_summary()
                    print(f"Continuing to next problem...")
                elif problem_count == check_in_after * 2:
                    print(f"\n{'='*70}")
                    print(f"CHECK-IN: First {check_in_after * 2} problems complete")
                    print(f"{'='*70}")
                    self.print_summary()
                    print(f"Continuing to next problem...")

        # Final summary
        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*70}")
        self.print_summary()
        self.save_final_report()

    def print_summary(self):
        """Print current evaluation summary."""
        if not self.results:
            print("No results yet")
            return

        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful

        print(f"\nResults so far:")
        print(f"  Total runs: {total}")
        print(f"  Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"  Failed: {failed} ({failed/total*100:.1f}%)")

        # Show which tasks failed
        if failed > 0:
            print(f"\nFailed runs:")
            for r in self.results:
                if not r["success"]:
                    print(f"  - {r['task']} Run {r['run']}")
                    if r["inspection_dir"]:
                        print(f"    Inspection: {r['inspection_dir']}")

        print(f"\nContext inspection directories:")
        print(f"  Success: {self.success_dir}")
        print(f"  Failure: {self.failure_dir}")

    def save_final_report(self):
        """Save detailed evaluation report."""
        report_path = self.output_dir / "evaluation_report.md"

        with open(report_path, 'w') as f:
            f.write("# L4-L7 Context Inspection Evaluation Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            total = len(self.results)
            successful = sum(1 for r in self.results if r["success"])
            failed = total - successful

            f.write("## Summary\n\n")
            f.write(f"- **Total runs**: {total}\n")
            f.write(f"- **Successful**: {successful} ({successful/total*100:.1f}%)\n")
            f.write(f"- **Failed**: {failed} ({failed/total*100:.1f}%)\n\n")

            f.write("## Detailed Results\n\n")
            f.write("| Task | Run | Success | Duration | Files | Validation | Inspection |\n")
            f.write("|------|-----|---------|----------|-------|------------|------------|\n")

            for r in self.results:
                success_icon = "✓" if r["success"] else "✗"
                files_icon = "✓" if r["files_exist"] else "✗"
                val_icon = "✓" if r["validation_passed"] else "✗"
                inspection_link = f"[View]({r['inspection_dir']})" if r['inspection_dir'] else "N/A"

                f.write(f"| {r['task']} | {r['run']} | {success_icon} | "
                       f"{r['duration']:.1f}s | {files_icon} | {val_icon} | {inspection_link} |\n")

            f.write(f"\n## Context Inspection Directories\n\n")
            f.write(f"- **Successful runs**: `{self.success_dir}`\n")
            f.write(f"- **Failed runs**: `{self.failure_dir}`\n\n")

            f.write("## Next Steps\n\n")
            f.write("1. Analyze failed run context inspections\n")
            f.write("2. Look for patterns in context growth\n")
            f.write("3. Check for repeated errors in agent behavior\n")
            f.write("4. Generate context inspection reports with:\n")
            f.write("   ```bash\n")
            f.write("   python tools/analyze_context.py <inspection_dir>\n")
            f.write("   ```\n")

        print(f"\n✓ Report saved to: {report_path}")


if __name__ == "__main__":
    evaluator = ContextInspectionEvaluator()
    evaluator.run_evaluation(
        levels=[4, 5, 6, 7],
        runs_per_task=2,
        check_in_after=2
    )
