#!/usr/bin/env python3
"""
Evaluation: Workspace Reuse and Workspace Task Notes (WTN) Changelog System

Tests:
1. Workspace reuse across multiple rounds
2. Changelog-style WTN generation
3. File change tracking (created/edited/deleted)
4. WTN context injection with delimiters
5. Documentation pointer detection
6. Matter-of-fact headers (not celebratory)

Flow:
- 5 different projects
- 3 rounds per project (initial, enhance, refactor)
- Verifies WTN between rounds
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import shutil

# Test projects with 3 rounds each
PROJECTS = [
    {
        "name": "calculator",
        "rounds": [
            {
                "goal": "Create a Python calculator package with add and subtract functions",
                "expected_files": ["calculator.py", "tests/test_calculator.py"],
                "expected_changes": {"created": True},
            },
            {
                "goal": "Add multiply and divide functions to the calculator package",
                "expected_files": ["calculator.py"],  # Should be edited
                "expected_changes": {"edited": True},
            },
            {
                "goal": "Add comprehensive docstrings and a README to the calculator",
                "expected_files": ["README.md"],
                "expected_changes": {"created": True, "edited": True},
            },
        ]
    },
    {
        "name": "web_scraper",
        "rounds": [
            {
                "goal": "Create a simple web scraper that extracts titles from a webpage",
                "expected_files": ["scraper.py", "requirements.txt"],
                "expected_changes": {"created": True},
            },
            {
                "goal": "Add error handling and retry logic to the web scraper",
                "expected_files": ["scraper.py"],
                "expected_changes": {"edited": True},
            },
            {
                "goal": "Add unit tests for the web scraper with mocked responses",
                "expected_files": ["tests/test_scraper.py"],
                "expected_changes": {"created": True},
            },
        ]
    },
    {
        "name": "todo_app",
        "rounds": [
            {
                "goal": "Create a simple command-line TODO app that can add and list tasks",
                "expected_files": ["todo.py"],
                "expected_changes": {"created": True},
            },
            {
                "goal": "Add task completion and deletion features to the TODO app",
                "expected_files": ["todo.py"],
                "expected_changes": {"edited": True},
            },
            {
                "goal": "Add persistent storage with JSON file for the TODO app",
                "expected_files": ["todo.py", "storage.py"],
                "expected_changes": {"created": True, "edited": True},
            },
        ]
    },
    {
        "name": "markdown_parser",
        "rounds": [
            {
                "goal": "Create a basic markdown to HTML converter for headings and paragraphs",
                "expected_files": ["parser.py"],
                "expected_changes": {"created": True},
            },
            {
                "goal": "Add support for links and bold/italic text to the markdown parser",
                "expected_files": ["parser.py"],
                "expected_changes": {"edited": True},
            },
            {
                "goal": "Add comprehensive tests and documentation for the parser",
                "expected_files": ["tests/test_parser.py", "README.md"],
                "expected_changes": {"created": True},
            },
        ]
    },
    {
        "name": "config_manager",
        "rounds": [
            {
                "goal": "Create a configuration manager that loads YAML and JSON configs",
                "expected_files": ["config.py", "requirements.txt"],
                "expected_changes": {"created": True},
            },
            {
                "goal": "Add environment variable substitution to the config manager",
                "expected_files": ["config.py"],
                "expected_changes": {"edited": True},
            },
            {
                "goal": "Add validation schema support and error reporting to config manager",
                "expected_files": ["config.py", "validator.py"],
                "expected_changes": {"created": True, "edited": True},
            },
        ]
    },
]


class EvaluationResults:
    """Track evaluation results."""

    def __init__(self):
        self.results = []
        self.start_time = datetime.now()

    def add_result(self, project: str, round_num: int, check: str, passed: bool, details: str = ""):
        """Add a test result."""
        self.results.append({
            "project": project,
            "round": round_num,
            "check": check,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def get_summary(self) -> dict:
        """Get summary statistics."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        by_project = {}
        for r in self.results:
            proj = r["project"]
            if proj not in by_project:
                by_project[proj] = {"total": 0, "passed": 0, "failed": 0}
            by_project[proj]["total"] += 1
            if r["passed"]:
                by_project[proj]["passed"] += 1
            else:
                by_project[proj]["failed"] += 1

        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "by_project": by_project,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }

    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"Duration: {summary['duration_seconds']:.1f}s")
        print()

        print("BY PROJECT:")
        for proj, stats in summary["by_project"].items():
            status = "✓" if stats["failed"] == 0 else "✗"
            print(f"  {status} {proj}: {stats['passed']}/{stats['total']} passed")

        print("=" * 80)

    def save_to_file(self, filepath: Path):
        """Save results to JSON file."""
        summary = self.get_summary()
        output = {
            "summary": summary,
            "results": self.results
        }

        filepath.write_text(json.dumps(output, indent=2))
        print(f"\n✓ Results saved to: {filepath}")


def verify_workspace_reused(workspace_path: Path, project_name: str, round_num: int, results: EvaluationResults):
    """Verify workspace was reused (same directory exists)."""
    check_name = f"workspace_reused_round_{round_num}"

    if workspace_path.exists():
        results.add_result(project_name, round_num, check_name, True,
                          f"Workspace exists at {workspace_path}")
        return True
    else:
        results.add_result(project_name, round_num, check_name, False,
                          f"Workspace not found at {workspace_path}")
        return False


def verify_wtn_exists(workspace_path: Path, project_name: str, round_num: int, results: EvaluationResults):
    """Verify workspace task notes file exists."""
    check_name = f"wtn_file_exists_round_{round_num}"
    wtn_file = workspace_path / ".agent_context" / "workspace_task_notes.md"

    if wtn_file.exists():
        results.add_result(project_name, round_num, check_name, True,
                          f"WTN file exists: {wtn_file}")
        return True
    else:
        results.add_result(project_name, round_num, check_name, False,
                          f"WTN file not found: {wtn_file}")
        return False


def verify_wtn_format(workspace_path: Path, project_name: str, round_num: int, results: EvaluationResults):
    """Verify WTN uses new changelog format."""
    wtn_file = workspace_path / ".agent_context" / "workspace_task_notes.md"
    if not wtn_file.exists():
        return False

    content = wtn_file.read_text()

    # Check for matter-of-fact header (not celebratory)
    has_matter_of_fact = "marked done" in content or "marked failed" in content
    results.add_result(project_name, round_num, "wtn_matter_of_fact_header", has_matter_of_fact,
                      "Found 'marked done/failed'" if has_matter_of_fact else "No matter-of-fact headers")

    # Check for no celebratory markers
    has_no_celebration = "GOAL COMPLETE" not in content and "✓" not in content
    results.add_result(project_name, round_num, "wtn_no_celebration", has_no_celebration,
                      "No celebratory markers" if has_no_celebration else "Found celebratory markers")

    # Check for timestamp
    has_timestamp = "Timestamp:" in content
    results.add_result(project_name, round_num, "wtn_has_timestamp", has_timestamp,
                      "Timestamp present" if has_timestamp else "Timestamp missing")

    # Check for file changes section
    has_file_changes = "File Changes:" in content
    results.add_result(project_name, round_num, "wtn_has_file_changes", has_file_changes,
                      "File changes section present" if has_file_changes else "File changes section missing")

    return has_matter_of_fact and has_no_celebration and has_timestamp and has_file_changes


def verify_file_tracking(workspace_path: Path, project_name: str, round_num: int,
                        expected_changes: dict, results: EvaluationResults):
    """Verify file change tracking worked correctly."""
    wtn_file = workspace_path / ".agent_context" / "workspace_task_notes.md"
    if not wtn_file.exists():
        return False

    content = wtn_file.read_text()

    # Get the last entry (most recent)
    entries = content.split("---")
    if len(entries) < 2:
        return False

    last_entry = entries[-2]  # -1 is empty after last ---

    # Verify expected changes are present
    if expected_changes.get("created"):
        has_created = "Created:" in last_entry
        results.add_result(project_name, round_num, "file_tracking_created", has_created,
                          "Created files tracked" if has_created else "Created files not tracked")

    if expected_changes.get("edited"):
        has_edited = "Edited:" in last_entry
        results.add_result(project_name, round_num, "file_tracking_edited", has_edited,
                          "Edited files tracked" if has_edited else "Edited files not tracked")

    if expected_changes.get("deleted"):
        has_deleted = "Deleted:" in last_entry
        results.add_result(project_name, round_num, "file_tracking_deleted", has_deleted,
                          "Deleted files tracked" if has_deleted else "Deleted files not tracked")

    return True


def verify_documentation_pointers(workspace_path: Path, project_name: str, round_num: int, results: EvaluationResults):
    """Verify documentation files are detected and listed."""
    wtn_file = workspace_path / ".agent_context" / "workspace_task_notes.md"
    if not wtn_file.exists():
        return False

    content = wtn_file.read_text()

    # Check if README or doc files exist in workspace
    has_readme = (workspace_path / "README.md").exists()
    has_docs_dir = (workspace_path / "docs").exists()

    # Check if documentation section exists in WTN
    has_doc_section = "Documentation:" in content

    if has_readme or has_docs_dir:
        # Should have documentation pointers
        results.add_result(project_name, round_num, "documentation_pointers", has_doc_section,
                          "Documentation section present" if has_doc_section else "Documentation section missing")
        return has_doc_section
    else:
        # No docs expected
        results.add_result(project_name, round_num, "documentation_pointers", True,
                          "No documentation files (section not required)")
        return True


def run_orchestrator(goal: str, workspace_path: Path = None, timeout: int = 180) -> dict:
    """
    Run orchestrator with a goal.

    Args:
        goal: Goal description
        workspace_path: Optional workspace path for reuse
        timeout: Timeout in seconds

    Returns:
        Dict with status, workspace, and execution info
    """
    import subprocess

    # Build command
    cmd = [sys.executable, "orchestrator_agent.py"]

    if workspace_path:
        cmd.extend(["--workspace", str(workspace_path)])

    cmd.append(goal)

    print(f"\n[orchestrator] Running: {' '.join(cmd)}")
    print(f"[orchestrator] Goal: {goal}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/workspace"
        )

        # Parse output for workspace path
        output = result.stdout + result.stderr

        # Try to find workspace path in output
        actual_workspace = None
        for line in output.split("\n"):
            if "workspace:" in line.lower() or "workspace_dir" in line.lower():
                # Try to extract path
                parts = line.split(":")
                if len(parts) >= 2:
                    path_str = parts[-1].strip()
                    if Path(path_str).exists():
                        actual_workspace = Path(path_str)
                        break

        # If not found in output, check .agent_workspaces for recent dirs
        if not actual_workspace:
            workspaces_dir = Path("/workspace/.agent_workspaces")
            if workspaces_dir.exists():
                # Find most recently modified workspace
                dirs = [d for d in workspaces_dir.iterdir() if d.is_dir()]
                if dirs:
                    actual_workspace = max(dirs, key=lambda d: d.stat().st_mtime)

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "workspace": actual_workspace,
            "output": output
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "returncode": -1,
            "workspace": None,
            "output": f"Timeout after {timeout}s"
        }
    except Exception as e:
        return {
            "status": "error",
            "returncode": -1,
            "workspace": None,
            "output": f"Error: {e}"
        }


def evaluate_project(project: dict, results: EvaluationResults, dry_run: bool = False):
    """Evaluate a single project across all rounds."""
    project_name = project["name"]
    print("\n" + "=" * 80)
    print(f"EVALUATING PROJECT: {project_name}")
    print("=" * 80)

    workspace_path = None

    for round_num, round_config in enumerate(project["rounds"], start=1):
        print(f"\n--- Round {round_num}/{len(project['rounds'])} ---")
        print(f"Goal: {round_config['goal']}")

        if dry_run:
            print("[DRY RUN] Skipping orchestrator execution")
            continue

        # Run orchestrator
        result = run_orchestrator(
            goal=round_config["goal"],
            workspace_path=workspace_path,
            timeout=180
        )

        print(f"Status: {result['status']}")

        # Update workspace path for next round
        if result["workspace"]:
            workspace_path = result["workspace"]
            print(f"Workspace: {workspace_path}")

        # Verify results
        if round_num > 1:
            # Should reuse workspace
            verify_workspace_reused(workspace_path, project_name, round_num, results)

        if workspace_path:
            verify_wtn_exists(workspace_path, project_name, round_num, results)
            verify_wtn_format(workspace_path, project_name, round_num, results)
            verify_file_tracking(workspace_path, project_name, round_num,
                               round_config["expected_changes"], results)
            verify_documentation_pointers(workspace_path, project_name, round_num, results)

        # Small delay between rounds
        time.sleep(2)


def main():
    """Main evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate workspace reuse and WTN changelog system")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--projects", nargs="+", help="Specific projects to run (default: all)")
    parser.add_argument("--output", type=str, default="evaluation_results/workspace_wtn_eval.json",
                       help="Output file for results")
    parser.add_argument("--cleanup", action="store_true", help="Clean up workspaces before running")

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("WORKSPACE REUSE AND WTN CHANGELOG EVALUATION")
    print("=" * 80)
    print(f"Total projects: {len(PROJECTS)}")
    print(f"Rounds per project: 3")
    print(f"Total orchestrator runs: {len(PROJECTS) * 3}")
    print(f"Output: {args.output}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No orchestrator execution]")

    if args.cleanup:
        print("\n[CLEANUP] Removing old workspaces...")
        workspaces_dir = Path("/workspace/.agent_workspaces")
        if workspaces_dir.exists():
            shutil.rmtree(workspaces_dir)
            workspaces_dir.mkdir(parents=True)
        print("[CLEANUP] Done")

    # Select projects
    projects_to_run = PROJECTS
    if args.projects:
        projects_to_run = [p for p in PROJECTS if p["name"] in args.projects]
        print(f"\nRunning selected projects: {[p['name'] for p in projects_to_run]}")

    # Run evaluation
    results = EvaluationResults()

    for project in projects_to_run:
        try:
            evaluate_project(project, results, dry_run=args.dry_run)
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Stopping evaluation...")
            break
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {project['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Save and print results
    results.print_summary()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.save_to_file(output_path)

    # Exit code based on pass rate
    summary = results.get_summary()
    if summary["pass_rate"] >= 90:
        print("\n✅ EVALUATION PASSED (≥90% pass rate)")
        sys.exit(0)
    else:
        print(f"\n❌ EVALUATION FAILED ({summary['pass_rate']:.1f}% pass rate)")
        sys.exit(1)


if __name__ == "__main__":
    main()
