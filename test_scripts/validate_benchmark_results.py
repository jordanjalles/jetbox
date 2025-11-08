#!/usr/bin/env python3
"""
Validate that benchmark results are accurate by:
1. Running pytest in each workspace
2. Running ruff linting where applicable
3. Spot-checking actual code quality
"""

import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of validating one workspace."""
    task_id: str
    model: str
    workspace: Path
    tests_exist: bool
    tests_pass: bool
    code_exists: bool
    ruff_clean: bool
    issues: list[str]

def validate_workspace(workspace: Path, task_id: str, model: str) -> ValidationResult:
    """Validate a single workspace."""
    issues = []

    # Check if workspace exists
    if not workspace.exists():
        return ValidationResult(
            task_id=task_id,
            model=model,
            workspace=workspace,
            tests_exist=False,
            tests_pass=False,
            code_exists=False,
            ruff_clean=False,
            issues=["Workspace does not exist"]
        )

    # Find test files
    test_files = list(workspace.glob("**/test_*.py")) + list(workspace.glob("**/tests/*.py"))
    tests_exist = len(test_files) > 0

    # Find Python code files (excluding tests and __pycache__)
    code_files = [
        f for f in workspace.rglob("*.py")
        if "__pycache__" not in str(f) and "test_" not in f.name and f.parent.name != "tests"
    ]
    code_exists = len(code_files) > 0

    if not code_exists:
        issues.append("No Python code files found")

    # Run pytest if tests exist
    tests_pass = False
    if tests_exist:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=30
            )
            tests_pass = result.returncode == 0
            if not tests_pass:
                issues.append(f"Tests failed: {result.stdout[:200]}")
        except subprocess.TimeoutExpired:
            issues.append("Tests timed out")
        except Exception as e:
            issues.append(f"Test execution error: {e}")
    else:
        issues.append("No test files found")

    # Run ruff if code exists
    ruff_clean = False
    if code_exists:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "."],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=10
            )
            ruff_clean = result.returncode == 0
            if not ruff_clean:
                issues.append(f"Ruff issues: {result.stdout[:200]}")
        except Exception as e:
            # Ruff not installed or error - not critical
            pass

    return ValidationResult(
        task_id=task_id,
        model=model,
        workspace=workspace,
        tests_exist=tests_exist,
        tests_pass=tests_pass,
        code_exists=code_exists,
        ruff_clean=ruff_clean,
        issues=issues
    )

def main():
    """Validate all benchmark results."""
    # Load benchmark results
    results_files = sorted(Path("evaluation_results").glob("model_comparison_*.json"))
    if not results_files:
        print("No benchmark results found!")
        return

    latest_results = results_files[-1]
    print(f"Validating results from: {latest_results}")
    print("=" * 80)

    with open(latest_results) as f:
        data = json.load(f)

    # Validate each result
    validations = []
    for result in data["results"]:
        task_id = f"{result['level']}-V{result['variation']}"
        model = result["model"]
        workspace_name = result.get("workspace")

        if not workspace_name:
            print(f"⚠️  {task_id} ({model}): No workspace recorded")
            continue

        workspace = Path(workspace_name)
        validation = validate_workspace(workspace, task_id, model)
        validations.append(validation)

        # Print validation result
        status = "✅" if validation.tests_pass and validation.code_exists else "❌"
        print(f"{status} {task_id} ({model}): ", end="")

        if validation.tests_pass and validation.code_exists:
            print("VALID")
        else:
            print(f"ISSUES: {', '.join(validation.issues)}")

    # Summary by model
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for model in ["gpt-oss:20b", "qwen3:8b", "qwen3:14b"]:
        model_vals = [v for v in validations if v.model == model]
        valid_count = sum(1 for v in model_vals if v.tests_pass and v.code_exists)
        total = len(model_vals)

        print(f"\n{model}:")
        print(f"  Valid: {valid_count}/{total} ({100*valid_count/total:.0f}%)")
        print(f"  Tests exist: {sum(1 for v in model_vals if v.tests_exist)}/{total}")
        print(f"  Tests pass: {sum(1 for v in model_vals if v.tests_pass)}/{total}")
        print(f"  Code exists: {sum(1 for v in model_vals if v.code_exists)}/{total}")

        # Show failed tasks
        failed = [v for v in model_vals if not (v.tests_pass and v.code_exists)]
        if failed:
            print(f"  Failed tasks:")
            for v in failed:
                print(f"    - {v.task_id}: {', '.join(v.issues)}")

if __name__ == "__main__":
    main()
