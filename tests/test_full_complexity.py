#!/usr/bin/env python3
"""
Comprehensive integration test for Jetbox v3.1

Tests all major systems:
- Workspace coordination across agents
- Behavior system (file tools, task management, workspace notes)
- Delegation (Orchestrator → Architect → TaskExecutor)
- Task breakdown management
- LLM-generated notes

Expected outcome:
- All agents work in same workspace
- Architecture files created by Architect
- Implementation files created by TaskExecutors
- Task breakdown properly managed
- Workspace notes generated with LLM summaries
"""
import sys
from pathlib import Path
import json

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator_main import OrchestratorAgent


def verify_workspace_structure(workspace_dir: Path) -> dict:
    """Verify expected files were created in the workspace."""
    results = {
        "workspace_exists": workspace_dir.exists(),
        "architecture_dir": (workspace_dir / "architecture").exists(),
        "task_breakdown": (workspace_dir / "architecture" / "task-breakdown.json").exists(),
        "workspace_notes": (workspace_dir / "workspace_task_notes.md").exists(),
        "python_files": list(workspace_dir.glob("**/*.py")),
        "architecture_files": list((workspace_dir / "architecture").glob("**/*.md")) if (workspace_dir / "architecture").exists() else [],
    }
    return results


def analyze_task_breakdown(workspace_dir: Path) -> dict:
    """Analyze the task breakdown file."""
    task_file = workspace_dir / "architecture" / "task-breakdown.json"
    if not task_file.exists():
        return {"exists": False}

    with open(task_file) as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    return {
        "exists": True,
        "total_tasks": len(tasks),
        "completed": sum(1 for t in tasks if t.get("status") == "completed"),
        "in_progress": sum(1 for t in tasks if t.get("status") == "in_progress"),
        "pending": sum(1 for t in tasks if t.get("status") == "pending"),
        "failed": sum(1 for t in tasks if t.get("status") == "failed"),
    }


def check_workspace_notes(workspace_dir: Path) -> dict:
    """Check if workspace notes were generated."""
    notes_file = workspace_dir / "workspace_task_notes.md"
    if not notes_file.exists():
        return {"exists": False}

    with open(notes_file) as f:
        content = f.read()

    return {
        "exists": True,
        "length": len(content),
        "has_sections": any(marker in content for marker in ["## ", "### "]),
        "has_bullets": "- " in content or "* " in content,
    }


def main():
    print("=" * 80)
    print("JETBOX v3.1 FULL COMPLEXITY INTEGRATION TEST")
    print("=" * 80)
    print()

    # Test goal: Create a Python package with tests
    goal = """Create a Python package called 'calculator' with:
1. Basic operations: add, subtract, multiply, divide
2. Advanced operations: power, square_root
3. Each function in its own module file
4. Comprehensive pytest tests for all functions
5. Handle edge cases (division by zero, negative square roots)
6. Run tests and fix any failures"""

    workspace_dir = Path(".agent_workspaces/full_complexity_test")

    print(f"Goal: {goal}")
    print(f"Workspace: {workspace_dir}")
    print()

    # Create orchestrator
    print("Creating OrchestratorAgent...")
    orchestrator = OrchestratorAgent(workspace=workspace_dir)

    # Run with timeout
    print("Running orchestrator (max 30 rounds)...")
    print("-" * 80)

    try:
        result = orchestrator.run(
            initial_message=goal,
            max_rounds=30
        )
        print("-" * 80)
        print(f"\nOrchestrator Result: {result}")
    except Exception as e:
        print("-" * 80)
        print(f"\nOrchestrator Error: {e}")
        result = {"success": False, "error": str(e)}

    print()
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()

    # Verify workspace structure
    print("1. Workspace Structure:")
    structure = verify_workspace_structure(workspace_dir)
    print(f"   - Workspace exists: {structure['workspace_exists']}")
    print(f"   - Architecture dir: {structure['architecture_dir']}")
    print(f"   - Task breakdown: {structure['task_breakdown']}")
    print(f"   - Workspace notes: {structure['workspace_notes']}")
    print(f"   - Python files created: {len(structure['python_files'])}")
    for py_file in structure['python_files'][:10]:  # Show first 10
        print(f"     • {py_file.relative_to(workspace_dir)}")
    print(f"   - Architecture docs: {len(structure['architecture_files'])}")
    for arch_file in structure['architecture_files'][:5]:  # Show first 5
        print(f"     • {arch_file.relative_to(workspace_dir / 'architecture')}")
    print()

    # Analyze task breakdown
    print("2. Task Breakdown:")
    breakdown = analyze_task_breakdown(workspace_dir)
    if breakdown["exists"]:
        print(f"   - Total tasks: {breakdown['total_tasks']}")
        print(f"   - Completed: {breakdown['completed']}")
        print(f"   - In progress: {breakdown['in_progress']}")
        print(f"   - Pending: {breakdown['pending']}")
        print(f"   - Failed: {breakdown['failed']}")
    else:
        print("   - No task breakdown found")
    print()

    # Check workspace notes
    print("3. Workspace Notes:")
    notes = check_workspace_notes(workspace_dir)
    if notes["exists"]:
        print("   - File exists: True")
        print(f"   - Content length: {notes['length']} chars")
        print(f"   - Has sections: {notes['has_sections']}")
        print(f"   - Has bullets: {notes['has_bullets']}")

        # Show first 500 chars
        notes_file = workspace_dir / "workspace_task_notes.md"
        with open(notes_file) as f:
            preview = f.read(500)
        print("\n   Preview:")
        for line in preview.split('\n')[:10]:
            print(f"   | {line}")
    else:
        print("   - No workspace notes found")
    print()

    # Overall assessment
    print("=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    checks = [
        ("Workspace created", structure['workspace_exists']),
        ("Architecture directory", structure['architecture_dir']),
        ("Task breakdown exists", structure['task_breakdown']),
        ("Python files created", len(structure['python_files']) > 0),
        ("Workspace notes generated", notes['exists']),
    ]

    passed = sum(1 for _, check in checks if check)
    total = len(checks)

    print()
    for check_name, check_result in checks:
        status = "✓" if check_result else "✗"
        print(f"{status} {check_name}")

    print()
    print(f"Result: {passed}/{total} checks passed")
    print()

    if passed == total:
        print("🎉 FULL COMPLEXITY TEST PASSED!")
    else:
        print("⚠️  Some checks failed - review output above")

    print()
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
