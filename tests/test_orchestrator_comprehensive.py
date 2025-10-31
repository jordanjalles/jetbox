"""
Comprehensive orchestrator testing suite.

Tests TaskExecutorAgent in complex scenarios:
1. Complex multi-step delegation
2. Error recovery and retry logic
3. Multiple executors on same workspace
4. Jetbox notes context continuity
5. Mixed success/failure scenarios
"""
import tempfile
from pathlib import Path
import json

from task_executor_agent import TaskExecutorAgent


def test_complex_web_project():
    """
    Test orchestrator handling complex web project with HTML, CSS, and JS.
    Tests: Multi-file creation, cross-file dependencies, testing workflow.
    """
    print("\n" + "="*70)
    print("TEST: Complex Web Project")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Complex goal requiring multiple files and testing
        goal = """Create a simple todo list web app with:
1. HTML file (index.html) with form to add todos and list to display them
2. CSS file (style.css) with clean, modern styling
3. JavaScript file (app.js) with add/delete/mark-complete functionality
4. All files should work together when opened in a browser
"""

        executor = TaskExecutorAgent(
            workspace=workspace,
            goal=goal,
            max_rounds=40
        )

        result = executor.run()

        # Check workspace
        workspace_dir = executor.workspace_manager.workspace_dir

        # Verify all files created
        html_file = workspace_dir / "index.html"
        css_file = workspace_dir / "style.css"
        js_file = workspace_dir / "app.js"

        files_created = all([
            html_file.exists(),
            css_file.exists(),
            js_file.exists()
        ])

        # Verify HTML references CSS and JS
        if html_file.exists():
            html_content = html_file.read_text()
            has_css_link = "style.css" in html_content
            has_js_link = "app.js" in html_content
        else:
            has_css_link = has_js_link = False

        print(f"\n{'='*70}")
        print(f"Result: {result.get('status')}")
        print(f"Files created: {files_created}")
        print(f"HTML links CSS: {has_css_link}")
        print(f"HTML links JS: {has_js_link}")
        print(f"{'='*70}")

        return {
            "test": "complex_web_project",
            "success": files_created and has_css_link and has_js_link,
            "files_created": files_created,
            "cross_references": has_css_link and has_js_link
        }


def test_iterative_refinement():
    """
    Test multiple executors refining the same codebase.
    Tests: Workspace reuse, context continuity, jetbox notes.
    """
    print("\n" + "="*70)
    print("TEST: Iterative Refinement")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Phase 1: Create basic calculator
        print("\nPhase 1: Create basic calculator")
        executor1 = TaskExecutorAgent(
            workspace=workspace,
            goal="Create calculator.py with add(a, b) and subtract(a, b) functions. Write tests.",
            max_rounds=20
        )
        result1 = executor1.run()
        calc_workspace = executor1.workspace_manager.workspace_dir

        # Phase 2: Add multiplication and division
        print("\nPhase 2: Add multiply and divide")
        executor2 = TaskExecutorAgent(
            workspace=workspace,
            workspace_path=calc_workspace,
            goal="Add multiply(a, b) and divide(a, b) functions to calculator.py. Update tests.",
            max_rounds=20
        )
        result2 = executor2.run()

        # Phase 3: Add power function
        print("\nPhase 3: Add power function")
        executor3 = TaskExecutorAgent(
            workspace=workspace,
            workspace_path=calc_workspace,
            goal="Add power(a, b) function to calculator.py that returns a**b. Update tests.",
            max_rounds=20
        )
        result3 = executor3.run()

        # Check final state
        calc_file = calc_workspace / "calculator.py"
        test_file = list(calc_workspace.glob("test_*.py"))
        notes_file = calc_workspace / "jetboxnotes.md"

        if calc_file.exists():
            calc_content = calc_file.read_text()
            has_add = "def add" in calc_content
            has_subtract = "def subtract" in calc_content
            has_multiply = "def multiply" in calc_content
            has_divide = "def divide" in calc_content
            has_power = "def power" in calc_content
        else:
            has_add = has_subtract = has_multiply = has_divide = has_power = False

        functions_count = sum([has_add, has_subtract, has_multiply, has_divide, has_power])

        print(f"\n{'='*70}")
        print(f"Phase 1: {result1.get('status')}")
        print(f"Phase 2: {result2.get('status')}")
        print(f"Phase 3: {result3.get('status')}")
        print(f"Functions implemented: {functions_count}/5")
        print(f"Test file exists: {len(test_file) > 0}")
        print(f"Notes file exists: {notes_file.exists()}")
        print(f"{'='*70}")

        return {
            "test": "iterative_refinement",
            "success": functions_count >= 4 and len(test_file) > 0,
            "functions_count": functions_count,
            "has_tests": len(test_file) > 0,
            "has_notes": notes_file.exists()
        }


def test_error_recovery():
    """
    Test executor handling errors and recovering.
    Tests: Error detection, retry logic, graceful degradation.
    """
    print("\n" + "="*70)
    print("TEST: Error Recovery")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Goal that will likely hit errors (complex requirement)
        goal = """Create a Python package 'mathlib' with:
1. __init__.py that exports all functions
2. advanced.py with factorial, fibonacci, and is_prime functions
3. Complete test coverage with pytest
4. All tests must pass
5. All code must pass ruff linting
"""

        executor = TaskExecutorAgent(
            workspace=workspace,
            goal=goal,
            max_rounds=50  # Give it room to recover from errors
        )

        result = executor.run()

        # Check if it recovered from errors
        workspace_dir = executor.workspace_manager.workspace_dir

        # Check for package structure
        init_file = workspace_dir / "mathlib" / "__init__.py"
        advanced_file = workspace_dir / "mathlib" / "advanced.py"
        test_files = list(workspace_dir.glob("**/test_*.py"))

        package_created = init_file.exists() and advanced_file.exists()
        tests_created = len(test_files) > 0

        print(f"\n{'='*70}")
        print(f"Result: {result.get('status')}")
        print(f"Package structure: {package_created}")
        print(f"Tests created: {tests_created}")
        print(f"Total rounds: {executor.state.total_rounds}")
        print(f"{'='*70}")

        return {
            "test": "error_recovery",
            "success": package_created and tests_created,
            "package_structure": package_created,
            "tests_created": tests_created,
            "rounds": executor.state.total_rounds
        }


def test_mixed_success_failure():
    """
    Test executor with partially achievable goals.
    Tests: Partial success handling, clear failure reporting.
    """
    print("\n" + "="*70)
    print("TEST: Mixed Success/Failure")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Goal with easy and hard parts
        goal = """Create:
1. hello.py that prints "Hello, World!" (EASY)
2. Complex neural network implementation with backprop (HARD - likely to fail)
3. tests for hello.py (EASY)
"""

        executor = TaskExecutorAgent(
            workspace=workspace,
            goal=goal,
            max_rounds=25
        )

        result = executor.run()

        workspace_dir = executor.workspace_manager.workspace_dir

        # Check what was achieved
        hello_file = workspace_dir / "hello.py"
        test_files = list(workspace_dir.glob("test_*.py"))
        network_files = list(workspace_dir.glob("*network*.py"))

        easy_tasks_done = hello_file.exists() and len(test_files) > 0
        hard_task_attempted = len(network_files) > 0

        print(f"\n{'='*70}")
        print(f"Result: {result.get('status')}")
        print(f"Easy tasks done: {easy_tasks_done}")
        print(f"Hard task attempted: {hard_task_attempted}")
        print(f"{'='*70}")

        return {
            "test": "mixed_success_failure",
            "success": easy_tasks_done,  # Success if easy tasks work
            "easy_tasks": easy_tasks_done,
            "hard_task_attempted": hard_task_attempted
        }


def test_context_size_across_executors():
    """
    Test that context management works across multiple executor runs.
    Tests: Context isolation, no context leakage between executors.
    """
    print("\n" + "="*70)
    print("TEST: Context Size Across Executors")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Run 3 separate goals
        goals = [
            "Create file1.py with class ClassOne",
            "Create file2.py with class ClassTwo",
            "Create file3.py with class ClassThree"
        ]

        results = []
        for i, goal in enumerate(goals):
            print(f"\nExecutor {i+1}: {goal}")
            executor = TaskExecutorAgent(
                workspace=workspace,
                goal=goal,
                max_rounds=10
            )
            result = executor.run()
            results.append({
                "status": result.get("status"),
                "rounds": executor.state.total_rounds,
                "messages": len(executor.state.messages)
            })

        # Each executor should start fresh (low message count)
        all_successful = all(r["status"] == "success" for r in results)
        reasonable_rounds = all(r["rounds"] < 8 for r in results)

        print(f"\n{'='*70}")
        for i, r in enumerate(results):
            print(f"Executor {i+1}: {r['status']}, {r['rounds']} rounds, {r['messages']} msgs")
        print(f"All successful: {all_successful}")
        print(f"Reasonable rounds: {reasonable_rounds}")
        print(f"{'='*70}")

        return {
            "test": "context_size_across_executors",
            "success": all_successful and reasonable_rounds,
            "all_successful": all_successful,
            "reasonable_rounds": reasonable_rounds,
            "results": results
        }


def test_jetbox_notes_accumulation():
    """
    Test that jetbox notes accumulate context across multiple runs.
    Tests: Notes persistence, context building, summary quality.
    """
    print("\n" + "="*70)
    print("TEST: Jetbox Notes Accumulation")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Phase 1: Create base
        executor1 = TaskExecutorAgent(
            workspace=workspace,
            goal="Create shapes.py with Circle and Rectangle classes",
            max_rounds=15
        )
        result1 = executor1.run()
        shapes_workspace = executor1.workspace_manager.workspace_dir
        notes_file = shapes_workspace / "jetboxnotes.md"

        if notes_file.exists():
            notes1_size = len(notes_file.read_text())
            notes1_content = notes_file.read_text()
        else:
            notes1_size = 0
            notes1_content = ""

        # Phase 2: Add more shapes
        executor2 = TaskExecutorAgent(
            workspace=workspace,
            workspace_path=shapes_workspace,
            goal="Add Triangle and Square classes to shapes.py",
            max_rounds=15
        )
        result2 = executor2.run()

        if notes_file.exists():
            notes2_size = len(notes_file.read_text())
            notes2_content = notes_file.read_text()
        else:
            notes2_size = 0
            notes2_content = ""

        # Phase 3: Add tests
        executor3 = TaskExecutorAgent(
            workspace=workspace,
            workspace_path=shapes_workspace,
            goal="Write comprehensive tests for all shape classes",
            max_rounds=15
        )
        result3 = executor3.run()

        if notes_file.exists():
            notes3_size = len(notes_file.read_text())
            notes3_content = notes_file.read_text()
        else:
            notes3_size = 0
            notes3_content = ""

        notes_grew = notes3_size > notes2_size > notes1_size
        mentions_shapes = all([
            "Circle" in notes3_content or "circle" in notes3_content.lower(),
            "Rectangle" in notes3_content or "rectangle" in notes3_content.lower()
        ])

        print(f"\n{'='*70}")
        print(f"Phase 1 result: {result1.get('status')}")
        print(f"Phase 2 result: {result2.get('status')}")
        print(f"Phase 3 result: {result3.get('status')}")
        print(f"Notes size progression: {notes1_size} → {notes2_size} → {notes3_size}")
        print(f"Notes grew: {notes_grew}")
        print(f"Mentions shapes: {mentions_shapes}")
        print(f"{'='*70}")

        return {
            "test": "jetbox_notes_accumulation",
            "success": notes_grew and mentions_shapes,
            "notes_growth": [notes1_size, notes2_size, notes3_size],
            "mentions_shapes": mentions_shapes
        }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPREHENSIVE ORCHESTRATOR TESTING")
    print("="*70)

    tests = [
        test_complex_web_project,
        test_iterative_refinement,
        test_error_recovery,
        test_mixed_success_failure,
        test_context_size_across_executors,
        test_jetbox_notes_accumulation
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} failed with exception: {e}")
            results.append({
                "test": test_func.__name__,
                "success": False,
                "error": str(e)
            })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for r in results if r.get("success"))
    total = len(results)
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.1f}%)")

    for result in results:
        status = "✓" if result.get("success") else "✗"
        print(f"{status} {result['test']}")

    # Save results
    output_file = Path("orchestrator_comprehensive_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {"passed": passed, "total": total, "pass_rate": passed/total},
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
