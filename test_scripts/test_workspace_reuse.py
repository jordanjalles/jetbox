#!/usr/bin/env python3
"""Quick test to verify workspace reuse works with --workspace flag."""
import subprocess
import sys
from pathlib import Path
import time

WORKSPACE = Path(".agent_workspaces/fractal-webapp-project")

def run_test_iteration(task: str, iteration: int) -> bool:
    """Run a test iteration."""
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}: {task[:50]}...")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        "orchestrator_agent.py",
        "--workspace",
        str(WORKSPACE),
        task,
        "--once"
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, timeout=300)
    elapsed = time.time() - start

    success = result.returncode == 0
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"\n{status} - Iteration {iteration} took {elapsed:.1f}s\n")

    return success

def main():
    """Run quick test."""
    print("WORKSPACE REUSE TEST")
    print("=" * 60)
    print(f"Workspace: {WORKSPACE}")

    # Ensure workspace exists
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    # Initial file count
    initial_files = len(list(WORKSPACE.glob("*")))
    print(f"Initial files in workspace: {initial_files}")
    print()

    # Iteration 0: Create initial app
    task0 = "Create a simple fractal renderer web app with Mandelbrot set, canvas rendering, and zoom controls"
    if not run_test_iteration(task0, 0):
        print("❌ Iteration 0 failed. Aborting.")
        return

    # Check workspace
    files_after_0 = list(WORKSPACE.glob("*"))
    print(f"Files after iteration 0: {len(files_after_0)}")
    for f in files_after_0:
        print(f"  - {f.name}")
    print()

    # Iteration 1: Improve
    task1 = "Add color controls to the fractal renderer"
    if not run_test_iteration(task1, 1):
        print("❌ Iteration 1 failed. Aborting.")
        return

    # Check workspace again
    files_after_1 = list(WORKSPACE.glob("*"))
    print(f"Files after iteration 1: {len(files_after_1)}")
    for f in files_after_1:
        print(f"  - {f.name}")
    print()

    # Check for unwanted workspaces
    unwanted = list(Path(".agent_workspaces").glob("create-*")) + \
               list(Path(".agent_workspaces").glob("add-*"))

    if unwanted:
        print("❌ FAILED: Found unwanted workspaces:")
        for w in unwanted:
            print(f"  - {w}")
    else:
        print("✅ SUCCESS: No unwanted workspaces created!")
        print(f"✅ All work done in: {WORKSPACE}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest crashed: {e}")
        import traceback
        traceback.print_exc()
