#!/usr/bin/env python3
"""Short 3-iteration test to verify workspace reuse."""
import subprocess
import sys
from pathlib import Path
import time

WORKSPACE = Path(".agent_workspaces/fractal-webapp-project")

tasks = [
    "Create index.html with a red canvas element (must be HTML/CSS, not Python)",
    "Add a green button to index.html that says 'Click Me'",
    "Add script.js with a function that alerts 'Hello' when button is clicked, and link it in index.html"
]

for i, task in enumerate(tasks):
    print(f"\n{'='*60}")
    print(f"ITERATION {i}: {task[:50]}...")
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"✅ SUCCESS - {elapsed:.1f}s")
    else:
        print(f"❌ FAILED - {elapsed:.1f}s")
        print(f"Error: {result.stderr[-500:]}")
        break

    # Check files
    files = list(WORKSPACE.glob("*.html")) + list(WORKSPACE.glob("*.js")) + list(WORKSPACE.glob("*.css"))
    print(f"Files in workspace: {[f.name for f in files]}")
    print()

print(f"\n{'='*60}")
print("FINAL WORKSPACE STATE")
print(f"{'='*60}\n")
subprocess.run(["ls", "-la", str(WORKSPACE)])
print()
subprocess.run(["find", str(WORKSPACE), "-name", "*.html", "-o", "-name", "*.js", "-o", "-name", "*.css"])
