#!/usr/bin/env python3
"""Two-iteration test to verify workspace reuse."""
import subprocess
import sys
from pathlib import Path
import time

WORKSPACE = Path(".agent_workspaces/fractal-webapp-project")

print("=" * 60)
print("TWO-ITERATION WORKSPACE REUSE TEST")
print("=" * 60)
print(f"Workspace: {WORKSPACE}")
print()

# Iteration 0: Create initial files
print("ITERATION 0: Creating initial HTML/CSS/JS files")
print("-" * 60)

cmd = [
    sys.executable,
    "orchestrator_agent.py",
    "--workspace",
    str(WORKSPACE),
    "Create a simple web page: index.html with heading, style.css with colors, script.js with alert function",
    "--once"
]

start = time.time()
result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
elapsed = time.time() - start

print(f"Exit code: {result.returncode} ({elapsed:.1f}s)")

if result.returncode != 0:
    print("FAILED!")
    print(result.stderr[-500:])
    sys.exit(1)

# Check files after iteration 0
files_0 = sorted([f.name for f in list(WORKSPACE.glob("*.html")) + list(WORKSPACE.glob("*.js")) + list(WORKSPACE.glob("*.css"))])
print(f"Files created: {files_0}")

if not files_0:
    print("❌ ERROR: No code files created in iteration 0!")
    sys.exit(1)

print()

# Iteration 1: Modify existing files
print("ITERATION 1: Adding features to existing files")
print("-" * 60)

cmd[5] = "Add a red button that says 'Click Me' to index.html and make it call a new function in script.js"

start = time.time()
result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
elapsed = time.time() - start

print(f"Exit code: {result.returncode} ({elapsed:.1f}s)")

if result.returncode != 0:
    print("FAILED!")
    print(result.stderr[-500:])
    sys.exit(1)

# Check files after iteration 1
files_1 = sorted([f.name for f in list(WORKSPACE.glob("*.html")) + list(WORKSPACE.glob("*.js")) + list(WORKSPACE.glob("*.css"))])
print(f"Files after iteration 1: {files_1}")

print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(f"Files: {files_1}")
print()
print("index.html content:")
with open(WORKSPACE / "index.html") as f:
    print(f.read()[:300])
print()
print("script.js content:")
with open(WORKSPACE / "script.js") as f:
    print(f.read()[:300])
print()

if files_1:
    print("✅ SUCCESS: Workspace reuse is working!")
else:
    print("❌ FAILED: No files in workspace")
    sys.exit(1)
