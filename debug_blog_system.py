#!/usr/bin/env python3
"""
Debug script to trace blog_system execution and identify hallucination point.
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

def main():
    workspace = tempfile.mkdtemp(prefix='blog_debug_')
    print(f"Workspace: {workspace}")
    print(f"Starting at: {time.strftime('%H:%M:%S')}\n")

    cmd = [
        sys.executable, '/workspace/agent.py',
        '--team', 'default',
        '--workspace', workspace,
        'Create blog system: Post model, Comment model, BlogManager with CRUD operations, persistence to JSON'
    ]

    print(f"Command: {' '.join(cmd)}\n")
    print("="*80)
    print("AGENT OUTPUT (streaming)")
    print("="*80)

    # Run with streaming output
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output_lines = []
    try:
        # Stream output line by line for 3 minutes
        timeout_time = time.time() + 180  # 3 minutes
        while time.time() < timeout_time:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    # Process ended
                    break
                time.sleep(0.1)
                continue

            output_lines.append(line.rstrip())
            print(line.rstrip())

        # Check if still running
        if proc.poll() is None:
            print("\n" + "="*80)
            print("TIMEOUT after 3 minutes - killing process")
            print("="*80)
            proc.terminate()
            proc.wait(timeout=5)
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("Interrupted - killing process")
        print("="*80)
        proc.terminate()
        proc.wait(timeout=5)

    returncode = proc.returncode

    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"Exit code: {returncode}")
    print(f"Total output lines: {len(output_lines)}")

    # Analyze output for delegation calls
    print("\n" + "="*80)
    print("DELEGATION ANALYSIS")
    print("="*80)

    delegation_calls = []
    for i, line in enumerate(output_lines):
        if 'DELEGATING TO' in line:
            delegation_calls.append((i, line))
        elif 'delegate_to_executor' in line.lower() or 'consult_architect' in line.lower():
            delegation_calls.append((i, line))

    if delegation_calls:
        print(f"Found {len(delegation_calls)} delegation-related lines:")
        for line_num, line in delegation_calls:
            print(f"  Line {line_num}: {line}")
    else:
        print("NO delegation calls found!")

    # Check for completion markers
    print("\n" + "="*80)
    print("COMPLETION ANALYSIS")
    print("="*80)

    completion_markers = []
    for i, line in enumerate(output_lines):
        if 'mark_complete' in line.lower() or 'completed' in line.lower() or 'task completed' in line.lower():
            completion_markers.append((i, line))

    if completion_markers:
        print(f"Found {len(completion_markers)} completion-related lines:")
        for line_num, line in completion_markers[-10:]:  # Last 10
            print(f"  Line {line_num}: {line}")

    # Check workspace files
    print("\n" + "="*80)
    print("WORKSPACE FILES")
    print("="*80)

    workspace_path = Path(workspace)
    all_files = []
    for root, dirs, files in workspace_path.rglob('*'):
        if root.is_file():
            all_files.append(root)

    if all_files:
        print(f"Found {len(all_files)} files:")
        for f in all_files:
            rel_path = f.relative_to(workspace_path)
            size = f.stat().st_size
            print(f"  {rel_path} ({size} bytes)")
    else:
        print("NO files created (workspace is empty)")

    # Show last 20 lines of output
    print("\n" + "="*80)
    print("LAST 20 LINES OF OUTPUT")
    print("="*80)
    for line in output_lines[-20:]:
        print(line)

if __name__ == "__main__":
    main()
