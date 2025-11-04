#!/usr/bin/env python3
"""
Iterative Improvement Test - Orchestrator builds and improves a project over time.

This script:
1. Tells orchestrator to build a fractal rendering web-app
2. Repeatedly tells it to analyze and improve the app
3. Runs for 8 hours, iterating on the same workspace
"""
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configuration
DURATION_HOURS = 8
WORKSPACE = Path(".agent_workspaces/fractal-web-app")
LOG_FILE = Path("iterative_improvement_log.txt")

def log_message(message: str):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")

def run_orchestrator_task(task: str, iteration: int) -> tuple[bool, str]:
    """
    Run orchestrator with a task.

    Args:
        task: Task description for orchestrator
        iteration: Iteration number (0 = initial build, 1+ = improvements)

    Returns:
        Tuple of (success, output_summary)
    """
    log_message(f"Starting iteration {iteration}: {task[:80]}...")

    # For iteration 0 (initial build), use new workspace
    # For iterations 1+ (improvements), use existing workspace
    if iteration == 0:
        # Initial build - let orchestrator create new workspace
        cmd = [
            sys.executable,
            "orchestrator_agent.py",
            task,
            "--once"
        ]
    else:
        # Improvement - work in existing workspace
        # The task should reference the existing project
        cmd = [
            sys.executable,
            "orchestrator_agent.py",
            task,
            "--once"
        ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per task
            cwd="/workspace"
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            log_message(f"✅ Iteration {iteration} completed in {elapsed:.1f}s")

            # Extract summary from output
            output = result.stdout + result.stderr
            if "Summary:" in output:
                summary_start = output.rfind("Summary:")
                summary = output[summary_start:summary_start+200].split('\n')[0]
            else:
                summary = "(no summary found)"

            return True, summary
        else:
            log_message(f"❌ Iteration {iteration} failed after {elapsed:.1f}s")
            log_message(f"Error: {result.stderr[-500:]}")  # Last 500 chars of error
            return False, f"Failed: {result.returncode}"

    except subprocess.TimeoutExpired:
        log_message(f"⏱️ Iteration {iteration} timed out after 10 minutes")
        return False, "Timeout"
    except Exception as e:
        log_message(f"💥 Iteration {iteration} crashed: {e}")
        return False, f"Exception: {e}"

def find_workspace_path() -> Path | None:
    """Find the fractal app workspace after initial build."""
    workspaces_dir = Path(".agent_workspaces")
    if not workspaces_dir.exists():
        return None

    # Look for workspace containing "fractal" in name
    for workspace in workspaces_dir.iterdir():
        if workspace.is_dir() and "fractal" in workspace.name.lower():
            return workspace

    return None

def main():
    """Main iterative improvement loop."""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=DURATION_HOURS)

    log_message("=" * 80)
    log_message("ITERATIVE IMPROVEMENT TEST - FRACTAL WEB APP")
    log_message("=" * 80)
    log_message(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Duration: {DURATION_HOURS} hours")
    log_message("")

    iteration = 0
    workspace_path = None

    # Iteration 0: Initial build
    initial_task = """Create a fractal rendering web application with the following features:
- Interactive HTML/CSS/JavaScript interface
- Render Mandelbrot set fractals
- Allow zooming and panning
- Color gradient controls
- Responsive design
- Use Canvas API for rendering

Make it visually appealing and performant."""

    success, summary = run_orchestrator_task(initial_task, iteration)
    log_message(f"Initial build summary: {summary}")
    log_message("")

    if not success:
        log_message("Initial build failed. Aborting.")
        return

    # Find the workspace that was created
    time.sleep(2)  # Give filesystem time to settle
    workspace_path = find_workspace_path()

    if workspace_path:
        log_message(f"Found workspace: {workspace_path}")
    else:
        log_message("Warning: Could not find fractal workspace. Will continue anyway.")

    log_message("")

    iteration = 1

    # Iterative improvement loop
    improvement_prompts = [
        "Analyze the fractal web app and identify the top 3 most impactful improvements. Then implement those improvements. Focus on: performance, visual quality, user experience, or new features.",

        "Review the current state of the fractal app. Think deeply about what would make it more impressive. Implement your best ideas for improvements.",

        "Examine the fractal renderer. What mathematical features or visual effects would make it more interesting? Implement new fractal types or rendering techniques.",

        "Look at the user interface. How can we make it more intuitive and powerful? Add new controls or improve existing ones.",

        "Consider the code quality and architecture. Refactor for better performance, maintainability, or extensibility. Add comments and documentation.",

        "Think about advanced features: Julia sets, other fractal types, save/load functionality, animation, parameter presets. Implement the most valuable additions.",

        "Analyze performance bottlenecks. Optimize the rendering pipeline. Consider Web Workers, GPU acceleration, or algorithm improvements.",

        "Review the visual design. Enhance the aesthetics with better colors, gradients, UI polish, or artistic rendering modes.",
    ]

    prompt_index = 0

    while datetime.now() < end_time:
        remaining = end_time - datetime.now()
        log_message(f"Time remaining: {remaining.total_seconds()/3600:.1f} hours")

        # Cycle through improvement prompts
        prompt = improvement_prompts[prompt_index % len(improvement_prompts)]
        prompt_index += 1

        # Reference the existing project in the prompt
        full_prompt = f"Working on the fractal web app project: {prompt}"

        success, summary = run_orchestrator_task(full_prompt, iteration)
        log_message(f"Iteration {iteration} summary: {summary}")
        log_message("")

        iteration += 1

        # Small delay between iterations
        time.sleep(5)

    # Final report
    log_message("=" * 80)
    log_message("ITERATIVE IMPROVEMENT TEST COMPLETE")
    log_message("=" * 80)
    log_message(f"Total iterations: {iteration}")
    log_message(f"Duration: {DURATION_HOURS} hours")

    if workspace_path and workspace_path.exists():
        log_message(f"Final workspace: {workspace_path}")
        log_message("Files created:")
        for file in workspace_path.rglob("*"):
            if file.is_file() and not file.name.startswith('.'):
                rel_path = file.relative_to(workspace_path)
                log_message(f"  - {rel_path}")

    log_message("")
    log_message("Test complete! Check the workspace for the evolved fractal app.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("\n\nTest interrupted by user (Ctrl+C)")
        log_message("Partial results may be available in workspace.")
    except Exception as e:
        log_message(f"\n\nTest crashed with exception: {e}")
        import traceback
        traceback.print_exc()
