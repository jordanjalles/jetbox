#!/usr/bin/env python3
"""
Iterative Improvement Test with Fixed Workspace.

This version explicitly tells orchestrator to use a specific workspace path
for all iterations, ensuring continuous improvement on the same codebase.
"""
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configuration
DURATION_HOURS = 8
WORKSPACE = Path(".agent_workspaces/fractal-webapp-project")
LOG_FILE = Path("iterative_improvement_fixed_log.txt")

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
        iteration: Iteration number

    Returns:
        Tuple of (success, output_summary)
    """
    log_message(f"Starting iteration {iteration}: {task[:80]}...")

    cmd = [
        sys.executable,
        "orchestrator_agent.py",
        "--workspace",
        str(WORKSPACE),
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

        # Capture and save debug output
        output = result.stdout + result.stderr
        debug_file = Path(f"/tmp/iteration_{iteration}_debug.log")
        debug_file.write_text(output)

        if result.returncode == 0:
            log_message(f"✅ Iteration {iteration} completed in {elapsed:.1f}s")

            if "Summary:" in output:
                summary_start = output.rfind("Summary:")
                summary = output[summary_start:summary_start+200].split('\n')[0]
            else:
                summary = "(no summary found)"

            return True, summary
        else:
            log_message(f"❌ Iteration {iteration} failed after {elapsed:.1f}s")
            log_message(f"Error: {result.stderr[-500:]}")
            return False, f"Failed: {result.returncode}"

    except subprocess.TimeoutExpired:
        log_message(f"⏱️ Iteration {iteration} timed out after 10 minutes")
        return False, "Timeout"
    except Exception as e:
        log_message(f"💥 Iteration {iteration} crashed: {e}")
        return False, f"Exception: {e}"

def main():
    """Main iterative improvement loop with fixed workspace."""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=DURATION_HOURS)

    log_message("=" * 80)
    log_message("ITERATIVE IMPROVEMENT TEST - FIXED WORKSPACE")
    log_message("=" * 80)
    log_message(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Duration: {DURATION_HOURS} hours")
    log_message(f"Workspace: {WORKSPACE}")
    log_message("")

    # Check if workspace exists
    if not WORKSPACE.exists():
        log_message(f"Workspace does not exist, will be created: {WORKSPACE}")
    else:
        log_message(f"Workspace exists with {len(list(WORKSPACE.glob('*')))} items")

    iteration = 0

    # Iteration 0: Initial build in specific workspace
    # Keep it SIMPLE to avoid model hallucination - details will be added iteratively
    initial_task = """Create a simple fractal rendering web application with HTML, CSS, and JavaScript.

Create these files: index.html, script.js, style.css

The HTML should have a canvas element. The JavaScript should render a basic Mandelbrot fractal on the canvas."""

    success, summary = run_orchestrator_task(initial_task, iteration)
    log_message(f"Initial build summary: {summary}")
    log_message("")

    if not success:
        log_message("Initial build failed. Aborting.")
        return

    iteration = 1

    # Iterative improvement loop
    improvement_prompts = [
        "Analyze the current fractal web app code and identify the top 3 most impactful improvements (performance, visual quality, UX, or features). Then implement them.",

        "Review the current state of the fractal app and implement your best ideas for enhancements.",

        "Enhance the fractal renderer by adding new fractal types (Julia sets, Burning Ship, etc.) or advanced rendering techniques.",

        "Improve the UI/UX to make the interface more intuitive and add new controls.",

        "Refactor the code to improve architecture, add documentation, and enhance maintainability.",

        "Add advanced features such as presets, save/load functionality, animation, or other valuable additions.",

        "Optimize performance by profiling and optimizing the rendering pipeline. Consider Web Workers or GPU acceleration.",

        "Enhance visual design with improved colors, gradients, UI polish, and artistic rendering modes.",
    ]

    prompt_index = 0

    while datetime.now() < end_time:
        remaining = end_time - datetime.now()
        log_message(f"Time remaining: {remaining.total_seconds()/3600:.1f} hours")

        # Cycle through improvement prompts
        prompt = improvement_prompts[prompt_index % len(improvement_prompts)]
        prompt_index += 1

        success, summary = run_orchestrator_task(prompt, iteration)
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
    log_message(f"Workspace: {WORKSPACE}")

    if WORKSPACE.exists():
        files = list(WORKSPACE.rglob("*"))
        log_message(f"Total files in workspace: {len([f for f in files if f.is_file()])}")
        log_message("Files created:")
        for file in WORKSPACE.rglob("*"):
            if file.is_file() and not file.name.startswith('.'):
                rel_path = file.relative_to(WORKSPACE)
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
