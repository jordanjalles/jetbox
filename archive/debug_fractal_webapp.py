#!/usr/bin/env python3
"""
3-Hour Debugging Session for Fractal Webapp.
Focus on making sure the webapp actually works.
"""
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configuration
DURATION_HOURS = 3
WORKSPACE = Path(".agent_workspaces/fractal-webapp-project")
LOG_FILE = Path("debug_fractal_webapp.log")

def log_message(message: str):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")

def run_orchestrator_task(task: str, iteration: int) -> tuple[bool, str]:
    """Run orchestrator with a task."""
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
        debug_file = Path(f"/tmp/debug_iteration_{iteration}.log")
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
            return False, f"Exit code: {result.returncode}"

    except subprocess.TimeoutExpired:
        log_message(f"⏱ Iteration {iteration} timed out after 10 minutes")
        return False, "Timeout"
    except Exception as e:
        log_message(f"💥 Iteration {iteration} crashed: {e}")
        return False, f"Exception: {e}"

def main():
    """Main debugging loop."""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=DURATION_HOURS)

    log_message("=" * 80)
    log_message("3-HOUR DEBUGGING SESSION - FRACTAL WEBAPP")
    log_message("=" * 80)
    log_message(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Duration: {DURATION_HOURS} hours")
    log_message(f"Workspace: {WORKSPACE}")
    log_message("")

    # Check workspace
    if not WORKSPACE.exists():
        log_message(f"Workspace does not exist, will be created: {WORKSPACE}")
    else:
        log_message(f"Workspace exists with {len(list(WORKSPACE.glob('*')))} items")

    iteration = 0

    # Iteration 0: Create basic working webapp
    initial_task = """Create a working fractal webapp with these files: index.html, script.js, style.css

The webapp must:
1. Display a canvas when index.html is opened in a browser
2. Actually render a Mandelbrot fractal (not just a blank canvas)
3. Have all JavaScript logic properly connected
4. Use proper HTML5 Canvas API calls

Make sure the HTML loads the script, the script accesses the canvas, and pixels are actually drawn."""

    success, summary = run_orchestrator_task(initial_task, iteration)
    log_message(f"Initial build summary: {summary}")
    log_message("")

    if not success:
        log_message("Initial build failed. Aborting.")
        return

    iteration = 1

    # Debugging iterations
    debugging_prompts = [
        "Test the fractal webapp by reading all files. Verify: 1) HTML links to script correctly, 2) Script gets canvas element correctly, 3) Mandelbrot calculation logic is correct, 4) Pixels are actually drawn to canvas. Fix any bugs found.",

        "Open and analyze index.html, script.js, and style.css. Check for: missing canvas.getContext(), incorrect element IDs, broken event listeners, mathematical errors in fractal calculation. Fix all issues.",

        "Debug the Mandelbrot rendering: verify the iteration loop runs, complex number math is correct, color mapping works, and ImageData is created and put on canvas. Add console.log statements if needed for debugging.",

        "Test that the canvas is visible and drawing happens on page load. Fix any issues with: canvas size, drawing initialization, DOMContentLoaded event, or script execution order.",

        "Review the complete webapp flow: HTML→script load→canvas access→Mandelbrot calculation→pixel drawing. Fix any broken links in this chain.",

        "Add zoom and pan controls to make the webapp interactive. Verify mouse events work correctly.",

        "Test color gradients and make the fractal visually appealing. Ensure colors map correctly to iteration counts.",

        "Optimize rendering performance. Add progress indication or loading states if rendering is slow.",
    ]

    prompt_index = 0

    while datetime.now() < end_time:
        time_remaining = (end_time - datetime.now()).total_seconds() / 3600
        log_message(f"Time remaining: {time_remaining:.1f} hours")

        # Cycle through debugging prompts
        task = debugging_prompts[prompt_index % len(debugging_prompts)]
        prompt_index += 1

        log_message(f"Starting iteration {iteration}: {task[:80]}...")
        success, summary = run_orchestrator_task(task, iteration)

        if success:
            log_message(f"Iteration {iteration} summary: {summary}")
        else:
            log_message(f"Iteration {iteration} failed: {summary}")

        log_message("")
        iteration += 1

        # Small delay between iterations
        time.sleep(5)

    log_message("=" * 80)
    log_message(f"DEBUGGING SESSION COMPLETE - {iteration} iterations completed")
    log_message("=" * 80)

if __name__ == "__main__":
    main()
