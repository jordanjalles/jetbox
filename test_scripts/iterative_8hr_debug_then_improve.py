#!/usr/bin/env python3
"""
8-Hour Session: 4 hours debugging to ensure webapp works, then 4 hours improvements.
"""
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configuration
DURATION_HOURS = 8
WORKSPACE = Path(".agent_workspaces/fractal-webapp-project")
LOG_FILE = Path("iterative_8hr_debug_improve.log")

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
            timeout=600,
            cwd="/workspace"
        )

        elapsed = time.time() - start_time

        # Save debug output
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
            return False, f"Exit code: {result.returncode}"

    except subprocess.TimeoutExpired:
        log_message(f"⏱ Iteration {iteration} timed out")
        return False, "Timeout"
    except Exception as e:
        log_message(f"💥 Iteration {iteration} crashed: {e}")
        return False, f"Exception: {e}"

def main():
    """Main 8-hour loop."""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=DURATION_HOURS)
    midpoint = start_time + timedelta(hours=DURATION_HOURS / 2)

    log_message("=" * 80)
    log_message("8-HOUR SESSION: DEBUG THEN IMPROVE")
    log_message("=" * 80)
    log_message(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Midpoint: {midpoint.strftime('%Y-%m-%d %H:%M:%S')} (switch to improvements)")
    log_message(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Workspace: {WORKSPACE}")
    log_message("")

    if not WORKSPACE.exists():
        log_message(f"Creating workspace: {WORKSPACE}")
    else:
        log_message(f"Workspace exists with {len(list(WORKSPACE.glob('*')))} items")

    iteration = 0

    # Initial task: Create basic working webapp
    initial_task = """Create a working Mandelbrot fractal webapp with index.html, script.js, style.css.

Requirements:
- HTML must have a canvas element with an ID
- Script must get the canvas, get 2D context, and draw Mandelbrot fractal pixels
- Canvas should be visible when HTML is opened
- Actual pixels must be drawn (not blank canvas)"""

    success, summary = run_orchestrator_task(initial_task, iteration)
    log_message(f"Initial build: {summary}")
    log_message("")

    iteration += 1

    # Phase 1: Debugging (first half)
    debugging_tasks = [
        "Read all webapp files. Verify HTML loads script, script gets canvas element by ID, getContext('2d') is called, and Mandelbrot pixels are drawn. Fix bugs.",

        "Test the Mandelbrot calculation: verify complex number math is correct (z = z² + c), iteration count is used, escape radius is 4, and colors map to iterations.",

        "Check canvas initialization: verify canvas.width and canvas.height are set, ImageData is created with correct size, and putImageData is called.",

        "Debug rendering: add console.log to verify draw function runs, iterations complete, and pixels are set. Fix any issues preventing rendering.",

        "Verify file connections: HTML <script src=\"script.js\">, script document.getElementById matches canvas id, DOMContentLoaded event fires correctly.",

        "Test the complete flow: page load → script runs → canvas accessed → Mandelbrot calculated → pixels drawn. Fix broken steps.",

        "Check for JavaScript errors: verify no undefined variables, no null canvas, proper iteration bounds, valid color calculations.",

        "Make the fractal visible and correct: verify zoom/pan values show interesting part of Mandelbrot set, colors are visible (not all black/white).",
    ]

    # Phase 2: Improvements (second half)
    improvement_tasks = [
        "Add mouse click zoom: click to zoom in at that point.",

        "Add pan controls: drag to pan around the fractal.",

        "Add zoom controls: buttons or mouse wheel to zoom in/out.",

        "Improve colors: use smooth color gradients instead of simple iteration count.",

        "Add Julia set: allow switching between Mandelbrot and Julia fractals.",

        "Add UI controls: sliders for max iterations, zoom level display, reset button.",

        "Optimize rendering: use Web Worker or requestAnimationFrame for smooth updates.",

        "Add save image feature: button to download current fractal as PNG.",

        "Improve visual design: better colors, smooth gradients, nice UI layout.",

        "Add more fractal types: Burning Ship, Tricorn, Newton fractals.",
    ]

    debugging = True
    task_index = 0

    while datetime.now() < end_time:
        time_remaining = (end_time - datetime.now()).total_seconds() / 3600

        # Switch from debugging to improvements at midpoint
        if debugging and datetime.now() >= midpoint:
            debugging = False
            task_index = 0
            log_message("=" * 80)
            log_message("SWITCHING TO IMPROVEMENT PHASE")
            log_message("=" * 80)
            log_message("")

        log_message(f"Time remaining: {time_remaining:.1f} hours")
        log_message(f"Phase: {'DEBUGGING' if debugging else 'IMPROVEMENTS'}")

        # Select task based on phase
        if debugging:
            task = debugging_tasks[task_index % len(debugging_tasks)]
        else:
            task = improvement_tasks[task_index % len(improvement_tasks)]

        task_index += 1

        success, summary = run_orchestrator_task(task, iteration)

        if success:
            log_message(f"Iteration {iteration} summary: {summary}")
        else:
            log_message(f"Iteration {iteration} failed: {summary}")

        log_message("")
        iteration += 1

        time.sleep(5)

    log_message("=" * 80)
    log_message(f"SESSION COMPLETE - {iteration} iterations")
    log_message("=" * 80)

if __name__ == "__main__":
    main()
