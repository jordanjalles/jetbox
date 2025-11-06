#!/usr/bin/env python3
"""
Short version of iterative improvement test - runs for 15 minutes with 3 iterations.

Good for testing the loop before running the full 8-hour version.
"""
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

DURATION_MINUTES = 15
LOG_FILE = Path("iterative_improvement_short_log.txt")

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
        task,
        "--once"
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per task
            cwd="/workspace"
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            log_message(f"✅ Iteration {iteration} completed in {elapsed:.1f}s")
            output = result.stdout + result.stderr
            if "Summary:" in output:
                summary_start = output.rfind("Summary:")
                summary = output[summary_start:summary_start+150].split('\n')[0]
            else:
                summary = "(no summary found)"
            return True, summary
        else:
            log_message(f"❌ Iteration {iteration} failed")
            return False, f"Failed: {result.returncode}"

    except subprocess.TimeoutExpired:
        log_message(f"⏱️ Iteration {iteration} timed out")
        return False, "Timeout"
    except Exception as e:
        log_message(f"💥 Iteration {iteration} crashed: {e}")
        return False, f"Exception: {e}"

def main():
    """Short test loop."""
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=DURATION_MINUTES)

    log_message("=" * 80)
    log_message("SHORT ITERATIVE IMPROVEMENT TEST - 15 MINUTES")
    log_message("=" * 80)
    log_message(f"Start: {start_time.strftime('%H:%M:%S')}")
    log_message(f"End: {end_time.strftime('%H:%M:%S')}")
    log_message("")

    # Iteration 0: Build
    log_message("ITERATION 0: Initial Build")
    task = "Create a simple fractal renderer web app with Mandelbrot set, canvas rendering, and zoom controls"
    success, summary = run_orchestrator_task(task, 0)
    log_message(f"Summary: {summary}\n")

    if not success:
        log_message("Initial build failed. Stopping.")
        return

    # Iteration 1: First improvement
    if datetime.now() < end_time:
        log_message("ITERATION 1: Add improvements")
        task = "Improve the fractal app: add color controls and performance optimizations"
        success, summary = run_orchestrator_task(task, 1)
        log_message(f"Summary: {summary}\n")

    # Iteration 2: Second improvement
    if datetime.now() < end_time:
        log_message("ITERATION 2: Add more features")
        task = "Further improve the fractal app: add Julia set support or animation features"
        success, summary = run_orchestrator_task(task, 2)
        log_message(f"Summary: {summary}\n")

    log_message("=" * 80)
    log_message("SHORT TEST COMPLETE")
    log_message("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_message("\n\nTest interrupted by user")

