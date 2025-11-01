"""
Safe Test Runner - Launch tests with proper Ollama checks and process tracking.

This script safely launches test suites by:
1. Checking if Ollama is ready
2. Clearing old contexts
3. Tracking subprocess PID
4. Streaming output in real-time
5. Cleaning up on completion or interruption

Usage:
    python safe_test_runner.py <test_script.py>

Example:
    python safe_test_runner.py run_three_level_eval.py
"""
import os
import sys
import subprocess
import time
from pathlib import Path
from process_tracker import ProcessTracker
from ollama_manager import OllamaManager


def safe_run_test_suite(script_path: str, description: str):
    """
    Safely run a test suite with proper cleanup.

    Args:
        script_path: Path to test script (e.g., "run_three_level_eval.py")
        description: Human-readable description

    Returns:
        int: Exit code from test subprocess
    """
    # Step 1: Check if Ollama is ready
    print("="*70)
    print(f"SAFE TEST RUNNER: {description}")
    print("="*70)

    print("\n[1/4] Checking Ollama status...")
    if OllamaManager.is_ollama_busy():
        print("⚠️  Ollama is busy")
        if not OllamaManager.wait_for_ollama(timeout=30):
            print("⚠️  Ollama still busy after 30s - clearing contexts")
            OllamaManager.clear_all_contexts()
            time.sleep(5)
    else:
        print("✓ Ollama is idle")

    # Step 2: Clear any old contexts
    print("\n[2/4] Clearing Ollama contexts...")
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    OllamaManager.clear_all_contexts(model)
    print(f"✓ Cleared contexts for {model}")

    # Step 3: Launch test subprocess
    print(f"\n[3/4] Launching test: {script_path}")

    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )

    # Track the process
    ProcessTracker.register_process(
        process.pid,
        "test_suite",
        description
    )

    print(f"✓ Test running as PID {process.pid}")
    print("  (To stop: python stop_tests.py or Ctrl+C)")

    # Step 4: Monitor output
    print("\n[4/4] Test output:")
    print("-"*70)

    try:
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')

        # Wait for completion
        process.wait()

        print("-"*70)
        if process.returncode == 0:
            print(f"\n✓ Test completed successfully")
        else:
            print(f"\n⚠️  Test completed with exit code {process.returncode}")

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user (Ctrl+C)")
        print("Terminating test process...")
        process.terminate()
        try:
            process.wait(timeout=10)
            print("✓ Test process terminated")
        except subprocess.TimeoutExpired:
            print("⚠️  Test process did not terminate, killing...")
            process.kill()
            process.wait()
            print("✓ Test process killed")

    except Exception as e:
        print(f"\n⚠️  Error during test execution: {e}")
        process.terminate()
        process.wait(timeout=10)

    finally:
        # Always unregister
        ProcessTracker.unregister_process(process.pid)
        print(f"✓ Cleaned up PID {process.pid}")

    return process.returncode


def main():
    if len(sys.argv) < 2:
        print("Usage: python safe_test_runner.py <test_script.py>")
        print("\nExample:")
        print("  python safe_test_runner.py run_three_level_eval.py")
        sys.exit(1)

    script = sys.argv[1]

    # Check if script exists
    if not Path(script).exists():
        print(f"❌ Error: Test script not found: {script}")
        sys.exit(1)

    desc = f"Test: {Path(script).stem}"

    exit_code = safe_run_test_suite(script, desc)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
