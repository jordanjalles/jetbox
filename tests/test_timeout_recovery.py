#!/usr/bin/env python3
"""Test the timeout recovery implementation."""

import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_stress_tests import TESTS, run_test

# Run a simple test to verify the timeout recovery logic works
test = {
    "id": "TEST-1",
    "level": 1,
    "name": "Health Check Test",
    "task": "Write a hello world script",
    "expected_files": ["hello_world.py"],
    "timeout": 60,
}

print("Running test with timeout recovery enabled...")
print("="*70)

result = run_test(test)

print("\n" + "="*70)
print("RESULT:")
print(f"  Success: {result['success']}")
print(f"  Failure mode: {result['failure_mode']}")
print(f"  Ollama restarts: {result['ollama_restarts']}")
print(f"  Duration: {result['duration']:.1f}s")
print(f"  Error: {result['error']}")

if result['ollama_restarts'] > 0:
    print(f"\n✓ Timeout recovery triggered {result['ollama_restarts']} times")
else:
    print("\n✓ No Ollama timeouts occurred")

if result['success']:
    print("✓ Test PASSED")
elif result['failure_mode'] == 'ollama_timeout_repeated':
    print("✗ Test failed after 5 Ollama timeout retries")
elif result['failure_mode'] == 'ollama_unavailable':
    print("✗ Test failed - Ollama not available")
else:
    print(f"✗ Test failed - {result['failure_mode']}")
