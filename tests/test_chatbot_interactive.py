#!/usr/bin/env python3
"""
Test chatbot interactive mode with context inspection.

Tests for the bug where chatbot doesn't respond to every query.
"""
import subprocess
import sys
import time

# Test queries - mix of poetry requests and casual conversation
test_queries = [
    "Write me a haiku about coding",
    "Tell me a joke about Python",
    "Write a short poem about debugging",
    "What's your favorite color?",
    "Can you write a limerick about functions?",
    "exit"
]

def test_chatbot():
    """Test chatbot with multiple queries."""
    print("=" * 80)
    print("CHATBOT INTERACTIVE TEST WITH CONTEXT INSPECTION")
    print("=" * 80)
    print()
    print("Starting chatbot with --ContextInspector flag...")
    print()

    # Start chatbot process
    proc = subprocess.Popen(
        [sys.executable, "agent.py", "--team", "chatbot", "--ContextInspector"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        # Give it time to start
        time.sleep(3)

        # Send each query
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"QUERY {i}: {query}")
            print('='*80)

            # Send query
            proc.stdin.write(query + "\n")
            proc.stdin.flush()

            # Read response with timeout
            start = time.time()
            response_lines = []
            while time.time() - start < 15:  # 15 second timeout per query
                if proc.poll() is not None:
                    break

                # Try to read a line
                try:
                    line = proc.stdout.readline()
                    if line:
                        print(line, end='')
                        response_lines.append(line)

                        # Check if we got a response (look for agent name or prompt)
                        if "simple_chatbot:" in line or "You:" in line:
                            start = time.time()  # Reset timeout when we see activity
                except:
                    break

                time.sleep(0.1)

            # Check if we got a response
            if not any("simple_chatbot:" in line for line in response_lines):
                print(f"\n⚠️  WARNING: No response detected for query {i}")
            else:
                print(f"\n✓ Response received for query {i}")

            time.sleep(1)  # Brief pause between queries

        # Wait for exit
        time.sleep(2)

    finally:
        # Clean up
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_chatbot()
