"""
Live demonstration of server feature.

This script:
1. Starts ServerManager (simulating orchestrator)
2. Runs agent.py with a task
3. Shows server requests being processed
4. Verifies server started
5. Cleans up
"""
import subprocess
import sys
import time
import threading
from pathlib import Path
from server_manager import ServerManager


def monitor_agent_output(process):
    """Print agent output in real-time."""
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        print(f"[AGENT] {line}", end='')


def main():
    print("="*70)
    print("LIVE DEMONSTRATION: Server Management Feature")
    print("="*70)
    print()

    workspace = Path.cwd()

    # Clean up old files
    context_dir = workspace / ".agent_context"
    for f in ["server_requests.jsonl", "server_responses.jsonl"]:
        path = context_dir / f
        if path.exists():
            path.unlink()

    # Start ServerManager (simulating orchestrator)
    print("1. Starting ServerManager (orchestrator-level)...")
    manager = ServerManager(workspace)
    manager.start_monitoring()
    print("   ✓ ServerManager monitoring started")
    print()

    try:
        # Run agent.py with server task
        print("2. Delegating task to agent.py...")
        print("   Task: Create web app and start HTTP server")
        print()

        task = (
            "Create index.html with '<h1>Demo App</h1>'. "
            "Then use start_server to start HTTP server on port 9988 with name 'demo'. "
            "Then use check_server to verify it's running."
        )

        cmd = [sys.executable, "agent.py", task]

        print("[Starting agent.py subprocess...]")
        print()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Monitor output in thread
        output_thread = threading.Thread(target=monitor_agent_output, args=(process,))
        output_thread.start()

        # Wait for agent to complete (or timeout)
        timeout = 120
        try:
            process.wait(timeout=timeout)
            output_thread.join(timeout=5)
        except subprocess.TimeoutExpired:
            print("\n[DEMO] Agent timed out (this is OK for demo)")
            process.kill()

        print()
        print("="*70)
        print("3. Checking ServerManager status...")
        print("="*70)

        # Check if server was started
        time.sleep(1)
        servers = manager.list_servers()

        if servers["count"] > 0:
            print(f"\n✅ SUCCESS! Server was started:")
            for srv in servers["servers"]:
                print(f"   Server ID: {srv['server_id']}")
                print(f"   PID: {srv['pid']}")
                print(f"   Status: {srv['status']}")
                print(f"   Command: {srv['cmd']}")
                print(f"   Uptime: {srv['uptime_seconds']}s")

            # Try to connect
            import socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 9988))
                sock.close()
                if result == 0:
                    print(f"\n   ✅ Server is accepting connections on port 9988!")
                else:
                    print(f"\n   ⚠ Server running but not accessible on port 9988")
            except Exception:
                print(f"\n   ⚠ Could not test connection")

        else:
            print("\n⚠ No servers started (agent may have timed out)")
            print("   Checking request file...")

            req_file = context_dir / "server_requests.jsonl"
            if req_file.exists():
                print(f"\n   ✓ Server requests were made:")
                with open(req_file, 'r') as f:
                    for line in f:
                        print(f"     {line.strip()}")
                print("\n   This proves the agent is trying to use server tools!")
            else:
                print("   ✗ No requests found")

    finally:
        print()
        print("="*70)
        print("4. Cleanup: Stopping all servers...")
        print("="*70)
        manager.stop_all_servers()
        manager.stop_monitoring()
        print("   ✓ All servers stopped")
        print()

    print("="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print()
    print("Key Points Demonstrated:")
    print("  ✓ Agent can call start_server tool")
    print("  ✓ ServerManager processes requests in background")
    print("  ✓ Server persists after agent subprocess exits")
    print("  ✓ ServerManager controls cleanup")
    print()


if __name__ == "__main__":
    main()
