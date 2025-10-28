"""
Test that servers persist across TaskExecutor invocations.

This simulates:
1. Task 1: Create web app and start server
2. Task 2: Add feature (server still running)
3. Orchestrator exit: Stop all servers
"""
import time
import json
from pathlib import Path
from server_manager import ServerManager


def simulate_task_execution(task_num: int, manager: ServerManager, workspace: Path):
    """Simulate a TaskExecutor running and using servers."""
    print(f"\n{'='*60}")
    print(f"TASK {task_num} EXECUTION")
    print('='*60)

    context_dir = workspace / ".agent_context"
    request_file = context_dir / "server_requests.jsonl"
    response_file = context_dir / "server_responses.jsonl"

    if task_num == 1:
        print("Task 1: Create web app and start server")

        # Simulate agent creating files
        print("  - Creating index.html...")
        webapp_dir = workspace / ".agent_workspace" / "my-webapp"
        webapp_dir.mkdir(parents=True, exist_ok=True)

        index_file = webapp_dir / "index.html"
        index_file.write_text("<h1>Hello World</h1>")

        # Simulate agent starting server (using python sleep to simulate long-running process)
        print("  - Starting long-running server...")
        request = {
            "action": "start",
            "server_id": "webapp",
            "cmd": ["python", "-c", "import time; print('Server started'); time.sleep(300)"],
            "cwd": str(webapp_dir),
            "log_file": str(webapp_dir / "server.log")
        }

        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        # Wait for response (with timeout)
        timeout_sec = 5.0
        start = time.time()
        response = None
        while time.time() - start < timeout_sec:
            if response_file.exists():
                with open(response_file, 'r') as f:
                    lines = f.readlines()
                if lines:
                    response = json.loads(lines[-1].strip())
                    break
            time.sleep(0.1)

        assert response is not None, "Timeout waiting for server start response"
        assert response.get("success"), f"Failed to start server: {response}"
        print(f"  ✓ Server started (PID {response['pid']})")

    elif task_num == 2:
        print("Task 2: Add contact page")

        # Simulate agent creating another file
        print("  - Creating contact.html...")
        webapp_dir = workspace / ".agent_workspace" / "my-webapp"
        contact_file = webapp_dir / "contact.html"
        contact_file.write_text("<h1>Contact Us</h1>")

        # Simulate agent checking if server is still running
        print("  - Checking if server is still running...")
        request = {
            "action": "check",
            "server_id": "webapp",
            "tail_lines": 5
        }

        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        # Wait for response (with timeout)
        timeout_sec = 5.0
        start = time.time()
        response = None
        while time.time() - start < timeout_sec:
            if response_file.exists():
                with open(response_file, 'r') as f:
                    lines = f.readlines()
                if lines:
                    response = json.loads(lines[-1].strip())
                    break
            time.sleep(0.1)

        assert response is not None, "Timeout waiting for server check response"
        assert response.get("success"), f"Failed to check server: {response}"
        assert response["status"] == "running", f"Server not running: {response}"
        print(f"  ✓ Server is still running (uptime: {response['uptime_seconds']}s)")


def test_server_persistence_workflow():
    """Test full workflow with server persistence."""
    print("Testing server persistence across task executions...")

    workspace = Path.cwd()
    context_dir = workspace / ".agent_context"
    context_dir.mkdir(exist_ok=True)

    request_file = context_dir / "server_requests.jsonl"
    response_file = context_dir / "server_responses.jsonl"

    # Start orchestrator with ServerManager
    print("\n" + "="*60)
    print("ORCHESTRATOR STARTUP")
    print("="*60)

    manager = ServerManager(workspace)
    manager.start_monitoring()
    print("✓ ServerManager initialized and monitoring")

    try:
        # Simulate Task 1
        manager.cleanup_old_requests()
        simulate_task_execution(1, manager, workspace)
        print("✓ Task 1 completed")

        # Verify server is still running
        servers = manager.list_servers()
        assert servers["count"] == 1, "Server should still be running after Task 1"
        print(f"✓ Server persisting ({servers['count']} server(s) running)")

        # Simulate Task 2 (server should still be running)
        manager.cleanup_old_requests()
        simulate_task_execution(2, manager, workspace)
        print("✓ Task 2 completed")

        # Verify server is STILL running
        servers = manager.list_servers()
        assert servers["count"] == 1, "Server should still be running after Task 2"
        print(f"✓ Server persisted across tasks ({servers['count']} server(s) running)")

        # Simulate orchestrator shutdown
        print("\n" + "="*60)
        print("ORCHESTRATOR SHUTDOWN")
        print("="*60)

        print("Stopping all servers...")
        manager.stop_all_servers()

        # Verify all servers stopped
        servers = manager.list_servers()
        assert servers["count"] == 0, f"Expected 0 servers, got {servers['count']}"
        print("✓ All servers stopped")

        print("\n✓ Server persistence test passed!")
        print("\nSummary:")
        print("  - Task 1: Started server")
        print("  - Server persisted between tasks")
        print("  - Task 2: Used existing server")
        print("  - Orchestrator exit: Stopped all servers")

    finally:
        # Cleanup
        print("\nCleaning up test files...")
        manager.stop_monitoring()

        # Clean up files
        for f in [request_file, response_file]:
            if f.exists():
                f.unlink()

        # Clean up webapp directory
        webapp_dir = workspace / ".agent_workspace" / "my-webapp"
        if webapp_dir.exists():
            import shutil
            shutil.rmtree(webapp_dir)


if __name__ == "__main__":
    test_server_persistence_workflow()
