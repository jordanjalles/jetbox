"""
Test agent server tools with ServerManager integration.

This simulates the TaskExecutor -> Orchestrator communication flow.
"""
import time
import json
from pathlib import Path
from server_manager import ServerManager


def test_server_request_response_flow():
    """Test the JSONL request/response flow between agent and orchestrator."""
    print("Testing agent <-> orchestrator server communication...")

    workspace = Path.cwd()
    context_dir = workspace / ".agent_context"
    context_dir.mkdir(exist_ok=True)

    request_file = context_dir / "server_requests.jsonl"
    response_file = context_dir / "server_responses.jsonl"

    # Clean up old files
    if request_file.exists():
        request_file.unlink()
    if response_file.exists():
        response_file.unlink()

    # Start ServerManager (simulates orchestrator)
    manager = ServerManager(workspace)
    manager.start_monitoring()

    try:
        # Test 1: Agent writes start_server request
        print("\n1. Agent requests server start...")
        request = {
            "action": "start",
            "server_id": "webapp",
            "cmd": ["python", "-m", "http.server", "8777"],
            "cwd": str(workspace),
            "log_file": str(workspace / "webapp.log")
        }

        with open(request_file, 'a') as f:
            f.write(json.dumps(request) + '\n')

        print("   Request written to server_requests.jsonl")

        # Wait for orchestrator to process (simulate _wait_for_server_response)
        timeout = 5.0
        start_time = time.time()
        response = None

        while time.time() - start_time < timeout:
            if response_file.exists():
                with open(response_file, 'r') as f:
                    lines = f.readlines()
                if lines:
                    response = json.loads(lines[-1].strip())
                    break
            time.sleep(0.1)

        assert response is not None, "No response received from orchestrator"
        print(f"   Response: {response}")
        assert response.get("success"), f"Server start failed: {response}"
        print(f"   ✓ Server started (PID {response['pid']})")

        # Test 2: Agent checks server status
        print("\n2. Agent checks server status...")
        check_request = {
            "action": "check",
            "server_id": "webapp",
            "tail_lines": 5
        }

        with open(request_file, 'a') as f:
            f.write(json.dumps(check_request) + '\n')

        # Wait for response
        existing_lines = 1  # We already read one response
        time.sleep(0.5)

        with open(response_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) > existing_lines, "No check response received"
        check_response = json.loads(lines[-1].strip())
        print(f"   Response: {check_response}")
        assert check_response.get("success"), f"Check failed: {check_response}"
        assert check_response["status"] == "running", f"Server not running: {check_response}"
        print(f"   ✓ Server is running, uptime: {check_response['uptime_seconds']}s")

        # Test 3: Agent lists servers
        print("\n3. Agent lists servers...")
        list_request = {"action": "list"}

        with open(request_file, 'a') as f:
            f.write(json.dumps(list_request) + '\n')

        time.sleep(0.5)

        with open(response_file, 'r') as f:
            lines = f.readlines()

        list_response = json.loads(lines[-1].strip())
        print(f"   Response: {list_response}")
        assert list_response.get("success"), f"List failed: {list_response}"
        assert list_response["count"] == 1, f"Expected 1 server, got {list_response['count']}"
        print(f"   ✓ Found {list_response['count']} server(s)")

        # Test 4: Agent stops server
        print("\n4. Agent requests server stop...")
        stop_request = {
            "action": "stop",
            "server_id": "webapp"
        }

        with open(request_file, 'a') as f:
            f.write(json.dumps(stop_request) + '\n')

        time.sleep(0.5)

        with open(response_file, 'r') as f:
            lines = f.readlines()

        stop_response = json.loads(lines[-1].strip())
        print(f"   Response: {stop_response}")
        assert stop_response.get("success"), f"Stop failed: {stop_response}"
        print(f"   ✓ {stop_response['message']}")

        print("\n✓ All integration tests passed!")

    finally:
        # Cleanup
        print("\nCleaning up...")
        manager.stop_all_servers()
        manager.stop_monitoring()

        # Clean up test files
        for f in [request_file, response_file, workspace / "webapp.log"]:
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    test_server_request_response_flow()
