"""
Test ServerManager functionality.
"""
import time
from pathlib import Path
from server_manager import ServerManager


def test_server_lifecycle():
    """Test starting, checking, and stopping a server."""
    print("Testing ServerManager...")

    workspace = Path.cwd()
    manager = ServerManager(workspace)
    manager.start_monitoring()

    try:
        # Test 1: Start a simple Python HTTP server
        print("\n1. Starting HTTP server...")
        result = manager.start_server(
            server_id="test_server",
            cmd=["python", "-m", "http.server", "8765"],
            cwd=str(workspace),
            log_file=str(workspace / "test_server.log")
        )
        print(f"   Result: {result}")
        assert result.get("success"), f"Failed to start server: {result}"
        assert result["server_id"] == "test_server"
        pid = result["pid"]
        print(f"   ✓ Server started with PID {pid}")

        # Give it a moment to fully start
        time.sleep(1)

        # Test 2: Check server status
        print("\n2. Checking server status...")
        status = manager.check_server("test_server", tail_lines=5)
        print(f"   Result: {status}")
        assert status.get("success"), f"Failed to check server: {status}"
        assert status["status"] == "running", f"Server not running: {status}"
        print(f"   ✓ Server is running")

        # Test 3: List servers
        print("\n3. Listing servers...")
        servers = manager.list_servers()
        print(f"   Result: {servers}")
        assert servers.get("success"), f"Failed to list servers: {servers}"
        assert servers["count"] == 1, f"Expected 1 server, got {servers['count']}"
        print(f"   ✓ Found {servers['count']} server(s)")

        # Test 4: Try to start duplicate server
        print("\n4. Trying to start duplicate server...")
        dup_result = manager.start_server(
            server_id="test_server",
            cmd=["python", "-m", "http.server", "8766"],
            cwd=str(workspace),
            log_file=str(workspace / "test_server2.log")
        )
        print(f"   Result: {dup_result}")
        assert "error" in dup_result, "Should have failed with duplicate server ID"
        print(f"   ✓ Duplicate server prevented: {dup_result['error']}")

        # Test 5: Stop server
        print("\n5. Stopping server...")
        stop_result = manager.stop_server("test_server")
        print(f"   Result: {stop_result}")
        assert stop_result.get("success"), f"Failed to stop server: {stop_result}"
        print(f"   ✓ Server stopped: {stop_result['message']}")

        # Verify server is gone
        time.sleep(0.5)
        servers_after = manager.list_servers()
        assert servers_after["count"] == 0, f"Expected 0 servers, got {servers_after['count']}"
        print(f"   ✓ Server list is empty")

        print("\n✓ All tests passed!")

    finally:
        # Cleanup
        print("\nCleaning up...")
        manager.stop_all_servers()
        manager.stop_monitoring()

        # Remove test log file
        log_file = workspace / "test_server.log"
        if log_file.exists():
            log_file.unlink()


if __name__ == "__main__":
    test_server_lifecycle()
