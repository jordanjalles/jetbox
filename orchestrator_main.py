"""
Main entry point for the multi-agent system.

Launches the Orchestrator agent which handles user conversation
and delegates coding tasks to TaskExecutor.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

# Ensure UTF-8 encoding on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent_registry import AgentRegistry
from agent_config import config
from server_manager import ServerManager


def _get_workspace_info(task_description: str) -> dict | None:
    """
    Determine workspace location and files created.

    Args:
        task_description: The task that was executed

    Returns:
        Dict with 'workspace' and 'files' keys, or None if not found
    """
    import re

    # Create workspace slug from task description (matches workspace_manager.py logic)
    slug = re.sub(r'[^a-z0-9]+', '-', task_description.lower())
    slug = slug.strip('-')[:60]

    workspace_path = Path.cwd() / ".agent_workspace" / slug

    if not workspace_path.exists():
        return None

    # List all files in workspace (excluding directories and hidden files)
    try:
        files = []
        for item in workspace_path.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                files.append(item.name)

        return {
            "workspace": str(workspace_path),
            "files": sorted(files),
        }
    except Exception:
        return None


def main():
    """Main entry point - launches Orchestrator."""

    # Parse command line args
    exit_after_initial = False
    args = sys.argv[1:]

    # Check for --once flag
    if "--once" in args:
        exit_after_initial = True
        args.remove("--once")

    if args:
        initial_message = " ".join(args)
    else:
        initial_message = None

    # Setup workspace
    workspace = Path.cwd()

    # Initialize agent registry
    registry = AgentRegistry(config_path="agents.yaml", workspace=workspace)

    # Get orchestrator agent
    orchestrator = registry.get_agent("orchestrator")

    # Initialize ServerManager
    server_manager = ServerManager(workspace)
    server_manager.start_monitoring()

    print("=" * 60)
    print("JETBOX ORCHESTRATOR")
    print("=" * 60)
    print()
    print("Type your request, or 'quit' to exit.")
    print()

    try:
        # If initial message provided, process it
        if initial_message:
            print(f"User: {initial_message}")
            print()

            # Clean up old server requests before task
            server_manager.cleanup_old_requests()

            orchestrator.add_user_message(initial_message)

            # Keep executing rounds until no more tool calls
            max_rounds = 10
            for round_num in range(max_rounds):
                # Execute orchestrator round
                response = orchestrator.execute_round(
                    model=config.llm.model,
                    temperature=config.llm.temperature,
                )

                # Display response
                if "message" in response:
                    msg = response["message"]

                    # Show content if present
                    if msg.get("content"):
                        print(f"Orchestrator: {msg['content']}")
                        print()

                    # Execute tool calls (show important ones)
                    if "tool_calls" in msg:
                        tool_results = []
                        for tc in msg["tool_calls"]:
                            tool_name = tc["function"]["name"]
                            args = tc["function"]["arguments"]

                            # Show clarification questions
                            if tool_name == "clarify_with_user":
                                question = args.get("question", "")
                                print(f"Orchestrator: {question}\n")

                            # Show delegation events
                            elif tool_name == "delegate_to_executor":
                                task_desc = args.get("task_description", "")
                                print(f"→ Delegating to TaskExecutor: {task_desc[:60]}...\n")

                            result = execute_orchestrator_tool(tc, registry, server_manager)
                            tool_results.append(result)

                        # Add tool results to conversation
                        orchestrator.add_message({
                            "role": "tool",
                            "content": str(tool_results),
                        })
                    else:
                        # No more tool calls, task is complete
                        break

            # Exit if --once flag was provided
            if exit_after_initial:
                print("\nTask completed. Exiting...")
                return

        # Interactive loop (only if not --once)
        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nShutting down...")
                    break

                # Clean up old server requests before each task
                server_manager.cleanup_old_requests()

                # Add user message
                orchestrator.add_user_message(user_input)

                # Keep executing rounds until no more tool calls
                # This allows orchestrator to:
                # 1. Think/analyze user request
                # 2. Call find_workspace if needed
                # 3. Then delegate_to_executor
                max_rounds = 10
                for round_num in range(max_rounds):
                    # Execute round
                    response = orchestrator.execute_round(
                        model=config.llm.model,
                        temperature=config.llm.temperature,
                    )

                    # Display response
                    if "message" in response:
                        msg = response["message"]

                        # Show content first if present
                        if msg.get("content"):
                            print(f"\nOrchestrator: {msg['content']}\n")

                        # Execute tool calls (show important ones)
                        if "tool_calls" in msg:
                            tool_results = []
                            for tc in msg["tool_calls"]:
                                tool_name = tc["function"]["name"]
                                args = tc["function"]["arguments"]

                                # Show clarification questions
                                if tool_name == "clarify_with_user":
                                    question = args.get("question", "")
                                    print(f"\nOrchestrator: {question}\n")

                                # Show delegation events
                                elif tool_name == "delegate_to_executor":
                                    task_desc = args.get("task_description", "")
                                    print(f"\n→ Delegating to TaskExecutor: {task_desc[:60]}...\n")

                                result = execute_orchestrator_tool(tc, registry, server_manager)
                                tool_results.append(result)

                            # Add tool results to conversation
                            orchestrator.add_message({
                                "role": "tool",
                                "content": str(tool_results),
                            })
                        else:
                            # No more tool calls, orchestrator is done
                            break

            except KeyboardInterrupt:
                print("\n\nInterrupted. Shutting down...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

    finally:
        # Clean shutdown
        print("\n[Orchestrator] Stopping all servers...")
        server_manager.stop_all_servers()
        server_manager.stop_monitoring()
        print("Goodbye!")


def execute_orchestrator_tool(
    tool_call: dict,
    registry: AgentRegistry,
    server_manager: ServerManager = None,
) -> dict:
    """
    Execute an orchestrator tool call.

    Args:
        tool_call: Tool call dict with function name and args
        registry: AgentRegistry instance

    Returns:
        Result dict
    """
    tool_name = tool_call["function"]["name"]
    args = tool_call["function"]["arguments"]

    if tool_name == "delegate_to_executor":
        # Delegate to TaskExecutor and run it
        task_description = args.get("task_description", "")
        context = args.get("context", "")
        workspace_mode = args.get("workspace_mode", "")
        workspace_path = args.get("workspace_path", "")

        # Validate workspace_mode parameter
        if not workspace_mode:
            return {
                "success": False,
                "message": "ERROR: workspace_mode parameter is REQUIRED. Must be 'new' or 'existing'."
            }

        if workspace_mode not in ["new", "existing"]:
            return {
                "success": False,
                "message": f"ERROR: workspace_mode must be 'new' or 'existing', got: {workspace_mode}"
            }

        # Validate workspace_path based on mode
        if workspace_mode == "existing":
            if not workspace_path:
                return {
                    "success": False,
                    "message": "ERROR: workspace_path is REQUIRED when workspace_mode='existing'. Use find_workspace tool first to get the path."
                }
            # Verify the workspace exists
            if not Path(workspace_path).exists():
                return {
                    "success": False,
                    "message": f"ERROR: workspace_path does not exist: {workspace_path}. Use find_workspace to get a valid path."
                }
        elif workspace_mode == "new":
            if workspace_path:
                return {
                    "success": False,
                    "message": "ERROR: workspace_path should NOT be provided when workspace_mode='new'. Remove workspace_path parameter."
                }

        # For backward compatibility, map workspace_mode to workspace parameter
        workspace = workspace_path if workspace_mode == "existing" else ""

        try:
            # Set up the task
            result = registry.delegate_task(
                from_agent="orchestrator",
                to_agent="task_executor",
                task_description=task_description,
                context=context,
                workspace=workspace,
            )

            if not result.get("success"):
                return result

            # Now actually RUN the task executor using the existing agent.py
            print("\n" + "=" * 60)
            print("TASK EXECUTOR RUNNING")
            print("=" * 60 + "\n")

            # Run the existing agent.py as a subprocess
            import subprocess
            import sys

            # Build command with optional workspace and context parameters
            cmd = [sys.executable, "agent.py"]
            if workspace:
                cmd.extend(["--workspace", workspace])
                print(f"[orchestrator] Using existing workspace: {workspace}\n")
            if context:
                cmd.extend(["--context", context])
                print("[orchestrator] Additional context provided\n")
            cmd.append(task_description)

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=False,  # Show output in real-time
                    text=True,
                    timeout=600,  # 10 minute timeout
                )

                print("\n" + "=" * 60)
                print("TASK EXECUTOR COMPLETED")
                print("=" * 60 + "\n")

                # Read messages from TaskExecutor if any
                messages_from_executor = []
                msg_file = Path(".agent_context/messages_to_orchestrator.jsonl")
                if msg_file.exists():
                    try:
                        with open(msg_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    messages_from_executor.append(json.loads(line))
                        # Clear the file after reading
                        msg_file.unlink()
                    except Exception as e:
                        print(f"[orchestrator] Warning: Failed to read executor messages: {e}")

                # Display messages from executor
                if messages_from_executor:
                    print("Messages from TaskExecutor:")
                    for msg in messages_from_executor:
                        severity = msg.get("severity", "info").upper()
                        content = msg.get("message", "")
                        print(f"  [{severity}] {content}")
                    print()

                # Verify actual task completion by checking state.json
                # Exit code 0 means success, 1 means failure/incomplete
                task_completed = proc.returncode == 0

                # Double-check with state.json to ensure tasks were actually completed
                if proc.returncode == 0:
                    state_file = Path(".agent_context/state.json")
                    if state_file.exists():
                        try:
                            with open(state_file, encoding="utf-8") as f:
                                state = json.load(f)
                                # Verify all tasks are marked completed
                                if state.get("goal", {}).get("tasks"):
                                    all_completed = all(
                                        t.get("status") == "completed"
                                        for t in state["goal"]["tasks"]
                                    )
                                    if not all_completed:
                                        # Exit code was 0 but tasks not completed - this shouldn't happen
                                        # but handle it defensively
                                        task_completed = False
                                        print("[orchestrator] Warning: Exit code 0 but not all tasks completed in state.json")
                        except Exception as e:
                            print(f"[orchestrator] Warning: Could not verify state.json: {e}")

                if task_completed:
                    # Try to determine workspace location and files created
                    workspace_info = _get_workspace_info(task_description)

                    result_msg = "Task execution completed successfully"
                    if workspace_info:
                        result_msg += f"\n\nWorkspace: {workspace_info['workspace']}"
                        if workspace_info.get('files'):
                            result_msg += f"\nFiles created: {', '.join(workspace_info['files'])}"

                    # Include executor messages in result
                    if messages_from_executor:
                        result_msg += "\n\nTaskExecutor Messages:"
                        for msg in messages_from_executor:
                            result_msg += f"\n  [{msg.get('severity', 'info')}] {msg.get('message', '')}"

                    return {
                        "success": True,
                        "message": result_msg,
                        "workspace": workspace_info.get('workspace') if workspace_info else None,
                        "files": workspace_info.get('files') if workspace_info else [],
                        "executor_messages": messages_from_executor,
                    }
                else:
                    error_msg = f"Task execution failed (exit code {proc.returncode})"
                    if messages_from_executor:
                        error_msg += "\n\nTaskExecutor Messages:"
                        for msg in messages_from_executor:
                            error_msg += f"\n  [{msg.get('severity', 'info')}] {msg.get('message', '')}"

                    return {
                        "success": False,
                        "message": error_msg,
                        "executor_messages": messages_from_executor,
                    }

            except subprocess.TimeoutExpired:
                return {"success": False, "message": "Task execution timed out"}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Execution failed: {e}"}

    elif tool_name == "clarify_with_user":
        # Question already displayed in assistant message content
        # Just acknowledge internally
        return {"success": True, "message": "Question posed to user"}

    elif tool_name == "create_task_plan":
        # Plan creation acknowledged internally
        tasks = args.get("tasks", [])
        return {"success": True, "message": f"Plan created with {len(tasks)} tasks"}

    elif tool_name == "get_executor_status":
        # Get TaskExecutor status
        try:
            status = registry.get_agent_status("task_executor")
            return {"success": True, "status": status}
        except Exception as e:
            return {"success": False, "message": f"Could not get status: {e}"}

    elif tool_name == "list_workspaces":
        # List all existing workspaces
        try:
            workspace_dir = Path.cwd() / ".agent_workspace"
            if not workspace_dir.exists():
                return {"success": True, "workspaces": [], "message": "No workspaces found"}

            workspaces = []
            for item in workspace_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Get file count and list files
                    files = []
                    for f in item.iterdir():
                        if f.is_file() and not f.name.startswith('.'):
                            files.append(f.name)

                    workspaces.append({
                        "name": item.name,
                        "path": str(item),
                        "files": sorted(files),
                        "file_count": len(files),
                    })

            # Sort by most recently modified
            workspaces.sort(key=lambda x: Path(x["path"]).stat().st_mtime, reverse=True)

            msg = f"Found {len(workspaces)} workspace(s):\n"
            for ws in workspaces:
                msg += f"\n- {ws['name']}/\n"
                msg += f"  Path: {ws['path']}\n"
                msg += f"  Files ({ws['file_count']}): {', '.join(ws['files'][:5])}"
                if ws['file_count'] > 5:
                    msg += f" ... and {ws['file_count'] - 5} more"
                msg += "\n"

            return {"success": True, "workspaces": workspaces, "message": msg}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Could not list workspaces: {e}"}

    elif tool_name == "find_workspace":
        # Find best matching workspace for a project name
        project_name = args.get("project_name", "").lower()

        try:
            workspace_dir = Path.cwd() / ".agent_workspace"
            if not workspace_dir.exists():
                return {
                    "success": False,
                    "message": f"No workspaces found. Cannot find workspace for '{project_name}'."
                }

            # Get all workspaces
            workspaces = []
            for item in workspace_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    workspaces.append({
                        "name": item.name,
                        "path": str(item),
                        "modified": item.stat().st_mtime,
                    })

            if not workspaces:
                return {
                    "success": False,
                    "message": f"No workspaces found. Cannot find workspace for '{project_name}'."
                }

            # Score each workspace by how well it matches project_name
            def score_match(workspace_name: str, query: str) -> int:
                """Score how well a workspace name matches a query. Higher is better."""
                ws_lower = workspace_name.lower()
                query_lower = query.lower()

                # Exact match
                if query_lower in ws_lower:
                    # Bonus for matching at word boundaries
                    words = ws_lower.split('-')
                    for word in words:
                        if word == query_lower:
                            return 100  # Exact word match
                        if word.startswith(query_lower):
                            return 80  # Word starts with query
                    return 60  # Contains query

                # Fuzzy match - check if all characters appear in order
                query_idx = 0
                for char in ws_lower:
                    if query_idx < len(query_lower) and char == query_lower[query_idx]:
                        query_idx += 1
                if query_idx == len(query_lower):
                    return 30  # All chars present in order

                # Check individual words
                query_words = query_lower.split()
                ws_words = ws_lower.split('-')
                matches = sum(1 for qw in query_words if any(qw in wsw for wsw in ws_words))
                if matches > 0:
                    return 20 * matches

                return 0

            # Score all workspaces
            scored = []
            for ws in workspaces:
                score = score_match(ws["name"], project_name)
                if score > 0:
                    scored.append((score, ws))

            if not scored:
                # No matches - return list of available workspaces
                ws_list = "\n".join(f"  - {ws['name']}" for ws in workspaces[:10])
                return {
                    "success": False,
                    "message": f"No workspace found matching '{project_name}'.\n\nAvailable workspaces:\n{ws_list}"
                }

            # Sort by score (descending), then by recency
            scored.sort(key=lambda x: (x[0], x[1]["modified"]), reverse=True)

            best_match = scored[0][1]
            best_score = scored[0][0]

            # If we have multiple good matches, show them
            other_matches = [ws for score, ws in scored[1:3] if score >= 30]

            msg = f"Found workspace for '{project_name}':\n"
            msg += f"  Best match: {best_match['name']}\n"
            msg += f"  Path: {best_match['path']}\n"

            if other_matches:
                msg += "\nOther possible matches:\n"
                for ws in other_matches:
                    msg += f"  - {ws['name']}\n"

            return {
                "success": True,
                "workspace": best_match["path"],
                "workspace_name": best_match["name"],
                "message": msg,
                "confidence": "high" if best_score >= 60 else "medium" if best_score >= 30 else "low",
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Error finding workspace: {e}"}

    else:
        return {"success": False, "message": f"Unknown tool: {tool_name}"}


if __name__ == "__main__":
    main()
