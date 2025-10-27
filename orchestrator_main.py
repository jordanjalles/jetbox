"""
Main entry point for the multi-agent system.

Launches the Orchestrator agent which handles user conversation
and delegates coding tasks to TaskExecutor.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Ensure UTF-8 encoding on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent_registry import AgentRegistry
from agent_config import config


def _get_workspace_info(task_description: str) -> dict | None:
    """
    Determine workspace location and files created.

    Args:
        task_description: The task that was executed

    Returns:
        Dict with 'workspace' and 'files' keys, or None if not found
    """
    import os
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
    if len(sys.argv) > 1:
        initial_message = " ".join(sys.argv[1:])
    else:
        initial_message = None

    # Setup workspace
    workspace = Path.cwd()

    # Initialize agent registry
    registry = AgentRegistry(config_path="agents.yaml", workspace=workspace)

    # Get orchestrator agent
    orchestrator = registry.get_agent("orchestrator")

    print("=" * 60)
    print("JETBOX ORCHESTRATOR")
    print("=" * 60)
    print()
    print("Type your request, or 'quit' to exit.")
    print()

    # If initial message provided, process it
    if initial_message:
        print(f"User: {initial_message}")
        print()
        orchestrator.add_user_message(initial_message)

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

                    result = execute_orchestrator_tool(tc, registry)
                    tool_results.append(result)

                # Add tool results to conversation
                orchestrator.add_message({
                    "role": "tool",
                    "content": str(tool_results),
                })

    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            # Add user message
            orchestrator.add_user_message(user_input)

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

                        result = execute_orchestrator_tool(tc, registry)
                        tool_results.append(result)

                    # Add tool results to conversation
                    orchestrator.add_message({
                        "role": "tool",
                        "content": str(tool_results),
                    })

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def execute_orchestrator_tool(
    tool_call: dict,
    registry: AgentRegistry,
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
        workspace = args.get("workspace", "")

        try:
            # Set up the task
            result = registry.delegate_task(
                from_agent="orchestrator",
                to_agent="task_executor",
                task_description=task_description,
                context=context,
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

            # Build command with optional workspace parameter
            cmd = [sys.executable, "agent.py"]
            if workspace:
                cmd.extend(["--workspace", workspace])
                print(f"[orchestrator] Using existing workspace: {workspace}\n")
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

                if proc.returncode == 0:
                    # Try to determine workspace location and files created
                    workspace_info = _get_workspace_info(task_description)

                    result_msg = "Task execution completed successfully"
                    if workspace_info:
                        result_msg += f"\n\nWorkspace: {workspace_info['workspace']}"
                        if workspace_info.get('files'):
                            result_msg += f"\nFiles created: {', '.join(workspace_info['files'])}"

                    return {
                        "success": True,
                        "message": result_msg,
                        "workspace": workspace_info.get('workspace') if workspace_info else None,
                        "files": workspace_info.get('files') if workspace_info else [],
                    }
                else:
                    return {"success": False, "message": f"Task execution failed with code {proc.returncode}"}

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

    else:
        return {"success": False, "message": f"Unknown tool: {tool_name}"}


if __name__ == "__main__":
    main()
