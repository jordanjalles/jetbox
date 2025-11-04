"""
Orchestrator agent - manages user conversation and delegates to other agents.

This agent is purely config-driven. All logic is in base_agent.py and behaviors.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json

from base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    """
    Agent specialized for user interaction and task delegation.

    This is a minimal wrapper around BaseAgent that passes config to BaseAgent.__init__().
    Everything (role, system_prompt, behaviors) is loaded from orchestrator_config.yaml.

    All logic is in base_agent.py or behaviors. NO unique logic here.
    """

    def __init__(self, workspace: Path | None = None, exclude_behaviors: list[str] | None = None):
        """
        Initialize Orchestrator agent.

        Args:
            workspace: Working directory (defaults to .agent_workspaces)
            exclude_behaviors: List of behavior names to exclude (e.g., ["ChatbotBehavior"])
        """
        super().__init__(
            name="orchestrator",
            workspace=workspace or Path(".agent_workspaces"),
            config_file="orchestrator_config.yaml",
            exclude_behaviors=exclude_behaviors,
        )

    # ===========================
    # CLI customization
    # ===========================

    @classmethod
    def create_agent_instance(cls, workspace: Path, args: dict[str, Any]):
        """
        Create orchestrator agent with conditional ChatbotBehavior exclusion.

        Args:
            workspace: Workspace directory path
            args: Parsed CLI arguments

        Returns:
            OrchestratorAgent instance
        """
        initial_message = args["initial_message"]
        force_chat_mode = args["force_chat_mode"]

        # Determine if ChatbotBehavior should be excluded
        # Exclude it when goal string is provided UNLESS --chat flag is set
        # Include it when no goal string (interactive mode) OR --chat flag
        exclude_behaviors = []
        if initial_message and not force_chat_mode:
            # Autonomous mode: exclude chatbot behavior to prevent conversational mode
            exclude_behaviors = ["ChatbotBehavior"]
            print("[OrchestratorAgent] Autonomous mode: ChatbotBehavior excluded")
        else:
            # Interactive mode or chat mode: include chatbot behavior for user interaction
            if force_chat_mode:
                print("[OrchestratorAgent] Chat mode (--chat): ChatbotBehavior enabled")
            else:
                print("[OrchestratorAgent] Interactive mode: ChatbotBehavior enabled")

        return cls(workspace=workspace, exclude_behaviors=exclude_behaviors)

    @classmethod
    def run_agent(cls, agent: BaseAgent, args: dict[str, Any]) -> None:
        """
        Execute orchestrator with ServerManager and ChatbotBehavior support.

        Args:
            agent: OrchestratorAgent instance
            args: Parsed CLI arguments
        """
        from server_manager import ServerManager
        from agent_registry import AgentRegistry

        initial_message = args["initial_message"]
        exit_after_initial = args["exit_after_initial"]

        # Initialize ServerManager
        server_manager = ServerManager(agent.workspace)
        server_manager.start_monitoring()

        # Initialize agent registry
        registry = AgentRegistry(config_path="agents.yaml", workspace=agent.workspace)

        # Get ChatbotBehavior instance for task completion detection
        chatbot_behavior = None
        for behavior in agent.behaviors:
            if behavior.get_name() == "chatbot":
                chatbot_behavior = behavior
                break

        # Define task execution callback for ChatbotBehavior
        def execute_task(user_message: str) -> None:
            """
            Execute a single orchestrator task.

            This function is called by ChatbotBehavior for each user message.
            It runs the orchestrator's LLM loop until the task completes.
            """
            # Clean up old server requests
            server_manager.cleanup_old_requests()

            # Add user message to history
            agent.add_message({"role": "user", "content": user_message})

            # Reset task_complete_flag for new task
            if chatbot_behavior:
                chatbot_behavior.task_complete_flag = False
                chatbot_behavior.consecutive_empty_rounds = 0

            # Execute rounds until task complete
            round_num = 0
            max_rounds = 100

            while True:
                round_num += 1

                # Check if ChatbotBehavior detected task completion (2 consecutive empty rounds)
                if chatbot_behavior and chatbot_behavior.task_complete_flag:
                    print("[orchestrator] Task complete (detected by ChatbotBehavior), returning to prompt")
                    break

                response = agent._execute_round(
                    round_no=round_num,
                    max_rounds=max_rounds,
                    model=agent.config.llm.model,
                    temperature=agent.config.llm.temperature,
                )

                # Check if goal completed/failed
                if response is None:
                    continue

                # Display response
                if "message" in response:
                    msg = response["message"]

                    if isinstance(msg, dict):
                        if msg.get("content"):
                            print(f"Orchestrator: {msg['content']}")
                            print()
                    elif isinstance(msg, str):
                        if msg:
                            print(f"Orchestrator: {msg}")
                            print()

                    # Execute tool calls
                    if isinstance(msg, dict) and "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            tool_name = tc["function"]["name"]
                            tool_args = tc["function"]["arguments"]

                            # Show delegation events
                            if tool_name == "clarify_with_user":
                                print(f"Orchestrator: {tool_args.get('question', '')}\n")
                            elif tool_name == "consult_architect":
                                print(f"→ Consulting Architect: {tool_args.get('project_description', '')[:60]}...\n")
                            elif tool_name == "delegate_to_executor":
                                print(f"→ Delegating to TaskExecutor: {tool_args.get('task_description', '')[:60]}...\n")

                            result = cls.execute_orchestrator_tool(tc, registry, server_manager, agent)

                            # Add tool result
                            agent.add_message({
                                "role": "tool",
                                "content": json.dumps(result),
                            })

                # Check max rounds
                if round_num >= max_rounds:
                    print("[orchestrator] Max rounds reached")
                    break

        try:
            # Use ChatbotBehavior's multi-task chat loop if available
            if chatbot_behavior and not exit_after_initial:
                # Multi-task chat mode
                chatbot_behavior.run_multi_task_chat_loop(
                    agent=agent,
                    execute_task_callback=execute_task,
                    initial_message=initial_message
                )
            elif initial_message and exit_after_initial:
                # Single task mode (--once flag)
                print(f"User: {initial_message}\n")
                execute_task(initial_message)
                print("\nTask completed. Exiting...")
            else:
                # Fallback to manual loop if ChatbotBehavior not available
                print("Warning: ChatbotBehavior not found, using fallback loop")
                if initial_message:
                    print(f"User: {initial_message}\n")
                    execute_task(initial_message)
                    if exit_after_initial:
                        print("\nTask completed. Exiting...")
                        return
                    print("\n✅ Task completed. Ready for next request.\n")

                while True:
                    try:
                        user_input = input("You: ").strip()
                        if not user_input:
                            continue
                        if user_input.lower() in ["quit", "exit", "q"]:
                            print("\nShutting down...")
                            break
                        execute_task(user_input)
                        print("\n✅ Task completed. Ready for next request.\n")
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

    @staticmethod
    def execute_orchestrator_tool(
        tool_call: dict,
        registry,  # AgentRegistry
        server_manager = None,  # ServerManager
        orchestrator_agent = None,  # OrchestratorAgent
    ) -> dict:
        """
        Execute an orchestrator tool call.

        Args:
            tool_call: Tool call dict with function name and args
            registry: AgentRegistry instance
            server_manager: ServerManager instance (optional)
            orchestrator_agent: OrchestratorAgent instance (optional)

        Returns:
            Result dict
        """
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        if tool_name == "consult_architect":
            # Consult architect for complex project design
            project_description = args.get("project_description", "")
            requirements = args.get("requirements", "")
            constraints = args.get("constraints", "")
            workspace_path = args.get("workspace_path", "")

            if not project_description:
                return {
                    "success": False,
                    "message": "ERROR: project_description is REQUIRED for architect consultation"
                }

            print("\n" + "=" * 60)
            print("ARCHITECT CONSULTATION")
            print("=" * 60)
            print(f"Project: {project_description}")
            if requirements:
                print(f"Requirements: {requirements[:100]}...")
            if constraints:
                print(f"Constraints: {constraints[:100]}...")
            print("=" * 60 + "\n")

            try:
                # Get architect agent from registry
                architect = registry.get_agent("architect")

                # Determine workspace (use provided or create new from project description)
                if workspace_path:
                    workspace = Path(workspace_path)
                else:
                    # Create new workspace for this project
                    import re
                    slug = re.sub(r'[^\w\s-]', '', project_description.lower())
                    slug = re.sub(r'[-\s]+', '-', slug)[:50]
                    workspace = Path(f".agent_workspace/{slug}")
                    workspace.mkdir(parents=True, exist_ok=True)

                # Configure architect with workspace (updates tools)
                architect.configure_workspace(workspace)

                # Run architect consultation
                result = architect.consult(
                    project_description=project_description,
                    requirements=requirements,
                    constraints=constraints,
                    max_rounds=10
                )

                if result["status"] == "success":
                    artifacts = result["artifacts"]

                    # Build detailed message about artifacts
                    message_parts = [
                        "\n✅ Architecture consultation complete!\n",
                        f"Workspace: {workspace}\n",
                    ]

                    if artifacts["docs"]:
                        message_parts.append(f"\nArchitecture documents ({len(artifacts['docs'])}):")
                        for doc in artifacts["docs"]:
                            message_parts.append(f"  - {doc}")

                    if artifacts["modules"]:
                        message_parts.append(f"\nModule specifications ({len(artifacts['modules'])}):")
                        for module in artifacts["modules"]:
                            message_parts.append(f"  - {module}")

                    # Read and include actual task list
                    task_list = None
                    if artifacts["task_breakdown"]:
                        message_parts.append(f"\nTask breakdown: {artifacts['task_breakdown']}")
                        task_file = workspace / artifacts["task_breakdown"]
                        if task_file.exists():
                            with open(task_file) as f:
                                task_data = json.load(f)
                            task_list = task_data.get("tasks", [])
                            message_parts.append(f"  ({task_data['total_tasks']} tasks ready for delegation)")

                            # Include task summary in message
                            message_parts.append("\nTasks to delegate:")
                            for task in task_list[:5]:  # Show first 5 tasks
                                deps = f" (depends on: {', '.join(task['dependencies'])})" if task.get('dependencies') else ""
                                message_parts.append(f"  [{task['id']}] {task['description']}{deps}")
                            if len(task_list) > 5:
                                message_parts.append(f"  ... and {len(task_list) - 5} more tasks")

                    message_parts.append("\n" + "=" * 60)

                    # Add task management enhancement to orchestrator if task breakdown exists
                    if orchestrator_agent and task_list:
                        orchestrator_agent.add_task_management(workspace)
                        print(f"[orchestrator] Added task management enhancement ({len(task_list)} tasks)")

                    return {
                        "success": True,
                        "message": "\n".join(message_parts),
                        "artifacts": artifacts,
                        "workspace": str(workspace),
                        "tasks": task_list,  # Include actual task list
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Architect consultation incomplete: {result.get('message', 'unknown error')}"
                    }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "message": f"Architect consultation failed: {e}"
                }

        elif tool_name == "delegate_to_executor":
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
                        workspace_info = OrchestratorAgent._get_workspace_info(task_description)

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

        # Check if this is a task management tool (if enhancement is active)
        elif tool_name in ["read_task_breakdown", "get_next_task", "mark_task_status", "update_task"]:
            # Dispatch to task management behavior
            from behaviors import TaskManagementBehavior

            # Get workspace manager from orchestrator if available
            workspace_manager = None
            if orchestrator_agent and hasattr(orchestrator_agent, 'workspace_manager'):
                workspace_manager = orchestrator_agent.workspace_manager

            # Create behavior instance
            task_mgmt = TaskManagementBehavior(workspace_manager=workspace_manager)

            try:
                result = task_mgmt.dispatch_tool(tool_name, args)
                return result
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"status": "error", "message": f"Task management tool failed: {e}"}

        else:
            return {"success": False, "message": f"Unknown tool: {tool_name}"}

    @staticmethod
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

        workspace_path = Path.cwd() / ".agent_workspaces" / slug

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


if __name__ == "__main__":
    OrchestratorAgent.main()
