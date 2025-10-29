"""
TaskExecutor agent - focused on executing specific coding tasks.

Uses hierarchical context management (Goal → Task → Subtask → Action).
Keeps last N message exchanges to stay focused on current work.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import time
import os

from base_agent import BaseAgent
from context_manager import ContextManager
from agent_config import config
from context_strategies import build_hierarchical_context  # Use FULL version
from workspace_manager import WorkspaceManager
from status_display import StatusDisplay, PerformanceStats
from llm_utils import chat_with_inactivity_timeout
import jetbox_notes
import tools


class TaskExecutorAgent(BaseAgent):
    """
    Agent specialized for executing coding tasks.

    Context strategy: Hierarchical (Goal/Task/Subtask tree)
    Tools: File operations, command execution, task completion markers
    """

    def __init__(
        self,
        workspace: Path,
        goal: str | None = None,
        max_rounds: int = 128,
        model: str = None,
        temperature: float = 0.2,
    ):
        """
        Initialize TaskExecutor with full agent.py features.

        Args:
            workspace: Working directory for task execution
            goal: Optional initial goal to set
            max_rounds: Maximum rounds before giving up
            model: Ollama model to use (default from env or config)
            temperature: LLM temperature
        """
        super().__init__(
            name="task_executor",
            role="Code task executor",
            workspace=workspace,
            config=config,
        )

        # Model configuration
        self.model = model or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        self.temperature = temperature
        self.max_rounds = max_rounds

        # Initialize context manager
        self.init_context_manager()
        self.context_manager = self.context_manager or ContextManager()

        # Workspace manager (initialized when goal is set)
        self.workspace_manager = None

        # Status display (initialized when goal is set)
        self.status_display = None

        # Performance tracking
        self.init_perf_stats()

        # Set initial goal if provided
        if goal:
            self.set_goal(goal)

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tools available to TaskExecutor.

        Tools:
        - write_file: Write content to a file
        - read_file: Read file contents
        - list_dir: List directory contents
        - run_cmd: Execute shell commands (whitelisted)
        - mark_subtask_complete: Mark current subtask as done
        - decompose_task: Break task into subtasks
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates parent dirs if needed)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to workspace"},
                            "content": {"type": "string", "description": "File content to write"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to workspace"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List directory contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path (default: workspace root)"},
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_cmd",
                    "description": "Run shell command (only python, pytest, ruff, pip allowed)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to execute"},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_subtask_complete",
                    "description": "Mark current subtask as complete (success or failure)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "description": "True if subtask completed successfully"},
                            "reason": {"type": "string", "description": "Reason for failure (if success=False)"},
                        },
                        "required": ["success"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "decompose_task",
                    "description": "Break current task into smaller subtasks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtasks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of subtask descriptions",
                            },
                        },
                        "required": ["subtasks"],
                    },
                },
            },
        ]

    def get_system_prompt(self) -> str:
        """Return system prompt from config."""
        return config.llm.system_prompt

    def get_context_strategy(self) -> str:
        """TaskExecutor uses hierarchical context management."""
        return "hierarchical"

    def dispatch_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch tool calls with context_manager injection.

        Overrides BaseAgent.dispatch_tool to inject context_manager
        for tools that need it (mark_subtask_complete, decompose_task).
        """
        import tools

        # Get tool name and args
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"].copy()

        # Tools that need context_manager injection
        tools_needing_context = {"mark_subtask_complete", "decompose_task"}

        # Inject context_manager if needed
        if tool_name in tools_needing_context:
            args["context_manager"] = self.context_manager

        # Map tool names to functions
        tool_map = {
            "list_dir": tools.list_dir,
            "read_file": tools.read_file,
            "grep_file": tools.grep_file,
            "write_file": tools.write_file,
            "run_cmd": tools.run_cmd,
            "start_server": tools.start_server,
            "stop_server": tools.stop_server,
            "check_server": tools.check_server,
            "list_servers": tools.list_servers,
            "mark_subtask_complete": tools.mark_subtask_complete,
            "decompose_task": tools.decompose_task,
        }

        # Execute the tool
        if tool_name in tool_map:
            result = tool_map[tool_name](**args)
            return {"result": result}
        else:
            return {"result": {"status": "error", "message": f"Unknown tool: {tool_name}"}}

    def build_context(self) -> list[dict[str, Any]]:
        """
        Build context using FULL hierarchical strategy.

        Returns:
            [system_prompt, task_info, probe_state, jetbox_notes, last_N_messages]
        """
        # Use FULL hierarchical context strategy (includes probe, notes, loops)
        return build_hierarchical_context(
            context_manager=self.context_manager,
            messages=self.state.messages,
            system_prompt=self.get_system_prompt(),
            config=self.config,
            probe_state_func=self._probe_state if hasattr(self, '_probe_state') else None,
            workspace=self.workspace_manager.workspace_dir if self.workspace_manager else None,
        )

    def execute_round(self, model: str, temperature: float) -> dict[str, Any]:
        """
        Execute one round of task execution.

        Returns:
            LLM response with tool calls
        """
        self.increment_round()
        response = self.call_llm(model, temperature)

        # Add assistant message to history
        if "message" in response:
            self.add_message(response["message"])

        return response

    def set_goal(self, goal: str, additional_context: str = "") -> None:
        """
        Set a new goal and initialize all subsystems.

        Args:
            goal: Goal description
            additional_context: Optional additional context for decomposition
        """
        # Initialize context manager with goal
        self.context_manager.load_or_init(goal)

        # Initialize workspace
        goal_slug = goal.lower()[:50].replace(" ", "-").replace("/", "-")
        self.init_workspace_manager(goal_slug)

        # Configure tools with workspace
        tools.set_workspace(self.workspace_manager)
        tools.set_ledger(self.workspace_manager.workspace_dir / "agent_ledger.log")

        # Initialize jetbox notes (pass WorkspaceManager object)
        jetbox_notes.set_workspace(self.workspace_manager)
        jetbox_notes.set_llm_caller(self._llm_caller_for_jetbox)

        # Load existing notes
        existing_notes = jetbox_notes.load_jetbox_notes()
        if existing_notes:
            print(f"[jetbox] Loaded notes: {len(existing_notes)} chars")

        # Initialize status display (reset stats for new goal)
        self.status_display = StatusDisplay(ctx=self.context_manager, reset_stats=True)

    def _llm_caller_for_jetbox(self, messages, temperature=0.2, timeout=30):
        """LLM caller for jetbox notes."""
        return chat_with_inactivity_timeout(
            model=self.model,
            messages=messages,
            options={"temperature": temperature},
            inactivity_timeout=timeout,
        )

    def get_current_status(self) -> dict[str, Any]:
        """
        Get current task execution status.

        Returns:
            Dict with goal, task, subtask info
        """
        goal = self.context_manager.state.goal
        if not goal:
            return {"status": "idle", "goal": None}

        task = self.context_manager._get_current_task()
        subtask = task.active_subtask() if task else None

        return {
            "status": "active",
            "goal": goal.description,
            "task": task.description if task else None,
            "subtask": subtask.description if subtask else None,
            "rounds": self.state.total_rounds,
        }

    def run(self, max_rounds: int = None) -> dict[str, Any]:
        """
        Main execution loop - replaces agent.py main().

        Returns:
            Result dict with status, message, etc.
        """
        max_rounds = max_rounds or self.max_rounds
        messages = []  # Local message stack (cleared on subtask transitions)

        for round_no in range(1, max_rounds + 1):
            # Show status
            if self.status_display:
                # Get current subtask rounds (with defensive checks)
                subtask_rounds = 0
                try:
                    if (self.context_manager
                        and self.context_manager.state.goal
                        and self.context_manager.state.goal.tasks
                        and self.context_manager.state.current_task_idx is not None
                        and self.context_manager.state.current_task_idx < len(self.context_manager.state.goal.tasks)):

                        task = self.context_manager.state.goal.tasks[self.context_manager.state.current_task_idx]
                        if task.active_subtask():
                            subtask_rounds = task.active_subtask().rounds_spent
                except (AttributeError, IndexError, TypeError):
                    # If anything goes wrong, just use 0
                    subtask_rounds = 0

                # Render and print status
                try:
                    status_output = self.status_display.render(
                        round_no=round_no,
                        subtask_rounds=subtask_rounds,
                        max_rounds=self.config.rounds.max_per_subtask
                    )
                    print(status_output)
                except Exception as e:
                    # If status display fails, don't crash the agent
                    print(f"[status_display] Error rendering status: {e}")

            # Build context
            context = self.build_context()

            # Call LLM
            start_time = time.time()
            response = chat_with_inactivity_timeout(
                model=self.model,
                messages=context,
                tools=self.get_tools(),
                options={"temperature": self.temperature},
            )
            duration = time.time() - start_time

            # Track performance
            if self.status_display:
                self.status_display.record_llm_call(duration, response.get("eval_count", 0))

            # Add assistant message
            if "message" in response:
                msg = response["message"]
                messages.append(msg)
                self.add_message(msg)

                # Execute tool calls
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        result = self.dispatch_tool(tool_call)

                        # Record action in stats
                        if self.status_display:
                            success = not (isinstance(result, dict) and result.get("error"))
                            self.status_display.record_action(success)

                        # Unwrap result
                        actual_result = result.get("result") if isinstance(result, dict) and "result" in result else result

                        # CRITICAL: Clear messages on subtask transitions
                        if isinstance(actual_result, dict) and actual_result.get("status") in ["subtask_advanced", "task_advanced"]:
                            old_count = len(messages)
                            messages.clear()
                            print(f"[context_isolation] Cleared {old_count} messages after subtask transition")

                        # Check for goal completion
                        if isinstance(actual_result, dict) and actual_result.get("status") == "goal_complete":
                            self._handle_goal_success()
                            return {"status": "success", "goal": self.context_manager.state.goal.description}

            # Increment rounds
            self.increment_round()

        # Max rounds reached
        goal_desc = self.context_manager.state.goal.description if self.context_manager.state.goal else "unknown"
        self._handle_goal_failure(goal_desc, "Max rounds exceeded")
        return {"status": "failure", "reason": "Max rounds exceeded"}

    def _handle_goal_success(self) -> None:
        """Handle goal success with jetbox notes."""
        goal_summary = jetbox_notes.prompt_for_goal_summary(
            goal_description=self.context_manager.state.goal.description,
            success=True,
        )
        jetbox_notes.append_to_jetbox_notes(goal_summary, section="goal_success")

        print("\n" + "="*70)
        print("GOAL COMPLETE - SUMMARY")
        print("="*70)
        print(goal_summary)
        print("="*70)

    def _handle_goal_failure(self, goal: str, reason: str) -> None:
        """Handle goal failure with jetbox notes."""
        goal_summary = jetbox_notes.prompt_for_goal_summary(
            goal_description=goal,
            success=False,
            reason=reason,
        )
        jetbox_notes.append_to_jetbox_notes(goal_summary, section="goal_failure")

        print("\n" + "="*70)
        print("GOAL FAILED - SUMMARY")
        print("="*70)
        print(goal_summary)
        print("="*70)

    def _probe_state(self) -> dict[str, Any]:
        """
        Probe current filesystem and tool state.

        Returns:
            Dict with files_exist, recent_errors, warning, etc.
        """
        # Simplified probe state - just check workspace files
        if not self.workspace_manager:
            return {}

        workspace_dir = self.workspace_manager.workspace_dir
        files = list(workspace_dir.glob("**/*.py"))

        return {
            "files_exist": [str(f.relative_to(workspace_dir)) for f in files[:10]],
        }
