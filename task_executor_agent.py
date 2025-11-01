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
from status_display import StatusDisplay
from llm_utils import chat_with_inactivity_timeout, clear_ollama_context
from completion_detector import analyze_llm_response
import jetbox_notes
import tools
import json

# Legacy strategy support (only imported when use_behaviors=False)
# Import moved inside __init__ to avoid unnecessary imports when using behaviors
try:
    from context_strategies import ContextStrategy
except ImportError:
    ContextStrategy = None  # Strategies not available


class TaskExecutorAgent(BaseAgent):
    """
    Agent specialized for executing coding tasks.

    Context strategy: Hierarchical (Goal/Task/Subtask tree)
    Tools: File operations, command execution, task completion markers
    """

    def __init__(
        self,
        workspace: Path | str | None = None,
        goal: str | None = None,
        max_rounds: int = 128,
        model: str = None,
        temperature: float = 0.2,
        context_strategy: "ContextStrategy | None" = None,
        timeout: int | None = None,
        use_behaviors: bool = True,  # Default to True (behavior system is now preferred)
        config_file: str = "task_executor_config.yaml",
    ):
        """
        Initialize TaskExecutor with full agent.py features.

        Args:
            workspace: Workspace directory for task execution
                      - If None: create NEW isolated workspace under .agent_workspace/
                      - If Path: REUSE existing workspace (no nesting)
            goal: Optional initial goal to set
            max_rounds: Maximum rounds before giving up
            model: Ollama model to use (default from env or config)
            temperature: LLM temperature
            context_strategy: Context management strategy (defaults to HierarchicalStrategy)
            timeout: Optional timeout override in seconds (defaults to config value)
            use_behaviors: If True, load behaviors from config instead of using strategies
            config_file: Path to behavior config YAML (only used if use_behaviors=True)
        """
        # Determine base workspace for BaseAgent (for .agent_context storage)
        base_workspace = Path(workspace) if workspace else Path(".")

        super().__init__(
            name="task_executor",
            role="Code task executor",
            workspace=base_workspace,
            config=config,
        )

        # Phase 4: Behavior system support
        self.use_behaviors = use_behaviors

        # Model configuration
        self.model = model or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        self.temperature = temperature
        self.max_rounds = max_rounds

        # Store workspace parameter for later use
        # None = create new, Path = reuse existing
        self.workspace = Path(workspace) if workspace else None

        # Context strategy (only used when use_behaviors=False)
        # Default to SubAgentStrategy for delegated work when strategies are used
        self.context_strategy = None
        self.enhancements = []

        if not self.use_behaviors:
            # Legacy strategy mode: Load strategies
            from context_strategies import SubAgentStrategy
            self.context_strategy = context_strategy or SubAgentStrategy()
            print(f"[task_executor] Using legacy strategy mode: {self.context_strategy.get_name()}")

        # Initialize context manager
        self.init_context_manager()
        self.context_manager = self.context_manager or ContextManager()

        # Workspace manager (initialized when goal is set)
        self.workspace_manager = None

        # Status display (initialized when goal is set)
        self.status_display = None

        # Performance tracking
        self.init_perf_stats()

        # Wall-clock timeout tracking
        self.goal_start_time = None
        self.timeout_override = timeout  # Optional override

        # Phase 4: Load behaviors if requested
        if self.use_behaviors:
            print(f"[task_executor] Loading behaviors from {config_file}")
            self.load_behaviors_from_config(config_file)

        # Set initial goal if provided
        if goal:
            self.set_goal(goal)

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tools available to TaskExecutor.

        Phase 4: If use_behaviors=True, returns tools from behaviors.
        Otherwise, uses legacy strategy/enhancement system.

        Merges base tools from tools.get_tool_definitions() with
        strategy-specific tools from the context strategy and
        enhancement tools.

        Base tools include:
        - File operations: write_file, read_file, list_dir, grep_file
        - Command execution: run_bash (any shell command)
        - Server management: start_server, stop_server, check_server, list_servers

        Strategy-specific tools (added by context strategy):
        - Hierarchical: mark_subtask_complete, decompose_task
        - Other strategies: custom tools as needed

        Enhancement tools (added by enhancements):
        - JetboxNotes: no tools (auto-generated)
        - TaskManagement: task CRUD operations

        Returns:
            Combined list of all available tools
        """
        # Phase 4: If using behaviors, return behavior tools
        if self.use_behaviors:
            return self.get_behavior_tools()

        # Legacy path: strategy + enhancements
        # Get base tools (file ops, bash, server management)
        base_tools = tools.get_tool_definitions()

        # Filter out hierarchical-specific tools from base (they'll come from strategy)
        hierarchical_tool_names = {"mark_subtask_complete", "decompose_task"}
        filtered_base = [
            tool for tool in base_tools
            if tool.get("function", {}).get("name") not in hierarchical_tool_names
        ]

        # Get strategy-specific tools
        strategy_tools = []
        if self.context_strategy:
            strategy_tools = self.context_strategy.get_strategy_tools()

        # Get enhancement tools
        enhancement_tools = []
        for enhancement in self.enhancements:
            enhancement_tools.extend(enhancement.get_enhancement_tools())

        # Merge: base tools + strategy tools + enhancement tools
        return filtered_base + strategy_tools + enhancement_tools

    def get_system_prompt(self) -> str:
        """
        Return system prompt with strategy-specific and enhancement instructions injected.

        Phase 4: If use_behaviors=True, includes behavior instructions.

        Combines:
        1. Base system prompt from config (generic coding instructions)
        2. Strategy-specific instructions (workflow, tools, guidelines)
        3. Enhancement instructions (jetbox notes, task management, etc.)
        4. Behavior instructions (if use_behaviors=True)

        Returns:
            Complete system prompt for LLM
        """
        base_prompt = config.llm.system_prompt

        parts = [base_prompt]

        # Phase 4: Add behavior instructions if using behaviors
        if self.use_behaviors:
            behavior_instructions = self.get_behavior_instructions()
            if behavior_instructions:
                parts.append(behavior_instructions)
        else:
            # Legacy path: strategy + enhancements
            # Get strategy-specific instructions
            strategy_instructions = ""
            if self.context_strategy:
                strategy_instructions = self.context_strategy.get_strategy_instructions()

            # Get enhancement instructions
            enhancement_instructions = []
            for enhancement in self.enhancements:
                inst = enhancement.get_enhancement_instructions()
                if inst:
                    enhancement_instructions.append(inst)

            # Combine base + strategy + enhancements
            if strategy_instructions:
                parts.append(strategy_instructions)
            parts.extend(enhancement_instructions)

        return "\n".join(parts)

    def get_context_strategy(self) -> str:
        """TaskExecutor uses hierarchical context management."""
        return "hierarchical"

    def dispatch_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch tool calls with context_manager injection.

        Phase 4: If use_behaviors=True, dispatches to behaviors.
        Otherwise, uses legacy tool dispatch.

        Overrides BaseAgent.dispatch_tool to inject context_manager
        for tools that need it (mark_subtask_complete, decompose_task).
        """
        # Phase 4: If using behaviors, dispatch to behavior system
        if self.use_behaviors:
            return self.dispatch_tool_to_behavior(tool_call)

        # Legacy path: manual tool dispatch
        import tools

        # Get tool name and args
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"].copy()

        # Tools that need context_manager injection
        tools_needing_context = {
            "mark_subtask_complete",
            "mark_goal_complete",
            "mark_complete",
            "mark_failed",
            "decompose_task"
        }

        # Inject context_manager if needed
        if tool_name in tools_needing_context:
            args["context_manager"] = self.context_manager

        # Map tool names to functions
        tool_map = {
            "write_file": tools.write_file,
            "read_file": tools.read_file,
            "list_dir": tools.list_dir,
            "run_bash": tools.run_bash,
            "start_server": tools.start_server,
            "stop_server": tools.stop_server,
            "check_server": tools.check_server,
            "list_servers": tools.list_servers,
            "mark_subtask_complete": tools.mark_subtask_complete,
            "mark_goal_complete": tools.mark_goal_complete,
            "mark_complete": tools.mark_complete,
            "mark_failed": tools.mark_failed,
            "decompose_task": tools.decompose_task,
        }

        # Execute the tool
        if tool_name in tool_map:
            result = tool_map[tool_name](**args)

            # Record action for loop detection (core context management responsibility)
            success = not (isinstance(result, dict) and result.get("error"))
            if self.context_strategy:
                loop_warning = self.context_strategy.record_action(
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    success=success
                )

                # Add loop warning to result if detected
                if loop_warning:
                    print(f"\n⚠️  LOOP DETECTED: {loop_warning['warning']}")
                    print(f"Suggestion: {loop_warning['suggestion']}\n")
                    # Inject warning into result so LLM sees it
                    if isinstance(result, dict):
                        result["_loop_warning"] = f"{loop_warning['warning']}\n{loop_warning['suggestion']}"
                    else:
                        result = {
                            "result": result,
                            "_loop_warning": f"{loop_warning['warning']}\n{loop_warning['suggestion']}"
                        }

            return {"result": result}
        else:
            return {"result": {"status": "error", "message": f"Unknown tool: {tool_name}"}}

    def build_context(self) -> list[dict[str, Any]]:
        """
        Build context using configured strategy + enhancements.

        Phase 4: If use_behaviors=True, uses behavior system.
        Otherwise, uses legacy strategy/enhancement system.

        Uses context_strategy.build_context() which handles compaction automatically,
        then injects enhancement context sections.

        Returns:
            Context list ready for LLM: [system_prompt, ...enhancements..., ...messages...]
        """
        # Phase 4: If using behaviors, delegate to behavior system
        if self.use_behaviors:
            # Build base context
            context = [
                {"role": "system", "content": self.get_system_prompt()},
                *self.state.messages
            ]

            # Let behaviors enhance context
            context = self.enhance_context_with_behaviors(context)

            return context

        # Legacy path: strategy + enhancements
        # Use configured context strategy (default: append-until-full)
        context = self.context_strategy.build_context(
            context_manager=self.context_manager,
            messages=self.state.messages,
            system_prompt=self.get_system_prompt(),
            config=self.config,
            probe_state_func=self._probe_state if hasattr(self, '_probe_state') else None,
            workspace=self.workspace_manager.workspace_dir if self.workspace_manager else None,
        )

        # Inject enhancements after system prompt (index 1)
        enhancement_index = 1
        for enhancement in self.enhancements:
            enhancement_context = enhancement.get_context_injection(
                context_manager=self.context_manager,
                workspace=self.workspace_manager.workspace_dir if self.workspace_manager else None,
            )
            if enhancement_context:
                context.insert(enhancement_index, enhancement_context)
                enhancement_index += 1

        return context

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

        # Initialize workspace manager
        # If self.workspace is None: create NEW isolated workspace
        # If self.workspace is Path: REUSE existing workspace
        goal_slug = goal.lower()[:50].replace(" ", "-").replace("/", "-")

        if self.workspace:
            # Reuse mode: use existing workspace directory
            print(f"[task_executor] Reusing workspace: {self.workspace}")
            self.init_workspace_manager(goal_slug, workspace_path=self.workspace)
        else:
            # Create new mode: create isolated workspace
            print(f"[task_executor] Creating new workspace for goal")
            self.init_workspace_manager(goal_slug, workspace_path=None)

        # Phase 4: If using behaviors, trigger on_goal_start event
        if self.use_behaviors:
            self.trigger_behavior_event(
                "on_goal_start",
                goal=goal,
                workspace=self.workspace,
                context_manager=self.context_manager,
                workspace_manager=self.workspace_manager
            )
        else:
            # Legacy path: manual configuration
            # Configure tools with workspace
            tools.set_workspace(self.workspace_manager)
            tools.set_ledger(self.workspace_manager.workspace_dir / "agent_ledger.log")

            # Initialize jetbox notes system (always set workspace)
            jetbox_notes.set_workspace(self.workspace_manager)
            jetbox_notes.set_llm_caller(self._llm_caller_for_jetbox)

            # Add JetboxNotesEnhancement to enhancements list
            # This will inject notes into context if they exist
            jetbox_enhancement = JetboxNotesEnhancement(workspace_manager=self.workspace_manager)
            self.enhancements.append(jetbox_enhancement)
            print(f"[task_executor] Added JetboxNotesEnhancement")

            # Load existing notes for display (optional)
            existing_notes = jetbox_notes.load_jetbox_notes()
            if existing_notes:
                print(f"[jetbox] Loaded notes: {len(existing_notes)} chars")

        # Initialize status display (reset stats for new goal)
        self.status_display = StatusDisplay(ctx=self.context_manager, reset_stats=True)

        # Start wall-clock timer for goal
        self.goal_start_time = time.time()

    def _llm_caller_for_jetbox(self, messages, temperature=0.2, timeout=30):
        """LLM caller for jetbox notes."""
        return chat_with_inactivity_timeout(
            model=self.model,
            messages=messages,
            options={"temperature": temperature},
            inactivity_timeout=timeout,
        )

    def _handle_timeout(self, elapsed_seconds: float, messages: list) -> dict[str, Any]:
        """
        Handle goal wall-clock timeout.

        Creates jetbox notes summary and context dump for troubleshooting.

        Args:
            elapsed_seconds: Total time elapsed
            messages: Current message history

        Returns:
            Result dict with timeout status
        """
        print(f"[timeout] Handling goal timeout after {elapsed_seconds:.1f}s")

        # Create jetbox notes summary if enabled by config and enhancement present
        jetbox_enabled = any(isinstance(e, JetboxNotesEnhancement) for e in self.enhancements)
        if self.config.timeouts.create_summary_on_timeout and jetbox_enabled:
            try:
                print("[timeout] Creating jetbox notes summary...")
                jetbox_notes.create_timeout_summary(
                    goal=self.context_manager.state.goal,
                    elapsed_seconds=elapsed_seconds,
                    action_history=self.context_manager.action_history,
                )
            except Exception as e:
                print(f"[timeout] Failed to create summary: {e}")

        # Save context dump if enabled
        if self.config.timeouts.save_context_dump:
            try:
                print("[timeout] Saving context dump...")
                self._save_timeout_context_dump(elapsed_seconds, messages)
            except Exception as e:
                print(f"[timeout] Failed to save context dump: {e}")

        # Calculate total rounds from task tree
        total_rounds = 0
        if self.context_manager.state.goal:
            for task in self.context_manager.state.goal.tasks:
                for subtask in task.subtasks:
                    total_rounds += subtask.rounds_used

        # Get actual timeout value used
        max_time = self.timeout_override if self.timeout_override is not None else self.config.timeouts.max_goal_time

        return {
            "status": "timeout",
            "reason": f"Goal wall-clock limit exceeded: {elapsed_seconds:.1f}s > {max_time}s",
            "elapsed_seconds": elapsed_seconds,
            "total_rounds": total_rounds,
        }

    def _save_timeout_context_dump(self, elapsed_seconds: float, messages: list) -> None:
        """
        Save context dump for timeout troubleshooting.

        Args:
            elapsed_seconds: Total elapsed time
            messages: Current message history
        """
        from pathlib import Path
        from datetime import datetime
        import json

        # Create dump directory
        dump_dir = Path(".agent_context/timeout_dumps")
        dump_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dump_file = dump_dir / f"goal_timeout_{timestamp}.json"

        # Calculate context stats
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        estimated_tokens = total_chars // 4

        # Build dump data
        # Calculate total rounds
        total_rounds = 0
        if self.context_manager.state.goal:
            for task in self.context_manager.state.goal.tasks:
                for subtask in task.subtasks:
                    total_rounds += subtask.rounds_used

        # Get actual timeout value used
        max_time = self.timeout_override if self.timeout_override is not None else self.config.timeouts.max_goal_time

        dump_data = {
            "timestamp": timestamp,
            "timeout_type": "goal_wall_clock",
            "elapsed_seconds": round(elapsed_seconds, 2),
            "max_goal_time": max_time,
            "model": self.model,
            "goal": self.context_manager.state.goal.description if self.context_manager.state.goal else None,
            "total_rounds": total_rounds,
            "context_stats": {
                "message_count": len(messages),
                "total_chars": total_chars,
                "estimated_tokens": estimated_tokens,
            },
            "task_tree": self._serialize_task_tree(),
            "messages": messages,
        }

        # Write dump file
        with open(dump_file, "w") as f:
            json.dump(dump_data, f, indent=2, default=str)

        print(f"[timeout] Context saved to {dump_file}")
        print(f"[timeout] Stats: {len(messages)} messages, ~{estimated_tokens:,} tokens, {elapsed_seconds:.1f}s elapsed")

    def _serialize_task_tree(self) -> dict:
        """Serialize current task tree for context dump."""
        if not self.context_manager.state.goal:
            return {}

        goal = self.context_manager.state.goal
        return {
            "goal": goal.description,
            "status": goal.status,
            "tasks": [
                {
                    "description": task.description,
                    "status": task.status,
                    "subtasks": [
                        {
                            "description": subtask.description,
                            "status": subtask.status,
                            "rounds_used": subtask.rounds_used,
                        }
                        for subtask in task.subtasks
                    ]
                }
                for task in goal.tasks
            ]
        }

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

        try:
            for round_no in range(1, max_rounds + 1):
                # Check wall-clock timeout (use override if provided)
                max_time = self.timeout_override if self.timeout_override is not None else self.config.timeouts.max_goal_time
                if self.goal_start_time and max_time > 0:
                    elapsed = time.time() - self.goal_start_time
                    if elapsed > max_time:
                        print(f"\n[timeout] Goal wall-clock limit exceeded: {elapsed:.1f}s > {max_time}s")
                        return self._handle_timeout(elapsed, messages)

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
                                subtask_rounds = task.active_subtask().rounds_used
                    except (AttributeError, IndexError, TypeError):
                        # If anything goes wrong, just use 0
                        subtask_rounds = 0

                    # Render and print status
                    # Show hierarchical display only for hierarchical strategies/behaviors
                    try:
                        show_hierarchical = False
                        if not self.use_behaviors and self.context_strategy:
                            # Legacy mode: check strategy type
                            from context_strategies import HierarchicalStrategy
                            show_hierarchical = isinstance(self.context_strategy, HierarchicalStrategy)
                        elif self.use_behaviors:
                            # Behavior mode: check for HierarchicalContextBehavior
                            show_hierarchical = any(
                                b.get_name() == "hierarchical_context" for b in self._behaviors
                            )

                        status_output = self.status_display.render(
                            round_no=round_no,
                            subtask_rounds=subtask_rounds,
                            max_rounds=self.config.rounds.max_per_subtask,
                            show_hierarchical=show_hierarchical
                        )
                        print(status_output)
                    except Exception as e:
                        # If status display fails, don't crash the agent
                        print(f"[status_display] Error rendering status: {e}")

                # Build context
                context = self.build_context()

                # Call LLM with 3-minute total timeout
                start_time = time.time()
                try:
                    response = chat_with_inactivity_timeout(
                        model=self.model,
                        messages=context,
                        tools=self.get_tools(),
                        options={"temperature": self.temperature},
                        inactivity_timeout=15,     # 15s inactivity = hung Ollama (reduced from 30s)
                        max_total_time=180,        # 3 minutes total = context too large or slow model
                    )
                except Exception as llm_error:
                    # Handle timeout errors for SubAgentStrategy/SubAgentContextBehavior
                    if isinstance(llm_error, TimeoutError):
                        # Check if using subagent context (strategy or behavior)
                        is_subagent_mode = False
                        if not self.use_behaviors and self.context_strategy:
                            from context_strategies import SubAgentStrategy
                            is_subagent_mode = isinstance(self.context_strategy, SubAgentStrategy)
                        elif self.use_behaviors:
                            is_subagent_mode = any(
                                b.get_name() == "subagent_context" for b in self._behaviors
                            )

                        # If using SubAgent mode, nudge agent to report completion/failure
                        if is_subagent_mode:
                            print(f"[timeout_nudge] LLM timeout detected with SubAgentStrategy - nudging agent to report status")

                            # Add a system message nudging completion
                            nudge_message = {
                                "role": "user",
                                "content": """⚠️ TIMEOUT WARNING - You must report task status immediately.

The LLM call timed out. You need to wrap up and report results to the controlling agent NOW.

Based on your current progress:
- If you've completed the core work: call mark_complete(summary="what you accomplished")
- If you cannot complete the task: call mark_failed(reason="why it cannot be completed")

You MUST call one of these tools in your next response. The controlling agent is waiting."""
                            }

                            # Add nudge to messages
                            messages.append(nudge_message)
                            self.add_message(nudge_message)

                            # Try ONE more LLM call with the nudge
                            print("[timeout_nudge] Giving agent one final chance to report status...")
                            try:
                                response = chat_with_inactivity_timeout(
                                    model=self.model,
                                    messages=self.build_context(),
                                    tools=self.get_tools(),
                                    options={"temperature": self.temperature},
                                    inactivity_timeout=30,
                                    max_total_time=60,  # Shorter timeout for final attempt
                                )
                                print("[timeout_nudge] Agent responded to nudge")
                            except Exception as final_error:
                                # Even the nudge failed - force mark_failed
                                print(f"[timeout_nudge] Final nudge attempt failed: {final_error}")
                                print("[timeout_nudge] Forcing mark_failed")

                                # Create synthetic mark_failed response
                                response = {
                                    "message": {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [
                                            {
                                                "function": {
                                                    "name": "mark_failed",
                                                    "arguments": {
                                                        "reason": f"Task timed out after LLM inactivity. Last error: {str(final_error)[:200]}"
                                                    }
                                                }
                                            }
                                        ]
                                    },
                                    "eval_count": 0,
                                    "prompt_eval_count": 0,
                                }
                        else:
                            # Not using SubAgentStrategy, re-raise
                            raise llm_error

                    # Try to recover from tool call parsing errors
                    elif isinstance(llm_error, Exception):
                        from ollama import ResponseError
                        from llm_utils import extract_tool_call_from_parse_error

                        if isinstance(llm_error, ResponseError) and "error parsing tool call" in str(llm_error):
                            # Attempt to extract JSON from error message
                            extracted = extract_tool_call_from_parse_error(str(llm_error))
                            if extracted:
                                print(f"[llm_recovery] Recovered tool call from parse error (LLM generated text before JSON)")

                                # Construct a valid response with the extracted tool call
                                # We need to figure out which tool was being called
                                # Strategy: Look for tool name in the context of recent messages
                                tool_name = None

                                # Check if extracted has name/arguments already
                                if 'name' in extracted and 'arguments' in extracted:
                                    tool_name = extracted['name']
                                    tool_args = extracted['arguments']
                                else:
                                    # Try to infer tool name from extracted keys
                                    # Common patterns: write_file has 'path'+'content', run_bash has 'command'
                                    if 'path' in extracted and 'content' in extracted:
                                        tool_name = 'write_file'
                                        tool_args = extracted
                                    elif 'command' in extracted:
                                        tool_name = 'run_bash'
                                        tool_args = extracted
                                    elif 'reason' in extracted:
                                        tool_name = 'mark_subtask_complete'
                                        tool_args = extracted
                                    else:
                                        # Can't determine tool name, log and re-raise
                                        print(f"[llm_recovery] Could not determine tool name from extracted args: {list(extracted.keys())}")
                                        raise llm_error

                                # Build synthetic response
                                response = {
                                    "message": {
                                        "role": "assistant",
                                        "content": "",  # No content when using tools
                                        "tool_calls": [
                                            {
                                                "function": {
                                                    "name": tool_name,
                                                    "arguments": tool_args
                                                }
                                            }
                                        ]
                                    },
                                    "eval_count": 0,  # Unknown, will be treated as 0
                                    "prompt_eval_count": 0,
                                }
                                print(f"[llm_recovery] Synthetic response created: tool={tool_name}, args_keys={list(tool_args.keys())}")
                            else:
                                # Could not extract - return error to LLM so it can retry
                                print(f"[llm_recovery] Could not extract JSON from parse error")
                                print(f"[llm_recovery] Sending parse error back to LLM for retry")

                                # Return the raw error text to LLM
                                response = {
                                    "message": {
                                        "role": "assistant",
                                        "content": str(llm_error),  # Just the raw error
                                        "tool_calls": []
                                    },
                                    "eval_count": 0,
                                    "prompt_eval_count": 0,
                                }
                                print("[llm_recovery] Agent will continue with error feedback to LLM")
                        else:
                            # Different error type, re-raise
                            raise llm_error
                    else:
                        # Different error type, re-raise
                        raise llm_error

                duration = time.time() - start_time

                # Track performance
                if self.status_display:
                    self.status_display.record_llm_call(
                        duration,
                        response.get("eval_count", 0),
                        response.get("prompt_eval_count", 0)
                    )

                # Add assistant message
                if "message" in response:
                    msg = response["message"]
                    messages.append(msg)
                    self.add_message(msg)

                    # Execute tool calls
                    if "tool_calls" in msg:
                        tool_calls = msg["tool_calls"]

                        # Analyze LLM response for completion signals
                        current_task = self.context_manager._get_current_task()
                        current_subtask = current_task.active_subtask() if current_task else None
                        subtask_desc = current_subtask.description if current_subtask else None
                        analysis = analyze_llm_response(msg.get("content", ""), tool_calls, subtask_desc)

                        for idx, tool_call in enumerate(tool_calls):
                            is_last_call = (idx == len(tool_calls) - 1)

                            result = self.dispatch_tool(tool_call)

                            # Record action in stats
                            if self.status_display:
                                success = not (isinstance(result, dict) and result.get("error"))
                                self.status_display.record_action(success)

                            # Add nudge to last tool result if needed
                            if is_last_call and analysis["should_nudge"]:
                                result_with_nudge = result.copy() if isinstance(result, dict) else {"result": result}
                                result_with_nudge["_nudge"] = analysis["nudge_message"]
                                result = result_with_nudge
                                print(f"[completion_detector] NUDGE: {analysis['reason']}")

                            # Add tool result to messages
                            tool_result_str = json.dumps(result)
                            tool_message = {
                                "role": "tool",
                                "content": tool_result_str,
                            }
                            messages.append(tool_message)  # For context isolation
                            self.add_message(tool_message)  # For LLM visibility (adds to self.state.messages)

                            # Unwrap result
                            actual_result = result.get("result") if isinstance(result, dict) and "result" in result else result

                            # Clear messages on subtask transitions (if strategy requires it)
                            if isinstance(actual_result, dict) and actual_result.get("status") in ["subtask_advanced", "task_advanced"]:
                                if self.context_strategy.should_clear_on_transition():
                                    old_count = len(self.state.messages)
                                    self.clear_messages()  # Clear self.state.messages (used by build_context)
                                    messages.clear()  # Also clear local list for consistency
                                    print(f"[context_isolation] Cleared {old_count} messages after subtask transition (strategy: {self.context_strategy.get_name()})")

                            # Check for goal completion
                            if isinstance(actual_result, dict) and actual_result.get("status") == "goal_complete":
                                self._handle_goal_success()
                                self._cleanup()
                                return {"status": "success", "goal": self.context_manager.state.goal.description}

                # Increment rounds
                self.increment_round()

            # Max rounds reached (end of for loop)
            goal_desc = self.context_manager.state.goal.description if self.context_manager.state.goal else "unknown"
            self._handle_goal_failure(goal_desc, "Max rounds exceeded")
            self._cleanup()
            return {"status": "failure", "reason": "Max rounds exceeded"}

        except Exception as e:
            # On any exception (including TimeoutError), cleanup and re-raise
            print(f"[cleanup] Exception during run: {e}")
            self._cleanup()
            raise

    def _handle_goal_success(self) -> None:
        """Handle goal success with jetbox notes (if enabled)."""
        jetbox_enabled = any(isinstance(e, JetboxNotesEnhancement) for e in self.enhancements)
        if jetbox_enabled:
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
        else:
            print("\n" + "="*70)
            print("GOAL COMPLETE")
            print("="*70)

    def _handle_goal_failure(self, goal: str, reason: str) -> None:
        """Handle goal failure with jetbox notes (if enabled)."""
        jetbox_enabled = any(isinstance(e, JetboxNotesEnhancement) for e in self.enhancements)
        if jetbox_enabled:
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
        else:
            print("\n" + "="*70)
            print(f"GOAL FAILED: {reason}")
            print("="*70)

    def _cleanup(self) -> None:
        """
        Clean up after task completion.

        Clears Ollama context to prevent state corruption from affecting
        subsequent tasks. This is especially important after:
        - Timeouts (model may be in bad reasoning loop)
        - Complex tasks (circuit breakers, state machines)
        - Any task that may have left model in confused state
        """
        print(f"[cleanup] Clearing Ollama context for {self.model}")
        clear_ollama_context(self.model, self.get_system_prompt())

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
