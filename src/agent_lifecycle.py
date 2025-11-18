"""
Agent lifecycle management - run loop execution and coordination.

This module handles the main agent execution loop:
- Setup and configuration
- Round-by-round execution
- Context building coordination
- Tool call execution coordination
- Completion detection and handling
- Timeout and circuit breaker handling
"""
from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from base_agent import BaseAgent


class AgentLifecycle:
    """
    Manages the agent's run loop execution.

    This class orchestrates the main agent loop by coordinating:
    - Context building (via agent.build_context())
    - LLM calling (via agent._call_llm_with_context())
    - Tool dispatch (via agent.dispatch_tool())
    - Event system triggers (via agent.event_system)
    - Completion detection

    The lifecycle delegates to other agent subsystems but controls
    the overall flow of execution.
    """

    def __init__(self, agent: BaseAgent):
        """
        Initialize lifecycle manager.

        Args:
            agent: The BaseAgent instance to manage
        """
        self.agent = agent

    # ===========================
    # Main entry point
    # ===========================

    def run(self, max_rounds: int | None = None) -> dict[str, Any]:
        """
        Generic agent run loop that works for all agent types.

        This method provides a standard execution loop that:
        1. Triggers behavior events (on_goal_start, on_round_start, etc.)
        2. Calls LLM in a loop
        3. Dispatches tool calls via dispatch_tool()
        4. Checks for completion (mark_complete, goal_complete, etc.)
        5. Handles timeouts and circuit breakers
        6. Returns structured results

        Args:
            max_rounds: Maximum rounds before giving up (defaults to config value)

        Returns:
            Result dict with status, message, etc.
        """
        # Setup: get config and trigger start events
        max_rounds, model, temperature = self._setup_run(max_rounds)

        try:
            # Main loop: execute rounds until completion or max rounds
            for round_no in range(1, max_rounds + 1):
                result = self._execute_round(round_no, max_rounds, model, temperature)
                if result:
                    # Stop display and return result
                    self._display_completion(result)
                    self.agent.display.stop()
                    return result

            # Max rounds reached without completion
            result = self._handle_max_rounds(max_rounds)
            self._display_completion(result)
            self.agent.display.stop()
            return result

        except RuntimeError as e:
            # Handle auto-fail from loop detection
            error_msg = str(e)
            if "Auto-fail" in error_msg or "consecutive empty rounds" in error_msg:
                print(f"[{self.agent.name}] Auto-fail triggered by loop detection: {e}")
                # Return as failure (not error) - this is expected behavior
                result = {
                    "status": "failure",
                    "reason": f"Auto-failed due to stuck state: {error_msg}",
                    "workspace": str(self.agent.workspace) if self.agent.workspace else None,
                }
                self.agent.display.stop()
                return result
            else:
                # Other RuntimeErrors - treat as errors
                print(f"[{self.agent.name}] RuntimeError during run: {e}")
                import traceback
                traceback.print_exc()
                result = {
                    "status": "error",
                    "reason": str(e),
                    "workspace": str(self.agent.workspace) if self.agent.workspace else None,
                }
                self.agent.display.stop()
                return result
        except Exception as e:
            print(f"[{self.agent.name}] Exception during run: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "status": "error",
                "reason": str(e),
                "workspace": str(self.agent.workspace) if self.agent.workspace else None,
            }
            self.agent.display.stop()
            return result

    def run_single_llm_round(self, user_message: str) -> None:
        """
        Run a single LLM round for chat-only mode (no task execution).

        Used by simple chatbots that just need conversation without
        workspace setup, tool loops, or completion detection.

        This method calls the LLM WITHOUT tools to ensure pure conversational
        responses. Tools like set_goal would confuse the chat-only flow.

        Args:
            user_message: User's message/question
        """
        # Add user message to context
        self.agent.add_message({"role": "user", "content": user_message})

        # Build context
        context = self.agent.build_context()

        # Get model and temperature from config (same logic as _setup_run)
        model = getattr(self.agent, 'model', None) or getattr(self.agent.config.llm, 'model', 'qwen3:8b') if self.agent.config else 'qwen3:8b'
        temperature = getattr(self.agent, 'temperature', None) or getattr(self.agent.config.llm, 'temperature', 0.2) if self.agent.config else 0.2

        # Call LLM WITHOUT tools (pure chat mode)
        from src.llm_utils import chat_with_inactivity_timeout

        options = {"temperature": temperature}
        if hasattr(self.agent, 'config') and self.agent.config:
            if hasattr(self.agent.config, 'llm') and self.agent.config.llm:
                if self.agent.config.llm.max_tokens is not None:
                    options["num_ctx"] = self.agent.config.llm.max_tokens

        try:
            response = chat_with_inactivity_timeout(
                model=model,
                messages=context,
                options=options,
                tools=None,  # NO TOOLS - pure conversation mode
                inactivity_timeout=30
            )
        except Exception as e:
            print(f"\n{self.agent.name}: [Error calling LLM: {e}]\n")
            return

        # Add assistant response to history
        if "message" in response:
            self.agent.add_message(response["message"])
            # Print assistant response
            content = response["message"].get("content", "")
            if content:
                print(f"\n{self.agent.name}: {content}\n")
            else:
                # Debug: LLM returned no content
                print(f"\n{self.agent.name}: [No response generated]\n")
                import json
                print(f"[DEBUG] Full response: {json.dumps(response, indent=2)}")

    def run_task_round_loop(
        self,
        user_message: str,
        max_rounds: int = 100,
        check_completion_callback: Callable[[], bool] | None = None
    ) -> None:
        """
        Execute round loop for a single task in multi-task mode.

        This is used by orchestrator for multi-task chat mode where each user
        message is treated as a separate task that runs its own round loop.

        Unlike run(), this method:
        - Does NOT trigger on_goal_set events
        - Does NOT return completion dict
        - Adds user message to history
        - Runs rounds until completion callback returns True or max rounds

        Args:
            user_message: User message to add to history and execute
            max_rounds: Maximum rounds for this task (default: 100)
            check_completion_callback: Optional callback to check if task complete
                                      (e.g., ChatbotBehavior.task_complete_flag)
        """
        # Add user message to history
        self.agent.add_message({"role": "user", "content": user_message})

        # Get model and temperature from config
        model = getattr(self.agent.config.llm, 'model', 'qwen3:8b') if self.agent.config else 'qwen3:8b'
        temperature = getattr(self.agent.config.llm, 'temperature', 0.2) if self.agent.config else 0.2

        # Execute rounds until completion
        for round_no in range(1, max_rounds + 1):
            # Check external completion signal (e.g., ChatbotBehavior detected idle)
            if check_completion_callback and check_completion_callback():
                print(f"[{self.agent.name}] Task complete (detected by callback)")
                break

            # Execute one round
            result = self._execute_round(round_no, max_rounds, model, temperature)

            # Check for goal completion (mark_complete, etc.)
            if result:
                # Task completed via tool call (mark_complete, mark_failed, etc.)
                break

            # Check max rounds
            if round_no >= max_rounds:
                print(f"[{self.agent.name}] Max rounds reached for task")
                break

    # ===========================
    # Setup and configuration
    # ===========================

    def _setup_run(self, max_rounds: int | None = None) -> tuple[int, str, float]:
        """
        Setup run configuration and trigger start events.

        Args:
            max_rounds: Maximum rounds (None = use config default)

        Returns:
            Tuple of (max_rounds, model, temperature)
        """
        # Get max rounds from config or parameter
        if max_rounds is None:
            max_rounds = getattr(self.agent.config.rounds, 'max_per_subtask', 128) if self.agent.config else 128

        # Get model and temperature from config or defaults
        model = getattr(self.agent, 'model', None) or getattr(self.agent.config.llm, 'model', 'qwen3:8b') if self.agent.config else 'qwen3:8b'
        temperature = getattr(self.agent, 'temperature', None) or getattr(self.agent.config.llm, 'temperature', 0.2) if self.agent.config else 0.2

        # Trigger goal start event
        goal_desc = self._get_goal_description()
        if goal_desc:
            self.agent.event_system.trigger_goal_start(goal_desc)

        # Trigger on_initial_context ONCE for first-round setup
        self.agent.event_system.inject_initial_context()

        # Start display
        self.agent.display.start()

        print(f"[{self.agent.name}] Starting run loop (max_rounds={max_rounds}, model={model})")
        return max_rounds, model, temperature

    def _get_goal_description(self) -> str | None:
        """
        Get goal description from context manager or core goal tracking.

        Returns:
            Goal description string or None
        """
        # Use core goal tracking
        if self.agent.goal:
            return self.agent.goal

        return None

    # ===========================
    # Round execution
    # ===========================

    def _execute_round(
        self,
        round_no: int,
        max_rounds: int,
        model: str,
        temperature: float
    ) -> dict[str, Any] | None:
        """
        Execute a single round of the agent loop.

        Args:
            round_no: Current round number
            max_rounds: Maximum rounds
            model: LLM model name
            temperature: Sampling temperature

        Returns:
            Completion dict if goal completed/failed, None to continue
        """
        # Build context for this round
        context = self.agent.build_context()

        # Trigger round start event (called every round before LLM)
        context = self.agent.event_system.trigger_round_start(round_no, context)

        # Update display status
        self._update_display_status(round_no, max_rounds, model)

        # Call LLM with modified context
        print(f"\n[{self.agent.name}] Round {round_no}/{max_rounds}")
        response = self.agent._call_llm_with_context(context, model=model, temperature=temperature)

        # Check for circuit breaker (consecutive timeouts)
        if response.get("_circuit_breaker"):
            print(f"[{self.agent.name}] Circuit breaker triggered - saving partial progress")

            # Trigger timeout event
            if self.agent.goal_start_time:
                elapsed = time.time() - self.agent.goal_start_time
                self.agent.event_system.trigger_timeout(elapsed)

            return self.agent._save_partial_progress()

        # Check for timeout (will retry)
        if response.get("_timeout"):
            print(f"[{self.agent.name}] LLM timeout - retrying next round")
            return None

        # Trigger LLM response event (after LLM responds, before tool dispatch)
        if "message" in response:
            response = self.agent.event_system.trigger_llm_response(response)

        # Add assistant message to history
        had_tool_calls = False
        if "message" in response:
            msg = response["message"]
            self.agent.add_message(msg)

            # Execute tool calls if present
            if "tool_calls" in msg and msg["tool_calls"]:
                completion = self._execute_tool_calls(msg["tool_calls"])
                if completion:
                    return completion

        # Check for completion signals and set nudge (core agent functionality)
        if self.agent.enable_completion_nudging and round_no >= self.agent.min_rounds_before_nudge:
            self.agent._check_completion_signals(response, msg.get("tool_calls", []) if "message" in response else [])

        # Trigger round end event
        self.agent.event_system.trigger_round_end(round_no)

        # Increment round counter
        self.agent.increment_round()

        return None

    # ===========================
    # Tool execution
    # ===========================

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Execute tool calls and check for completion after each.

        Args:
            tool_calls: List of tool call dicts

        Returns:
            Completion dict if goal completed, None otherwise
        """
        import json

        print(f"[{self.agent.name}] Executing {len(tool_calls)} tool call(s)")

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            args = tool_call["function"].get("arguments", {})

            # Create preview of arguments
            preview = self._format_tool_call_preview(tool_name, args)
            print(f"[{self.agent.name}] -> {preview}")

            # Dispatch tool
            result = self.agent.dispatch_tool(tool_call)

            # Check for parameter validation error
            if result.get("status") == "parameter_error":
                print(f"[{self.agent.name}] Parameter validation failed:")
                print(f"  Tool: {result.get('tool_name')}")
                print(f"  Invalid params: {result.get('invalid_params')}")
                # Continue execution - LLM will see error message and retry

            # Add tool result to messages
            tool_result_str = json.dumps(result)
            tool_message = {
                "role": "tool",
                "content": tool_result_str,
            }
            self.agent.add_message(tool_message)

            # Check for completion signal
            completion = self._check_completion_signal(result)
            if completion:
                return completion

        return None

    def _format_tool_call_preview(self, tool_name: str, args: dict[str, Any], max_length: int = 30) -> str:
        """
        Format a tool call with truncated argument preview.

        Args:
            tool_name: Name of the tool
            args: Tool arguments dict
            max_length: Maximum characters for each argument value preview

        Returns:
            Formatted string like "write_file(path=calculator.py, conte...)"
        """
        if not args:
            return f"{tool_name}()"

        # Format each argument with truncated value
        arg_previews = []
        for key, value in args.items():
            # Convert value to string and truncate
            value_str = str(value)
            if len(value_str) > max_length:
                value_preview = value_str[:max_length] + "..."
            else:
                value_preview = value_str

            # Clean up newlines and multiple spaces for display
            value_preview = value_preview.replace("\n", "\\n").replace("\r", "")
            value_preview = " ".join(value_preview.split())  # Collapse whitespace

            arg_previews.append(f"{key}={value_preview}")

        # Join arguments with comma
        args_str = ", ".join(arg_previews)
        return f"{tool_name}({args_str})"

    # ===========================
    # Completion handling
    # ===========================

    def _check_completion_signal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """
        Check if tool result contains a completion signal.

        Args:
            result: Tool execution result

        Returns:
            Completion dict if goal completed/failed, None otherwise
        """
        if not isinstance(result, dict):
            return None

        # IMPORTANT: Exclude delegation results (they have "target_agent" field)
        # Delegation success should NOT auto-complete the calling agent's goal
        is_delegation_result = "target_agent" in result
        if is_delegation_result:
            return None

        # Check for mark_complete (success=True + summary)
        if result.get("success") is True and "summary" in result:
            print(f"[{self.agent.name}] Goal marked complete")
            # Trigger goal complete event
            self.agent.event_system.trigger_goal_complete(success=True, summary=result.get("summary", ""))
            return {
                "status": "success",
                "summary": result.get("summary"),
                "workspace": str(self.agent.workspace) if self.agent.workspace else None,
            }

        # Check for mark_failed (success=False + reason)
        if result.get("success") is False and "reason" in result:
            print(f"[{self.agent.name}] Goal marked failed")
            # Trigger goal complete event
            self.agent.event_system.trigger_goal_complete(success=False, summary=result.get("reason", ""))
            return {
                "status": "failure",
                "reason": result.get("reason"),
                "workspace": str(self.agent.workspace) if self.agent.workspace else None,
            }

        # Check for legacy goal_complete status
        actual_result = result.get("result", result)
        if isinstance(actual_result, dict) and actual_result.get("status") == "goal_complete":
            print(f"[{self.agent.name}] Goal completed (legacy signal)")
            # Trigger goal complete event
            self.agent.event_system.trigger_goal_complete(success=True, summary=actual_result.get("message", "Goal completed"))
            return {
                "status": "success",
                "message": actual_result.get("message", "Goal completed"),
                "workspace": str(self.agent.workspace) if self.agent.workspace else None,
            }

        return None

    def _handle_max_rounds(self, max_rounds: int) -> dict[str, Any]:
        """
        Handle case when max rounds reached without completion.

        Args:
            max_rounds: Maximum rounds that was reached

        Returns:
            Failure result dict
        """
        print(f"[{self.agent.name}] Max rounds ({max_rounds}) reached without completion")
        goal_desc = self._get_goal_description() or "unknown"
        return {
            "status": "failure",
            "reason": f"Max rounds ({max_rounds}) exceeded",
            "goal": goal_desc,
            "workspace": str(self.agent.workspace) if self.agent.workspace else None,
        }

    # ===========================
    # Display helpers
    # ===========================

    def _update_display_status(self, round_no: int, max_rounds: int, model: str) -> None:
        """
        Update TUI display with current agent status.

        Args:
            round_no: Current round number
            max_rounds: Maximum rounds
            model: Model name
        """
        from tui import AgentEvent, EventType
        elapsed_time = time.time() - self.agent.goal_start_time if self.agent.goal_start_time else 0

        self.agent.display.update_status(
            goal=self.agent.goal or "No goal set",
            agent_name=self.agent.name,
            model=model,
            current_round=round_no,
            max_rounds=max_rounds,
            elapsed_time=elapsed_time,
            status="Running",
            tokens_used=None,  # TODO: track tokens if needed
            tokens_max=None,
        )

    def _display_completion(self, result: dict[str, Any]) -> None:
        """
        Show completion summary on display.

        Args:
            result: Completion result dict
        """
        success = result.get("status") == "success"
        summary = result.get("summary") or result.get("reason") or result.get("message") or "Unknown"
        elapsed_time = time.time() - self.agent.goal_start_time if self.agent.goal_start_time else 0

        # Get list of created files if available
        files_created = []
        if hasattr(self.agent, 'workspace_manager') and self.agent.workspace_manager:
            try:
                files_created = list(self.agent.workspace_manager.get_created_files())
            except Exception:
                pass

        self.agent.display.show_completion(
            success=success,
            summary=summary,
            duration=elapsed_time,
            files_created=files_created,
        )
