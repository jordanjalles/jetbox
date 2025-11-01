"""
HierarchicalContextBehavior - Goal/Task/Subtask hierarchy management.

This behavior provides:
- Context: Goal/Task/Subtask hierarchy injection
- Tools: decompose_task, mark_subtask_complete
- Messages: Keep last N exchanges (default 12)
- Clear on subtask transitions
- Instructions: Explain hierarchical workflow
"""

from typing import Any
from behaviors.base import AgentBehavior


class HierarchicalContextBehavior(AgentBehavior):
    """
    Context behavior for hierarchical task management.

    Injects Goal → Task → Subtask hierarchy into context and provides
    tools for task decomposition and subtask completion signaling.

    Features:
    - Shows current goal/task/subtask at top of context
    - Keeps only recent N message exchanges
    - Clears messages on subtask transitions (configurable)
    - Loop detection warnings from context manager
    - Filesystem probe state injection
    - Jetbox notes integration
    """

    def __init__(
        self,
        history_keep: int = 12,
        clear_on_transition: bool = True,
    ):
        """
        Initialize hierarchical context behavior.

        Args:
            history_keep: Number of message exchanges to keep in history (default: 12)
            clear_on_transition: Whether to clear messages on subtask transitions (default: True)
        """
        self.history_keep = history_keep
        self.clear_on_transition = clear_on_transition

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "hierarchical_context"

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Enhance context with hierarchical task information.

        This method:
        1. Injects goal/task/subtask hierarchy after system prompt
        2. Adds loop detection warnings if present
        3. Adds filesystem probe state if available
        4. Prunes message history to last N exchanges

        Args:
            context: Current context (system + messages)
            **kwargs: Additional context (context_manager, probe_state_func, workspace, config)

        Returns:
            Modified context with hierarchy information
        """
        context_manager = kwargs.get('context_manager')
        probe_state_func = kwargs.get('probe_state_func')
        workspace = kwargs.get('workspace')
        config = kwargs.get('config')

        if not context_manager:
            return context

        # Build hierarchy information
        hierarchy_info = self._build_hierarchy_info(
            context_manager=context_manager,
            config=config
        )

        # Add loop detection warnings
        loop_warnings = self._get_loop_warnings(context_manager)
        if loop_warnings:
            hierarchy_info.append("")
            hierarchy_info.extend(loop_warnings)

        # Add filesystem probe state
        if probe_state_func:
            probe_info = self._get_probe_info(probe_state_func)
            if probe_info:
                hierarchy_info.append("")
                hierarchy_info.extend(probe_info)

        # Add jetbox notes if available
        if workspace:
            notes_info = self._get_jetbox_notes()
            if notes_info:
                hierarchy_info.append("")
                hierarchy_info.extend(notes_info)

        # Insert hierarchy info after system prompt (index 1)
        if len(context) > 0:
            context.insert(1, {
                "role": "user",
                "content": "\n".join(hierarchy_info)
            })

        # Prune message history (keep last N exchanges)
        # Find where messages start (after system + hierarchy)
        messages_start_idx = 2  # system + hierarchy
        if len(context) > messages_start_idx:
            messages = context[messages_start_idx:]

            # Keep last N exchanges (each exchange is 2 messages: user + assistant)
            keep_messages = self.history_keep * 2
            if len(messages) > keep_messages:
                pruned_messages = messages[-keep_messages:]
                context = context[:messages_start_idx] + pruned_messages

        return context

    def _build_hierarchy_info(
        self,
        context_manager: Any,
        config: Any = None
    ) -> list[str]:
        """
        Build goal/task/subtask hierarchy information.

        Args:
            context_manager: ContextManager instance
            config: Agent configuration

        Returns:
            List of info lines
        """
        info = []

        if not context_manager.state.goal:
            return info

        info.append(f"GOAL: {context_manager.state.goal.description}")

        # Get current task and subtask
        task = context_manager._get_current_task()
        subtask = task.active_subtask() if task else None

        if task:
            info.append(f"CURRENT TASK: {task.description}")

            if subtask:
                info.append(f"ACTIVE SUBTASK: {subtask.description}")

                # Add depth/rounds info if config provided
                if config:
                    if hasattr(config, 'hierarchy') and hasattr(config.hierarchy, 'max_depth'):
                        info.append(f"Subtask Depth: {subtask.depth}/{config.hierarchy.max_depth}")

                    if hasattr(config, 'rounds') and hasattr(config.rounds, 'max_per_subtask'):
                        info.append(f"Rounds Used: {subtask.rounds_used}/{config.rounds.max_per_subtask}")
            else:
                info.append("ACTIVE SUBTASK: (none - call mark_subtask_complete to advance)")
        else:
            # No tasks yet - need to decompose goal
            info.append("")
            info.append("⚠️  NO TASKS YET")
            info.append("The goal has not been broken down into tasks.")
            info.append("Use decompose_task to create an initial task/subtask structure.")

        return info

    def _get_loop_warnings(self, context_manager: Any) -> list[str]:
        """
        Get loop detection warnings from context manager.

        Args:
            context_manager: ContextManager instance

        Returns:
            List of warning lines (empty if no loops detected)
        """
        if not hasattr(context_manager, 'state') or not hasattr(context_manager.state, 'loop_counts'):
            return []

        loop_warnings = []
        for sig, count in context_manager.state.loop_counts.items():
            if count > 0:
                loop_warnings.append(f"  • Action repeated {count}x: {sig[:80]}")

        if not loop_warnings:
            return []

        warnings = [
            "⚠️  LOOP DETECTION WARNING:",
            "You appear to be repeating actions. Consider:",
            "- Trying a COMPLETELY DIFFERENT approach",
            "- Reading error messages carefully",
            "- Checking if assumptions are wrong",
            "- Asking yourself: 'Why didn't the last attempt work?'",
            "",
            "Detected loops:"
        ]
        warnings.extend(loop_warnings)
        warnings.append("")
        warnings.append("Try something NEW this round.")

        return warnings

    def _get_probe_info(self, probe_state_func: callable) -> list[str]:
        """
        Get filesystem probe state information.

        Args:
            probe_state_func: Function that returns probe state dict

        Returns:
            List of probe info lines
        """
        probe = probe_state_func()
        info = []

        # Add workspace warning if present
        if probe.get("warning"):
            info.append("")
            info.append(probe["warning"])
            info.append("")

        if probe.get("files_exist"):
            info.append(f"FILES CREATED: {', '.join(probe['files_exist'])}")
        if probe.get("recent_errors"):
            info.append(f"RECENT ERRORS: {probe['recent_errors'][-1][:100]}")

        return info

    def _get_jetbox_notes(self) -> list[str]:
        """
        Get jetbox notes content if available.

        Returns:
            List of notes lines (empty if no notes)
        """
        try:
            import jetbox_notes
            notes_content = jetbox_notes.load_jetbox_notes(max_chars=2000)
            if not notes_content:
                return []

            return [
                "="*70,
                "JETBOX NOTES (Previous Work Summary)",
                "="*70,
                notes_content,
                "="*70
            ]
        except Exception:
            return []

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Provide hierarchical workflow tools.

        Returns:
            Tool definitions for decompose_task and mark_subtask_complete
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "mark_subtask_complete",
                    "description": "Mark current subtask as complete and advance to next subtask. Call this when you finish a subtask.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether the subtask was completed successfully"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief explanation of what was done or why it failed"
                            }
                        },
                        "required": ["success"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "decompose_task",
                    "description": "Break down the goal into tasks and subtasks. Use this when NO TASKS YET to create initial structure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtasks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of subtask descriptions (2-6 subtasks recommended)"
                            }
                        },
                        "required": ["subtasks"]
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Handle tool calls for this behavior.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: Additional context (context_manager, etc.)

        Returns:
            Tool result dict
        """
        context_manager = kwargs.get('context_manager')

        if not context_manager:
            return {"error": "No context manager available"}

        if tool_name == "mark_subtask_complete":
            success = args.get('success', True)
            reason = args.get('reason', 'Subtask completed')

            try:
                result = context_manager.mark_subtask_complete(
                    success=success,
                    reason=reason
                )
                return {
                    "success": True,
                    "result": result
                }
            except Exception as e:
                return {"error": f"Failed to mark subtask complete: {e}"}

        elif tool_name == "decompose_task":
            subtasks = args.get('subtasks', [])

            if not subtasks:
                return {"error": "No subtasks provided"}

            try:
                # Create task with subtasks
                context_manager.create_task_with_subtasks(
                    description="Main task",
                    subtasks=subtasks
                )
                return {
                    "success": True,
                    "result": f"Created task with {len(subtasks)} subtasks",
                    "subtasks": subtasks
                }
            except Exception as e:
                return {"error": f"Failed to decompose task: {e}"}

        return super().dispatch_tool(tool_name, args, **kwargs)

    def get_instructions(self) -> str:
        """
        Return hierarchical workflow instructions.

        Returns:
            Instructions for using hierarchical tools
        """
        return """
HIERARCHICAL WORKFLOW:
1. If NO TASKS YET: Use decompose_task to break down the goal into a task structure
2. Work on your current subtask using the available tools
3. When you complete a subtask, call mark_subtask_complete(success=True)
4. If you cannot complete it, call mark_subtask_complete(success=False, reason="...")
5. The system will automatically advance you to the next subtask

Hierarchical-specific guidelines:
- For simple goals, create ONE task with ONE subtask, then immediately work on it
- When you finish your current subtask, ALWAYS call mark_subtask_complete
- Focus on the current subtask shown in the context

Additional tools for hierarchical workflow:
- mark_subtask_complete(success, reason): Mark current subtask done
- decompose_task(subtasks): Break down goal into tasks (only if NO TASKS YET)
"""
