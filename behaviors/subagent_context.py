"""
SubAgentContextBehavior - Context management for delegated work.

This behavior is for agents invoked by a controlling agent (orchestrator)
to complete a specific delegated task.

Features:
- Context: "DELEGATED GOAL" injection at top
- Tools: mark_complete, mark_failed
- Timeout nudging logic

COMPOSITION:
- This behavior does NOT handle compaction (use CompactWhenNearFullBehavior)
- This behavior does NOT handle notes loading (use WorkspaceTaskNotesBehavior)
- This behavior ONLY manages delegated goal context and completion tools
"""

from typing import Any
from behaviors.base import AgentBehavior


class SubAgentContextBehavior(AgentBehavior):
    """
    Context behavior for sub-agents working on delegated tasks.

    Designed for agents invoked by a controlling agent (orchestrator)
    to complete a specific delegated task. Provides completion tools
    to report results back to the controlling agent.

    Features:
    - Inserts "DELEGATED GOAL" at top of context
    - Provides mark_complete/mark_failed tools
    - Timeout nudging (future: could emit events for timeout detection)

    This behavior is COMPOSABLE:
    - Does NOT handle compaction (delegate to CompactWhenNearFullBehavior)
    - Does NOT load notes (delegate to WorkspaceTaskNotesBehavior)
    - ONLY manages delegated goal context injection and completion signaling
    """

    def __init__(self):
        """
        Initialize sub-agent context behavior.

        No configuration needed - this behavior only injects context and provides tools.
        """
        self.goal = None  # Store goal for context injection

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "subagent_context"

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Enhance context with delegated goal information.

        This method ONLY:
        1. Injects "DELEGATED GOAL" header after system prompt
        2. Adds completion instructions

        It does NOT:
        - Load workspace task notes (use WorkspaceTaskNotesBehavior)
        - Compact context (use CompactWhenNearFullBehavior)

        Args:
            context: Current context (system + messages)
            **kwargs: Additional context (context_manager)

        Returns:
            Modified context with delegated goal info
        """
        context_manager = kwargs.get('context_manager')

        if not context_manager:
            return context

        # Build delegated goal context
        context_parts = []

        if context_manager.state.goal:
            context_parts.append(f"DELEGATED GOAL: {context_manager.state.goal.description}")
            context_parts.append("")
            context_parts.append("You are working on a task delegated by the orchestrator.")
            context_parts.append("When complete, call mark_complete(summary) with what you accomplished.")
            context_parts.append("If you cannot complete it, call mark_failed(reason) explaining why.")

        # Insert after system prompt (index 1)
        if context_parts and len(context) > 0:
            context.insert(1, {
                "role": "user",
                "content": "\n".join(context_parts)
            })

        return context

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Provide completion tools for sub-agent.

        Returns:
            Tool definitions for mark_complete and mark_failed
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "mark_complete",
                    "description": "Mark the delegated task as complete and report success to controlling agent. REQUIRED when work is finished.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Brief summary of what was accomplished (2-4 sentences)"
                            }
                        },
                        "required": ["summary"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_failed",
                    "description": "Mark the delegated task as failed and report reason to controlling agent. Use when you cannot complete the task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Explanation of why the task could not be completed"
                            }
                        },
                        "required": ["reason"]
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
            **kwargs: Additional context

        Returns:
            Tool result dict
        """
        context_manager = kwargs.get('context_manager')

        if tool_name == "mark_complete":
            summary = args.get('summary', 'Task completed')

            # Mark goal as complete in context manager if available
            if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
                context_manager.state.goal.mark_complete(success=True)

            return {
                "success": True,
                "result": f"Task marked complete: {summary}",
                "summary": summary
            }

        elif tool_name == "mark_failed":
            reason = args.get('reason', 'Task failed')

            # Mark goal as failed in context manager if available
            if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
                context_manager.state.goal.mark_complete(success=False)

            return {
                "success": False,
                "result": f"Task marked failed: {reason}",
                "reason": reason
            }

        return super().dispatch_tool(tool_name, args, **kwargs)

    def get_instructions(self) -> str:
        """
        Return sub-agent workflow instructions.

        Returns:
            Instructions for completing delegated work
        """
        return """
SUB-AGENT WORKFLOW:
You are working on a task delegated by a controlling agent (orchestrator).

Your job is to:
1. Complete the delegated goal using available tools
2. Report results back to the controlling agent when done

Available tools:
- write_file: Create/modify files
- run_bash: Run commands (tests, linters, build tools, etc.)
- read_file: Check existing files
- list_dir: Explore directories
- mark_complete(summary): Signal successful completion with summary
- mark_failed(reason): Signal failure with explanation

IMPORTANT - YOU MUST SIGNAL COMPLETION:
- When work is done and tests pass: call mark_complete(summary="what you accomplished")
- If you cannot complete the task: call mark_failed(reason="why it failed")
- DO NOT just stop - always call one of these tools to report results

The controlling agent is waiting for your completion signal.
"""

    def on_goal_set(self, agent: Any, goal: str, **kwargs: Any) -> None:
        """
        Handle goal setting event.

        This method initializes all subsystems needed for goal execution:
        - Context manager with goal
        - Workspace manager (new or existing)
        - Performance tracking
        - Tool configuration

        Args:
            agent: Agent instance
            goal: Goal description
            **kwargs: Additional context (workspace, context_manager, workspace_manager, etc.)
        """
        # Store goal for context injection
        self.goal = goal

        # Initialize context manager with goal
        if not agent.context_manager:
            from context_manager import ContextManager
            agent.context_manager = ContextManager()
        agent.context_manager.load_or_init(goal)

        # Initialize workspace manager
        # workspace parameter: None = create new, Path = reuse existing
        workspace_param = kwargs.get('workspace')
        goal_slug = goal.lower()[:50].replace(" ", "-").replace("/", "-")

        if workspace_param:
            # Reuse mode: use existing workspace directory
            print(f"[subagent_context] Reusing workspace: {workspace_param}")
            agent.init_workspace_manager(goal_slug, workspace_path=workspace_param)
        else:
            # Create new mode: create isolated workspace
            print(f"[subagent_context] Creating new workspace for goal")
            agent.init_workspace_manager(goal_slug, workspace_path=None)

        # Initialize performance tracking
        agent.init_perf_stats()

        # Configure tools with workspace (for FileToolsBehavior, CommandToolsBehavior)
        # This is handled by behaviors themselves - no need to do it here

        # Initialize status display if not already present
        if not hasattr(agent, 'status_display') or agent.status_display is None:
            from status_display import StatusDisplay
            agent.status_display = StatusDisplay(ctx=agent.context_manager, reset_stats=True)

        # Start wall-clock timer for goal
        import time
        agent.goal_start_time = time.time()

        print(f"[subagent_context] Goal set: {goal}")
