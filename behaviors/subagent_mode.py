"""
SubAgentModeBehavior - Makes agents delegatable (can be invoked by other agents).

This behavior enables an agent to be delegated to in two ways:
1. CLI mode: python agent.py "goal" → execute goal, return result, exit
2. Tool call mode: parent_agent.delegate_task(agent, goal) → execute, return result

Key features:
- Detects if goal was provided (CLI arg or tool call)
- Sets up workspace, context manager, performance tracking
- Provides completion signaling tools (mark_complete, mark_failed)
- Handles execution mode (no chat, just work and return)

COMPOSITION:
- This behavior does NOT handle compaction (use CompactWhenNearFullBehavior)
- This behavior does NOT handle notes loading (use WorkspaceTaskNotesBehavior)
- This behavior does NOT handle chat mode (use ChatbotBehavior)
- This behavior ONLY manages delegated execution mode and completion tools

This is a RENAMED and ENHANCED version of SubAgentContextBehavior.
"""

from typing import Any
from behaviors.base import AgentBehavior


class SubAgentModeBehavior(AgentBehavior):
    """
    Behavior that makes agents delegatable by other agents or CLI.

    Designed for agents that can be invoked by:
    - A controlling agent (orchestrator) for delegation
    - CLI invocation with a goal argument

    Features:
    - Inserts "DELEGATED GOAL" or "GOAL" at top of context
    - Provides mark_complete/mark_failed tools for signaling results
    - Initializes workspace, context manager, perf stats on goal_set event
    - Supports both new workspace creation and existing workspace reuse

    This behavior is COMPOSABLE:
    - Does NOT handle compaction (delegate to CompactWhenNearFullBehavior)
    - Does NOT load notes (delegate to WorkspaceTaskNotesBehavior)
    - Does NOT handle chat mode (delegate to ChatbotBehavior)
    - ONLY manages execution mode setup and completion signaling
    """

    def __init__(self, is_subagent: bool = True):
        """
        Initialize subagent mode behavior.

        Args:
            is_subagent: If True, agent is being delegated to (shows "DELEGATED GOAL").
                        If False, agent runs standalone (shows "GOAL").
        """
        self.is_subagent = is_subagent
        self.goal = None  # Store goal for context injection

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "subagent_mode"

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Enhance context with goal information.

        This method ONLY:
        1. Injects goal header after system prompt
        2. Adds completion instructions

        It does NOT:
        - Load workspace task notes (use WorkspaceTaskNotesBehavior)
        - Compact context (use CompactWhenNearFullBehavior)

        Args:
            context: Current context (system + messages)
            **kwargs: Additional context (context_manager)

        Returns:
            Modified context with goal info
        """
        context_manager = kwargs.get('context_manager')

        if not context_manager:
            return context

        # Build goal context
        context_parts = []

        if context_manager.state.goal:
            # Use different header based on delegation status
            if self.is_subagent:
                context_parts.append(f"DELEGATED GOAL: {context_manager.state.goal.description}")
                context_parts.append("")
                context_parts.append("You are working on a task delegated by a parent agent.")
            else:
                context_parts.append(f"GOAL: {context_manager.state.goal.description}")
                context_parts.append("")
                context_parts.append("You are working on a standalone task.")

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
        Provide completion tools for signaling results.

        Returns:
            Tool definitions for mark_complete and mark_failed
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "mark_complete",
                    "description": "Mark the task/goal as complete and report success. REQUIRED when work is finished.",
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
                    "description": "Mark the task/goal as failed and report reason. Use when you cannot complete the task.",
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
                context_manager.state.goal.status = "success"

            return {
                "success": True,
                "result": f"Task marked complete: {summary}",
                "summary": summary,
                "status": "goal_complete"
            }

        elif tool_name == "mark_failed":
            reason = args.get('reason', 'Task failed')

            # Mark goal as failed in context manager if available
            if context_manager and hasattr(context_manager, 'state') and context_manager.state.goal:
                context_manager.state.goal.status = "failed"

            return {
                "success": False,
                "result": f"Task marked failed: {reason}",
                "reason": reason,
                "status": "goal_failed"
            }

        return super().dispatch_tool(tool_name, args, **kwargs)

    def get_instructions(self) -> str:
        """
        Return execution workflow instructions.

        Returns:
            Instructions for completing delegated work or standalone goals
        """
        if self.is_subagent:
            return """
SUB-AGENT EXECUTION MODE:
You are working on a task delegated by a parent agent (orchestrator or another agent).

Your job is to:
1. Complete the delegated goal using available tools
2. Report results back to the parent agent when done

Available workflow tools:
- mark_complete(summary): Signal successful completion with summary
- mark_failed(reason): Signal failure with explanation

IMPORTANT - YOU MUST SIGNAL COMPLETION:
- When work is done and tests pass: call mark_complete(summary="what you accomplished")
- If you cannot complete the task: call mark_failed(reason="why it failed")
- DO NOT just stop - always call one of these tools to report results

The parent agent is waiting for your completion signal.
"""
        else:
            return """
STANDALONE EXECUTION MODE:
You are working on a goal provided via command line or direct invocation.

Your job is to:
1. Complete the goal using available tools
2. Signal completion when done

Available workflow tools:
- mark_complete(summary): Signal successful completion with summary
- mark_failed(reason): Signal failure with explanation

IMPORTANT - YOU MUST SIGNAL COMPLETION:
- When work is done and tests pass: call mark_complete(summary="what you accomplished")
- If you cannot complete the task: call mark_failed(reason="why it failed")
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

        # Initialize context manager with goal (DEPRECATED with behavior system)
        # Context is now managed by behaviors, not a separate ContextManager
        # if not agent.context_manager:
        #     from context_manager import ContextManager
        #     agent.context_manager = ContextManager()
        # agent.context_manager.load_or_init(goal)

        # Initialize workspace manager
        # workspace parameter: None = create new, Path = reuse existing
        workspace_param = kwargs.get('workspace')
        goal_slug = goal.lower()[:50].replace(" ", "-").replace("/", "-")

        if workspace_param:
            # Reuse mode: use existing workspace directory
            print(f"[subagent_mode] Reusing workspace: {workspace_param}")
            agent.init_workspace_manager(goal_slug, workspace_path=workspace_param)
        else:
            # Create new mode: create isolated workspace
            print(f"[subagent_mode] Creating new workspace for goal")
            agent.init_workspace_manager(goal_slug, workspace_path=None)

        # Initialize performance tracking
        agent.init_perf_stats()

        # Configure tools with workspace (for FileToolsBehavior, CommandToolsBehavior)
        # This is handled by behaviors themselves - no need to do it here

        # Initialize status display if not already present
        # DEPRECATED: StatusDisplay is being redesigned for behavior system
        # if not hasattr(agent, 'status_display') or agent.status_display is None:
        #     from behaviors.status_display import StatusDisplay
        #     agent.status_display = StatusDisplay(ctx=agent.context_manager, reset_stats=True)

        # Start wall-clock timer for goal
        import time
        agent.goal_start_time = time.time()

        print(f"[subagent_mode] Goal set: {goal}")


# Backward compatibility alias
SubAgentContextBehavior = SubAgentModeBehavior
