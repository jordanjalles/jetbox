"""
Base class for composable agent behaviors.

Behaviors extend agent capabilities through:
- Context injection (enhance_context)
- Tool registration (get_tools, dispatch_tool)
- Event handling (on_* methods)

All behaviors are:
- Self-contained (no dependencies on other behaviors)
- Composable (multiple behaviors work together)
- Optional (agents choose which behaviors to use)
"""

from abc import ABC, abstractmethod
from typing import Any


class AgentBehavior(ABC):
    """
    Base class for composable agent behaviors.

    Behaviors extend agent capabilities by providing:
    - Context modifications (enhance_context)
    - Tool definitions and dispatch (get_tools, dispatch_tool)
    - System prompt instructions (get_instructions)
    - Event handlers (on_goal_start, on_tool_call, etc.)

    Example:
        ```python
        class MyBehavior(AgentBehavior):
            def get_name(self) -> str:
                return "my_behavior"

            def get_tools(self) -> list[dict[str, Any]]:
                return [{
                    "type": "function",
                    "function": {
                        "name": "my_tool",
                        "description": "Does something useful",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "arg": {"type": "string"}
                            }
                        }
                    }
                }]

            def dispatch_tool(
                self,
                tool_name: str,
                args: dict[str, Any],
                **kwargs
            ) -> dict[str, Any]:
                if tool_name == "my_tool":
                    return {"result": f"Processed: {args['arg']}"}
                return super().dispatch_tool(tool_name, args, **kwargs)
        ```
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Return behavior identifier.

        This name is used for:
        - Logging and debugging
        - Error messages
        - Tool registration tracking

        Returns:
            Unique behavior identifier (e.g., 'file_tools', 'loop_detection')

        Example:
            ```python
            def get_name(self) -> str:
                return "file_tools"
            ```
        """
        pass

    # Context injection

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Modify context before LLM call.

        Called by the agent before each LLM invocation to allow behaviors
        to inject additional information, modify system prompts, or
        transform message history.

        Args:
            context: Current context as list of message dicts with 'role' and 'content'
                    Typically: [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
            **kwargs: Additional context information that may include:
                - agent: The agent instance
                - workspace: Workspace path
                - round_number: Current round number
                - context_manager: ContextManager instance (if applicable)
                - goal: Current goal string
                - Any other agent-specific data

        Returns:
            Modified context (same format as input)

        Default Implementation:
            Returns context unchanged.

        Example:
            ```python
            def enhance_context(self, context, **kwargs):
                # Inject additional information into system prompt
                if context and context[0]["role"] == "system":
                    context[0]["content"] += "\\n\\nNote: Remember to check files exist."
                return context
            ```
        """
        return context

    # Tool registration

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tools provided by this behavior.

        Tools follow the OpenAI function calling schema:
        https://platform.openai.com/docs/guides/function-calling

        Returns:
            List of tool definitions in OpenAI function calling format.
            Each tool is a dict with:
            - type: "function"
            - function: {name, description, parameters}

        Default Implementation:
            Returns empty list (no tools).

        Example:
            ```python
            def get_tools(self) -> list[dict[str, Any]]:
                return [{
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to file to read"
                                }
                            },
                            "required": ["path"]
                        }
                    }
                }]
            ```
        """
        return []

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Handle tool call for this behavior's tools.

        Called by the agent when the LLM requests to use one of this
        behavior's tools. The behavior should execute the tool and
        return the result.

        Args:
            tool_name: Name of the tool being called
            args: Tool arguments (parsed from LLM's function call)
            **kwargs: Additional context that may include:
                - agent: The agent instance
                - workspace: Workspace path
                - context_manager: ContextManager instance
                - Any other agent-specific data

        Returns:
            Tool result as a dict. Common patterns:
            - Success: {"result": "...", "success": True}
            - Error: {"error": "error message"}
            - Data: {"data": {...}, "success": True}

        Raises:
            NotImplementedError: If this behavior doesn't handle the tool.
            This is the default implementation.

        Example:
            ```python
            def dispatch_tool(self, tool_name, args, **kwargs):
                if tool_name == "read_file":
                    path = args["path"]
                    try:
                        with open(path) as f:
                            content = f.read()
                        return {"content": content, "success": True}
                    except Exception as e:
                        return {"error": str(e)}

                # Fall through to parent for unknown tools
                return super().dispatch_tool(tool_name, args, **kwargs)
            ```
        """
        raise NotImplementedError(
            f"Behavior {self.get_name()} does not handle tool: {tool_name}"
        )

    # Instructions

    def get_instructions(self) -> str:
        """
        Return instructions to add to system prompt.

        These instructions help guide the LLM on how to use this behavior's
        capabilities (e.g., when to call certain tools, usage patterns,
        constraints).

        Returns:
            Instructions text (markdown formatted). Empty string if no instructions.

        Default Implementation:
            Returns empty string (no instructions).

        Example:
            ```python
            def get_instructions(self) -> str:
                return '''
                ## File Operations

                You have access to file tools:
                - read_file: Read file contents
                - write_file: Write or create a file
                - list_dir: List directory contents

                Always check if a file exists before reading.
                '''
            ```
        """
        return ""

    # Event handlers

    def on_goal_start(self, goal: str, **kwargs: Any) -> None:
        """
        Called when goal starts.

        Triggered when the agent begins working on a new goal. Use this to:
        - Initialize behavior state
        - Set up resources
        - Log goal start
        - Prepare workspace

        Args:
            goal: The goal string
            **kwargs: Additional context (agent, workspace, etc.)

        Default Implementation:
            Does nothing (no-op).

        Example:
            ```python
            def on_goal_start(self, goal, **kwargs):
                self.start_time = time.time()
                self.action_count = 0
                print(f"Starting goal: {goal}")
            ```
        """
        pass

    def on_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Called after each tool execution.

        Triggered immediately after any tool is executed (regardless of which
        behavior owns it). Use this to:
        - Track tool usage
        - Detect loops (repeated tool calls)
        - Log actions
        - Update statistics

        Args:
            tool_name: Name of the tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool
            **kwargs: Additional context (agent, workspace, etc.)

        Default Implementation:
            Does nothing (no-op).

        Example:
            ```python
            def on_tool_call(self, tool_name, args, result, **kwargs):
                self.action_count += 1
                if "error" in result:
                    self.error_count += 1
                print(f"Tool called: {tool_name}")
            ```
        """
        pass

    def on_round_end(self, round_number: int, **kwargs: Any) -> None:
        """
        Called at end of each round.

        A round is one LLM call + tool execution cycle. Use this to:
        - Render status displays
        - Persist state
        - Check timeout conditions
        - Update progress tracking

        Args:
            round_number: Current round number (1-indexed)
            **kwargs: Additional context (agent, workspace, etc.)

        Default Implementation:
            Does nothing (no-op).

        Example:
            ```python
            def on_round_end(self, round_number, **kwargs):
                if round_number % 5 == 0:
                    print(f"Round {round_number}: {self.action_count} actions")
                self.persist_state()
            ```
        """
        pass

    def on_timeout(self, elapsed_seconds: float, **kwargs: Any) -> None:
        """
        Called when goal times out.

        Triggered when the agent exceeds its time budget for a goal. Use this to:
        - Generate timeout summaries
        - Save partial progress
        - Clean up resources
        - Log timeout reason

        Args:
            elapsed_seconds: Time elapsed since goal start
            **kwargs: Additional context (agent, workspace, goal, etc.)

        Default Implementation:
            Does nothing (no-op).

        Example:
            ```python
            def on_timeout(self, elapsed_seconds, **kwargs):
                print(f"Goal timed out after {elapsed_seconds:.1f}s")
                self.save_partial_results()
            ```
        """
        pass

    def on_goal_complete(self, success: bool, **kwargs: Any) -> None:
        """
        Called when goal completes.

        Triggered when the agent finishes a goal (either successfully or with failure).
        Use this to:
        - Generate completion summaries
        - Clean up resources
        - Save final results
        - Log completion status

        Args:
            success: True if goal completed successfully, False if failed
            **kwargs: Additional context (agent, workspace, goal, etc.)

        Default Implementation:
            Does nothing (no-op).

        Example:
            ```python
            def on_goal_complete(self, success, **kwargs):
                status = "SUCCESS" if success else "FAILED"
                print(f"Goal {status}: {self.action_count} actions in {elapsed}s")
                self.generate_summary(success)
            ```
        """
        pass
