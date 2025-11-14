"""
Base class for composable agent behaviors.

Behaviors extend agent capabilities through:
- Context injection (on_initial_context, on_round_start)
- Tool registration (get_tools, dispatch_tool)
- Event handling (on_* methods)
- Custom event communication (on_custom_event, fire_custom_event)

All behaviors are:
- Self-contained (no dependencies on other behaviors)
- Composable (multiple behaviors work together)
- Optional (agents choose which behaviors to use)

Lifecycle Events (in order):
1. on_goal_start() - Once at goal start
2. on_initial_context() - Once for initial context injection
3. on_round_start() - Every round before LLM
4. on_llm_response() - After LLM responds
5. on_tool_call() - After each tool execution
6. on_round_end() - After all tools in round
7. on_goal_complete() or on_timeout() - At goal end

Custom Events (any time):
- on_custom_event() - Called when behaviors broadcast custom events
- Enables inter-behavior communication for mode coordination, state sync, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from behaviors.base_agent import BaseAgent

# Import for type hints only (avoids circular import)
try:
    from behaviors.rule_of_two_types import RuleOfTwoProperty
except ImportError:
    RuleOfTwoProperty = None  # type: ignore


class AgentBehavior(ABC):
    """
    Base class for composable agent behaviors.

    Behaviors extend agent capabilities by providing:
    - Context modifications (on_initial_context, on_round_start)
    - Tool definitions and dispatch (get_tools, dispatch_tool)
    - System prompt instructions (get_instructions)
    - Event handlers (on_goal_start, on_tool_call, on_round_end, etc.)
    - Custom event communication (on_custom_event)

    Security (Rule of Two):
    - Each behavior declares its security properties via rule_of_two_properties
    - Properties: [A] untrusted input, [B] sensitive access, [C] external action
    - Default: {A, B, C} (safest assumption, requires explicit override)

    Lifecycle Events (in chronological order):
    1. on_goal_start(agent, goal) - Once at goal start
    2. on_initial_context(agent, context) - Once for initial setup
    3. on_round_start(agent, round_number, context) - Every round before LLM
    4. on_llm_response(agent, response) - After LLM responds
    5. on_tool_call(agent, tool_name, args, result) - After each tool execution
    6. on_round_end(agent, round_number) - After all tools in round
    7. on_goal_complete(agent, success, summary) or on_timeout(agent, elapsed) - At goal end

    Custom Events (any time):
    - on_custom_event(agent, event_name, **event_data) - Inter-behavior communication

    Example:
        ```python
        class MyBehavior(AgentBehavior):
            # Declare security properties (override safe default)
            rule_of_two_properties = {RuleOfTwoProperty.EXTERNAL_ACTION}  # [C] only

            def get_name(self) -> str:
                return "my_behavior"

            def on_goal_start(self, agent, goal):
                self.start_time = time.time()
                print(f"Starting goal: {goal}")

            def on_round_start(self, agent, round_number, context):
                # Append dynamic warnings at END (immediate feedback)
                if self.should_warn:
                    warning = "⚠️ Consider a different approach"
                    context.append({"role": "user", "content": warning})
                return context

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
                            },
                            "required": ["arg"]
                        }
                    }
                }]

            def dispatch_tool(self, agent, tool_name, args):
                if tool_name == "my_tool":
                    # Can access agent.workspace directly
                    result = f"Processed: {args['arg']}"
                    return {"result": result, "success": True}
                return super().dispatch_tool(agent, tool_name, args)

            def on_tool_call(self, agent, tool_name, args, result):
                self.tool_count += 1
                print(f"Tool called: {tool_name}")
        ```
    """

    # Rule of Two security properties
    # Default: {A, B, C} = safest assumption, requires explicit override
    # Behaviors MUST override to declare their actual properties
    # Example: rule_of_two_properties = {RuleOfTwoProperty.EXTERNAL_ACTION}
    rule_of_two_properties: set  # Will be set to {A, B, C} after import

    def get_rule_of_two_properties(
        self,
        agent: "BaseAgent",
        security_context: Any = None
    ) -> set:
        """
        Get Rule of Two properties for this behavior (context-aware).

        This method enables dynamic, context-aware property resolution.
        Behaviors can override this to change properties based on:
        - Workspace characteristics (untrusted files, sensitive files, network access)
        - Workspace trust level (isolated vs user files)
        - Execution mode

        Args:
            agent: Agent instance (for accessing workspace, state, etc.)
            security_context: SecurityContext with runtime security state (optional)

        Returns:
            Set of RuleOfTwoProperty values for current security context

        Default Implementation:
            Returns static rule_of_two_properties attribute (backwards compatible).

        Example (Dynamic Properties):
            ```python
            def get_rule_of_two_properties(self, agent, security_context):
                props = {RuleOfTwoProperty.SENSITIVE_ACCESS}  # Always [B]

                # [A] only if workspace contains untrusted files
                if context and context.workspace_trust_level == "user":
                    props.add(RuleOfTwoProperty.UNTRUSTED_INPUT)

                return props
            ```

        Example (Static Properties - default behavior):
            ```python
            # Don't override method, just set class attribute:
            rule_of_two_properties = {RuleOfTwoProperty.EXTERNAL_ACTION}
            ```
        """
        # Default: return static class attribute (backwards compatible)
        return self.rule_of_two_properties

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

    # =================================================================
    # LIFECYCLE EVENTS (new API - preferred)
    # =================================================================

    def on_goal_start(self, agent: "BaseAgent", goal: str) -> None:
        """
        Called ONCE when goal is set (beginning of agent execution).

        TIMING: Called once at goal start, before any rounds.
        FREQUENCY: Once per goal.

        Use for:
        - Initializing behavior state
        - Setting up resources
        - Preparing workspace
        - Logging goal start

        Args:
            agent: Agent instance (access agent.workspace, agent.state, etc.)
            goal: Goal string

        Default Implementation:
            Does nothing (no-op).

        Example:
            ```python
            def on_goal_start(self, agent, goal):
                self.start_time = time.time()
                self.action_count = 0
                print(f"Starting goal: {goal}")
            ```
        """
        pass

    def on_initial_context(
        self,
        agent: "BaseAgent",
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Called ONCE to inject initial context (first round only).

        TIMING: Called once at start of goal, after system prompt is built,
                before first LLM call.
        FREQUENCY: Once per goal (not repeated on subsequent rounds).

        Use for:
        - Injecting goal description
        - Adding initial instructions
        - Loading workspace notes
        - Setting up delegation options
        - Any one-time context injection

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)
                    Format: [{"role": "system", "content": "..."}]

        Returns:
            Modified context with initial messages injected

        Default Implementation:
            Returns context unchanged.

        Example:
            ```python
            def on_initial_context(self, agent, context):
                # Inject goal description
                goal_msg = f"GOAL: {agent.goal}"
                return self.inject_message_after_system(context, goal_msg, role="user")
            ```
        """
        return context

    def on_round_start(
        self,
        agent: "BaseAgent",
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Called at start of EVERY round (before LLM call).

        A round is one LLM call + tool execution cycle.

        TIMING: Called every round, after context is built, before LLM call.
        FREQUENCY: Every round (1, 2, 3, ...).

        Use for:
        - Resetting per-round state
        - Injecting dynamic warnings/guidance
        - Modifying context based on current state
        - Adding loop detection warnings
        - Any per-round context modification

        Args:
            agent: Agent instance
            round_number: Current round number (1-indexed)
            context: Current context to be sent to LLM

        Returns:
            Modified context

        Default Implementation:
            Returns context unchanged (no-op).

        Example:
            ```python
            def on_round_start(self, agent, round_number, context):
                # Append warning at END (immediate feedback about recent rounds)
                if self.consecutive_empty_rounds > 2:
                    warning = "⚠️ Empty rounds detected - call a tool!"
                    context.append({"role": "user", "content": warning})

                self.round_start_time = time.time()
                return context
            ```
        """
        return context

    def on_llm_response(
        self,
        agent: "BaseAgent",
        response: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Called after LLM responds, before tool dispatch.

        TIMING: Called after LLM API call returns, before tools are dispatched.
        FREQUENCY: Every round.

        Use for:
        - Analyzing LLM response
        - Detecting empty rounds (no tool_calls)
        - Logging LLM behavior
        - Tracking reasoning patterns
        - Modifying/enriching response (e.g., parsing tool calls from content)

        Args:
            agent: Agent instance
            response: LLM response dict (contains 'content', 'tool_calls', etc.)

        Returns:
            Modified or unmodified response dict (MUST return response)

        Default Implementation:
            Returns response unchanged (no-op).

        Example:
            ```python
            def on_llm_response(self, agent, response):
                if not response.get('tool_calls'):
                    self.empty_round_count += 1
                    print(f"Empty round: LLM didn't call tools")
                return response  # MUST return response!
            ```
        """
        return response

    def on_tool_call(
        self,
        agent: "BaseAgent",
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any]
    ) -> None:
        """
        Called after each tool execution.

        TIMING: Called immediately after any tool is executed (regardless of
                which behavior owns it).
        FREQUENCY: Once per tool call (multiple times per round if LLM calls
                   multiple tools).

        Use for:
        - Tracking tool usage
        - Detecting loops (repeated tool calls)
        - Logging actions
        - Updating statistics
        - Recording action history

        Args:
            agent: Agent instance
            tool_name: Name of the tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool

        Default Implementation:
            Does nothing (no-op).

        Example:
            ```python
            def on_tool_call(self, agent, tool_name, args, result):
                self.action_count += 1
                if "error" in result:
                    self.error_count += 1
                print(f"Tool called: {tool_name}")
            ```
        """
        pass

    def on_round_end(self, agent: "BaseAgent", round_number: int) -> None:
        """
        Called at end of each round, after all tool calls.

        TIMING: Called after all tools in the round have been executed.
        FREQUENCY: Every round.

        Use for:
        - Rendering status displays
        - Persisting state
        - Checking timeout conditions
        - Updating progress tracking
        - Detecting empty rounds (no tools called)

        Args:
            agent: Agent instance
            round_number: Current round number (1-indexed)

        Default Implementation:
            Does nothing (no-op).

        Example:
            ```python
            def on_round_end(self, agent, round_number):
                if round_number % 5 == 0:
                    print(f"Round {round_number}: {self.action_count} actions")
                self.persist_state()
            ```
        """
        pass

    def on_timeout(self, agent: "BaseAgent", elapsed_seconds: float) -> None:
        """
        Called when goal times out.

        TIMING: Called when the agent exceeds its time budget for a goal.
        FREQUENCY: Once per goal (only if timeout occurs).

        Use for:
        - Generating timeout summaries
        - Saving partial progress
        - Cleaning up resources
        - Logging timeout reason

        Args:
            agent: Agent instance
            elapsed_seconds: Time elapsed since goal start

        Default Implementation:
            Calls deprecated on_timeout() with old signature for backwards
            compatibility.

        Example:
            ```python
            def on_timeout(self, agent, elapsed_seconds):
                print(f"Goal timed out after {elapsed_seconds:.1f}s")
                self.save_partial_results()
            ```
        """
        pass

    def on_goal_complete(
        self,
        agent: "BaseAgent",
        success: bool,
        summary: str
    ) -> None:
        """
        Called when goal completes (successfully or with failure).

        TIMING: Called when the agent finishes a goal (either successfully
                or with failure).
        FREQUENCY: Once per goal.

        Use for:
        - Generating completion summaries
        - Cleaning up resources
        - Saving final results
        - Logging completion status
        - Passing summaries to parent agent

        Args:
            agent: Agent instance
            success: True if goal completed successfully, False if failed
            summary: Summary message (from mark_complete or mark_failed)

        Default Implementation:
            Does nothing (no-op).

        Example:
            ```python
            def on_goal_complete(self, agent, success, summary):
                status = "SUCCESS" if success else "FAILED"
                elapsed = time.time() - self.start_time
                print(f"{status}: {self.action_count} actions in {elapsed:.1f}s")
            ```
        """
        pass

    def on_custom_event(
        self,
        agent: "BaseAgent",
        event_name: str,
        **event_data
    ) -> None:
        """
        Called when a custom event is fired by another behavior.

        TIMING: Called whenever agent.event_system.fire_custom_event() is invoked.
        FREQUENCY: Variable - depends on behavior interactions.

        Custom events enable inter-behavior communication without tight coupling.
        Behaviors can broadcast events to coordinate state changes, mode transitions,
        or other cross-cutting concerns.

        Use for:
        - Mode coordination (e.g., "mode_changed" events)
        - Context state changes (e.g., "context_full", "context_compacted")
        - Resource notifications (e.g., "workspace_changed", "file_created")
        - Custom coordination patterns between behaviors

        Args:
            agent: Agent instance
            event_name: Name of the custom event (e.g., "mode_changed")
            **event_data: Event-specific data passed by the event source

        Default Implementation:
            Does nothing (no-op). Behaviors opt-in to events they care about.

        Example:
            ```python
            def on_custom_event(self, agent, event_name, **event_data):
                # Listen for mode change events
                if event_name == "mode_changed":
                    new_mode = event_data.get("mode")
                    reason = event_data.get("reason", "unknown")
                    print(f"Mode changed to {new_mode}: {reason}")
                    self.current_mode = new_mode

                # Listen for context state events
                elif event_name == "context_full":
                    tokens_used = event_data.get("tokens_used", 0)
                    self.prepare_for_compaction(tokens_used)

            # To fire a custom event from another behavior:
            # agent.event_system.fire_custom_event(
            #     "mode_changed",
            #     mode="architect",
            #     reason="Design phase started"
            # )
            ```
        """
        pass

    # =================================================================
    # CONTEXT HELPER METHODS
    # =================================================================

    def inject_message_after_system(
        self,
        context: list[dict[str, Any]],
        message: str,
        role: str = "system"
    ) -> list[dict[str, Any]]:
        """
        Inject a message after the system prompt (position 1).

        Use for framework instructions that need to appear early in context
        (tool docs, delegation info, format examples).

        Args:
            context: Current context (list of message dicts)
            message: Message content to inject
            role: Message role - "system" for framework (default),
                  "user" for actual user messages

        Returns:
            Modified context with message injected

        Examples:
            # Tool documentation (system - default)
            self.inject_message_after_system(context, tool_docs)

            # Goal from user (user - explicit)
            self.inject_message_after_system(context, f"GOAL: {goal}", role="user")
        """
        if len(context) > 0:
            context.insert(1, {
                "role": role,
                "content": message
            })
        return context

    def append_message(
        self,
        context: list[dict[str, Any]],
        message: str,
        role: str = "system"
    ) -> list[dict[str, Any]]:
        """
        Append a message to the end of context.

        Use for nudges, warnings, and reminders that should appear at the
        end of context (near where the issue is occurring).

        Args:
            context: Current context
            message: Message content to append
            role: Message role - "system" for framework nudges (default),
                  "user" for actual user messages

        Returns:
            Modified context with message appended

        Examples:
            # Loop warning (system - default)
            self.append_message(context, loop_warning)

            # Chat mode instructions (user - explicit)
            self.append_message(context, chat_instructions, role="user")
        """
        context.append({
            "role": role,
            "content": message
        })
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
        agent: "BaseAgent",
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle tool call for this behavior's tools.

        Called by the agent when the LLM requests to use one of this
        behavior's tools. The behavior should execute the tool and
        return the result.

        Args:
            agent: Agent instance (access agent.workspace, agent.state, etc.)
            tool_name: Name of the tool being called
            args: Tool arguments (parsed from LLM's function call)

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
            def dispatch_tool(self, agent, tool_name, args):
                if tool_name == "read_file":
                    path = args["path"]
                    try:
                        # Can access agent.workspace directly
                        full_path = agent.workspace / path
                        with open(full_path) as f:
                            content = f.read()
                        return {"content": content, "success": True}
                    except Exception as e:
                        return {"error": str(e)}

                # Fall through to parent for unknown tools
                return super().dispatch_tool(agent, tool_name, args)
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


# Set default rule_of_two_properties after class definition
# Default to {A, B, C} for maximum safety (requires explicit override)
if RuleOfTwoProperty is not None:
    AgentBehavior.rule_of_two_properties = {
        RuleOfTwoProperty.UNTRUSTED_INPUT,
        RuleOfTwoProperty.SENSITIVE_ACCESS,
        RuleOfTwoProperty.EXTERNAL_ACTION
    }
