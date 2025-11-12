"""
ChatbotBehavior - Manages chat mode for conversational interaction.

This behavior enables interactive chat mode using event-based mode coordination:
- Activates when: Agent starts (default) or clarify_with_user() called
- Deactivates when: set_goal() called (execution mode activates)
- Provides tools for mode transition and clarification
- Coordinates with ExecutionModeBehavior via events

Key features:
- self.is_active state tracking (starts active by default)
- activate() / deactivate() lifecycle methods
- Event-based coordination via on_custom_event()
- Provides set_goal, clarify_with_user, activate_chat_mode tools
- Broadcasts 'mode_activated' event when activating
- Automatically deactivates when execution mode activates

COMPOSITION:
- This behavior does NOT handle execution mode (use ExecutionModeBehavior)
- This behavior does NOT handle delegation (use DelegationBehavior)
- This behavior does NOT implement its own chat loop (agent handles input/output)
- This behavior ONLY manages chat mode state and provides chat tools

Configuration:
- params: Optional configuration dict (currently unused)

Usage:
    When agent runs without a goal, this behavior:
    1. Starts active (is_active = True)
    2. Provides set_goal, clarify_with_user, activate_chat_mode tools
    3. Waits for agent to call set_goal
    4. When set_goal called:
       - Sets agent.goal
       - Finds ExecutionModeBehavior and calls activate()
       - ExecutionModeBehavior fires 'mode_activated' event
       - ChatbotBehavior hears event and deactivates
    5. Mode transition complete (chat → execution)
"""

from typing import Any
from behaviors.base import AgentBehavior


class ChatbotBehavior(AgentBehavior):
    """
    Manages chat mode for conversational interaction using event-based coordination.

    Mode State:
    - self.is_active: Tracks whether chat mode is currently active (default: True)
    - Starts active by default when agent has no goal
    - Deactivates automatically when execution mode activates

    Mode Lifecycle:
    - activate(): Sets is_active=True, broadcasts 'mode_activated' event
    - deactivate(): Sets is_active=False, appends deactivation message

    Event Coordination:
    - Listens for 'mode_activated' events from ExecutionModeBehavior
    - Automatically deactivates when execution mode activates
    - No coupling between behaviors (event-based)

    Tools Provided:
    - set_goal: Activates execution mode (finds ExecutionModeBehavior.activate())
    - clarify_with_user: Ensures chat mode is active
    - activate_chat_mode: Explicitly activates chat mode

    This behavior is COMPOSABLE:
    - Does NOT handle execution (use ExecutionModeBehavior)
    - Does NOT handle delegation (use DelegationBehavior)
    - Does NOT implement its own chat loop (agent handles I/O)
    - ONLY manages chat mode state and provides chat tools

    Security: [] None (utility behavior, no security properties)
    """

    # Rule of Two: Empty (utility behavior for chat mode management)
    rule_of_two_properties = set()

    def __init__(self, params: dict[str, Any] | None = None):
        """Initialize chatbot behavior."""
        super().__init__()
        self.params = params or {}

        # Mode state (owned by behavior) - starts active by default
        self.is_active = True

        # Legacy tracking for transition compatibility
        self.chat_instructions_injected = False  # Track if chat instructions already injected

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "chatbot"

    # ============================================
    # Mode Lifecycle
    # ============================================

    def activate(self, agent: Any, **context: Any) -> dict[str, Any]:
        """
        Activate chat mode.

        Called when:
        - Agent starts (default mode)
        - clarify_with_user() tool is called
        - activate_chat_mode() tool is called
        - Execution mode completes

        Args:
            agent: Agent instance
            **context: Optional context data

        Returns:
            Activation result dict
        """
        if self.is_active:
            return {"already_active": True}

        self.is_active = True

        # Append mode activation message
        activation_message = """
╔════════════════════════════════════════════════════════╗
║  💬 CHAT MODE ACTIVATED                                ║
╚════════════════════════════════════════════════════════╝

CHAT MODE GUIDELINES:
- Respond conversationally (text-only responses are fine)
- Be helpful and friendly
- Ask clarifying questions if needed
- Tools are available but completely optional

To start working on a task: call set_goal(goal, requirements)

How can I help you?
"""

        agent.add_message({"role": "user", "content": activation_message})

        # Broadcast mode activation event
        if hasattr(agent, 'event_system'):
            agent.event_system.fire_custom_event(
                'mode_activated',
                mode_name='chat',
                behavior=self
            )

        return {
            "success": True,
            "mode": "chat",
            "active": True
        }

    def deactivate(self, agent: Any, reason: str = "transition") -> dict[str, Any]:
        """
        Deactivate chat mode.

        Called when execution mode activates.

        Args:
            agent: Agent instance
            reason: Reason for deactivation ("transition", "conflict", etc.)

        Returns:
            Deactivation result dict
        """
        if not self.is_active:
            return {"already_inactive": True}

        self.is_active = False

        # Append deactivation message
        if reason == "conflict":
            message = """
╔════════════════════════════════════════════════════════╗
║  💬 CHAT MODE → EXECUTION MODE                         ║
╚════════════════════════════════════════════════════════╝

Transitioning to execution mode to work on task...
"""
        else:
            message = f"💬 Chat mode deactivated ({reason})"

        agent.add_message({"role": "user", "content": message})

        return {
            "success": True,
            "mode": "chat",
            "active": False,
            "reason": reason
        }

    # ============================================
    # Tools
    # ============================================

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Provide goal extraction tool for chat mode.

        Only provides set_goal tool if agent doesn't already have a goal.
        This prevents confusion when agent is invoked with a goal parameter.

        Returns:
            Tool definitions for set_goal (used to transition from chat to execution)
        """
        # Check if agent already has a goal set
        # If yes, don't provide set_goal tool (agent is in execution mode, not chat mode)
        if hasattr(self, 'agent') and self.agent:
            # Check if agent.goal is set (core agent functionality)
            if hasattr(self.agent, 'goal') and self.agent.goal:
                # Goal set - don't provide chatbot tools
                return []

        # No goal set - provide chat mode tools
        return [
            {
                "type": "function",
                "function": {
                    "name": "set_goal",
                    "description": "Start working on a task. This activates execution mode. Call this when user wants to build something and you have basic requirements. Be decisive - don't wait for perfect clarity. If user says 'do it', 'build it', 'go ahead', call this IMMEDIATELY.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "Clear goal statement describing what to build (e.g., 'Build a colorful Mandelbrot fractal renderer using HTML5 and WebGPU')"
                            },
                            "requirements": {
                                "type": "string",
                                "description": "Requirements details (optional - can be brief or left empty if minimal info available)"
                            }
                        },
                        "required": ["goal"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clarify_with_user",
                    "description": "Ask user for clarification. Ensures chat mode is active for conversational interaction.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to ask user"
                            }
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "activate_chat_mode",
                    "description": "Explicitly activate chat mode. Usually not needed as chat mode is default, but can be used to return to conversational mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle tool calls for this behavior.

        Args:
            agent: Agent instance
            tool_name: Tool being called
            args: Tool arguments

        Returns:
            Tool result dict
        """

        if tool_name == "set_goal":
            goal = args.get('goal', '')
            requirements = args.get('requirements', '')

            if not goal:
                return {
                    "success": False,
                    "error": "Goal cannot be empty"
                }

            # Build full goal
            full_goal = goal
            if requirements:
                full_goal += f"\n\nRequirements:\n{requirements}"

            # Set goal on agent (core functionality)
            # Note: agent.set_goal() automatically activates ExecutionModeBehavior
            agent.set_goal(full_goal)

            return {
                "success": True,
                "result": f"Goal set: {goal}",
                "goal": goal,
                "requirements": requirements,
                "mode_transition": "chat → execution"
            }

        elif tool_name == "clarify_with_user":
            question = args.get('question', '')

            # Ensure chat mode is active
            if not self.is_active:
                self.activate(agent)

            return {
                "success": True,
                "result": f"Question for user: {question}",
                "question": question,
                "mode": "chat"
            }

        elif tool_name == "activate_chat_mode":
            return self.activate(agent)

        return super().dispatch_tool(agent, tool_name, args)

    # ============================================
    # Initial Context (Static, KV Cached)
    # ============================================

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Explain chat mode in initial context (STATIC, KV cached).

        This is part of the system prompt that never changes.
        Explains what chat mode IS and guidelines.

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Modified context with chat mode explanation
        """
        mode_explanation = f"""
═══════════════════════════════════════════════════════════
CHAT MODE (default conversational mode)
═══════════════════════════════════════════════════════════

Purpose: Natural conversational interaction with user

Guidelines when chat mode is active:
- Respond conversationally to user messages
- Text-only responses are perfectly fine
- Be helpful, friendly, and concise
- Ask clarifying questions when needed
- Tools are available but completely optional

Transitioning to execution mode: call set_goal(goal, requirements)

Status: {"💬 ACTIVE" if self.is_active else "⏸️  INACTIVE"}
"""

        return self.inject_user_message_after_system(context, mode_explanation)

    # ============================================
    # Event Handlers
    # ============================================

    def on_custom_event(
        self,
        agent: Any,
        event_name: str,
        **event_data: Any
    ) -> None:
        """
        Listen for mode activation events from other behaviors.

        If execution mode activates, deactivate chat mode.

        Args:
            agent: Agent instance
            event_name: Name of the event
            **event_data: Event data
        """
        if event_name == "mode_activated":
            activated_mode = event_data.get('mode_name')

            # If execution mode activates and we're active, deactivate
            if activated_mode == "execution" and self.is_active:
                self.deactivate(agent, reason="conflict")

    # ============================================
    # Runtime Hooks (Legacy)
    # ============================================

    def on_round_start(
        self,
        agent: Any,
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        LEGACY: Inject chat mode instructions once when chat mode becomes active.

        NOTE: This is legacy behavior kept for backward compatibility.
        The new event-based system injects activation messages via activate().

        Args:
            agent: Agent instance
            round_number: Current round number
            context: Current context

        Returns:
            Modified context with chat instructions
        """
        # Inject chat instructions ONCE when chat mode first becomes active
        if self.is_active and not self.chat_instructions_injected:
            chat_instructions = """CHAT MODE ACTIVE:

No goal has been provided yet. You can have a normal conversation with the user.

**How to respond:**
- If user is just chatting or asking general questions → Respond naturally (no tools needed)
- If user wants to build/create something → Ask 1-2 clarifying questions MAX, then call set_goal
- If user says "do it", "build it", "go", "start", etc. → IMMEDIATELY call set_goal
- If user repeats their request or gets impatient → Stop asking questions and call set_goal NOW

**Available tools:**
- set_goal: Call this when you're ready to start working on a task (be decisive!)
- clarify_with_user: Rarely needed - only for critical ambiguity

Guidelines:
- Be decisive - don't over-clarify or ask too many questions
- If you can build something reasonable with current info, DO IT
- User confirmation ("build it", "do it") means START IMMEDIATELY
- When in doubt, make reasonable assumptions and call set_goal
"""
            # Append to messages (not inject after system)
            context.append({
                "role": "user",
                "content": chat_instructions
            })
            self.chat_instructions_injected = True

        return context

    def run_chat_loop(
        self,
        agent: Any,
        execute_task_callback: Any,
        initial_message: str | None = None
    ) -> None:
        """
        OPTIONAL CONVENIENCE METHOD: Provide default CLI chat loop implementation.

        This is NOT the core responsibility of ChatbotBehavior (which is tools + hooks).
        This is a convenience method for quick CLI setup. Advanced users can:
        - Use this for simple CLI chat mode
        - Implement custom loop using this behavior's tools
        - Ignore this entirely in non-CLI contexts (web, API, Discord)

        Args:
            agent: Agent instance
            execute_task_callback: Function(task_description) that executes a task
            initial_message: Optional first message to execute before entering loop

        Workflow:
            1. Execute initial message if provided
            2. Show prompt and get user input
            3. Execute via callback
            4. Return to prompt for next task
            5. Repeat until user types 'quit'
        """
        # Execute initial message if provided
        if initial_message:
            execute_task_callback(initial_message)
            # No completion message for initial - response is inline

        # Interactive loop
        agent_name = getattr(agent, 'name', 'Agent').replace('_', ' ').title()
        print("\n" + "=" * 60)
        print(f"{agent_name}")
        print("=" * 60)
        print("Chat mode - ask me anything!")
        print("(Type 'quit' or 'exit' to end session)")
        print()

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nShutting down...")
                    break

                # Execute task via callback
                execute_task_callback(user_input)

                # No completion message - response is inline
                print()  # Just add spacing

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Shutting down...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                print("Ready for next request.\n")

    def get_instructions(self) -> str:
        """
        Return chat mode workflow instructions.

        Only provides instructions if agent doesn't already have a goal.
        This prevents confusing instructions when agent is in execution mode.

        Returns:
            Instructions for interactive chat mode (or empty string if goal set)
        """
        # Check if agent already has a goal set
        # If yes, don't provide chat mode instructions (agent is in execution mode)
        if hasattr(self, 'agent') and self.agent:
            # Check if agent.goal is set (core agent functionality)
            if hasattr(self.agent, 'goal') and self.agent.goal:
                # Goal set - don't provide chat instructions
                return ""

        # No goal set - provide chat mode instructions
        return """
CHAT MODE:
When the agent is invoked without a goal, you enter chat mode to gather requirements.

Workflow:
1. Greet the user and ask how you can help
2. Engage in conversation to clarify their needs
3. Ask specific questions about:
   - What they want to build/accomplish
   - Technical requirements or constraints
   - Preferred technologies or approaches
4. When requirements are clear, form a goal statement
5. Call set_goal(goal, requirements) to transition to execution mode

Example conversation:
User: "I need help with a project"
Agent: "I'd be happy to help! What kind of project are you working on?"
User: "A web scraper for news articles"
Agent: "Great! A few questions: What news sites? What data do you need? Any specific format for output?"
User: "CNN and BBC. I need headlines, dates, and article URLs. JSON output would be good."
Agent: [calls set_goal(goal="Create a web scraper for CNN and BBC news articles", requirements="...")]

After set_goal is called, you will automatically transition to execution mode.
"""
