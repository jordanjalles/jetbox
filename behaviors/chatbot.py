"""
ChatbotBehavior - Enables interactive chat mode when no goal is provided.

This behavior allows agents to enter chat mode when invoked without a goal:
- Activated when NO goal provided to the agent
- Enters interactive conversation loop with user
- Extracts requirements and forms goal from conversation
- Transitions to execution mode once goal is clear

Key features:
- Interactive chat loop (input/output)
- Requirements extraction from conversation
- Goal formation and transition to execution
- Works with any agent type (TaskExecutor, Orchestrator, Architect)

COMPOSITION:
- This behavior does NOT handle execution mode (use SubAgentModeBehavior)
- This behavior does NOT handle delegation (use DelegationBehavior)
- This behavior ONLY manages interactive chat mode and goal extraction

Usage:
    When agent runs without a goal, this behavior:
    1. Enters chat loop
    2. Converses with user to clarify requirements
    3. Forms a clear goal statement
    4. Triggers on_goal_set event to transition to execution mode
"""

from typing import Any
from behaviors.base import AgentBehavior


class ChatbotBehavior(AgentBehavior):
    """
    Behavior that enables interactive chat mode for requirement gathering.

    Designed for agents that can be invoked without a goal:
    - CLI: python agent.py (no goal argument)
    - Tool call: agent.run() (no goal parameter)

    Features:
    - Interactive chat loop for requirement gathering
    - Provides set_goal tool for transitioning to execution mode
    - Conversational interface for clarifying ambiguous requests
    - Supports multi-turn conversations to refine requirements

    This behavior is COMPOSABLE:
    - Does NOT handle execution (delegate to SubAgentModeBehavior)
    - Does NOT handle delegation (delegate to DelegationBehavior)
    - ONLY manages chat mode and goal extraction
    """

    def __init__(self):
        """Initialize chatbot behavior."""
        self.chat_mode_active = False
        self.conversation_history: list[dict[str, Any]] = []

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "chatbot"

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
            # Check via context_manager if available
            if hasattr(self.agent, 'context_manager') and self.agent.context_manager:
                if self.agent.context_manager.state.goal:
                    # Goal already set - don't provide chatbot tools
                    return []

            # Check if SubAgentModeBehavior has set a goal (for delegated agents)
            for behavior in self.agent._behaviors:
                if behavior.get_name() in ['subagent_mode', 'subagent_context']:
                    if hasattr(behavior, 'goal') and behavior.goal:
                        # Goal set by delegation - don't provide chatbot tools
                        return []

        # No goal set - provide set_goal tool for chat mode
        return [
            {
                "type": "function",
                "function": {
                    "name": "set_goal",
                    "description": "Set a clear goal and transition from chat mode to execution mode. Use this when requirements are clear.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "Clear, concise goal statement (1-2 sentences)"
                            },
                            "requirements": {
                                "type": "string",
                                "description": "Detailed requirements gathered from conversation"
                            }
                        },
                        "required": ["goal"]
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
            **kwargs: Additional context (agent)

        Returns:
            Tool result dict
        """
        agent = kwargs.get('agent')

        if tool_name == "set_goal":
            goal = args.get('goal', '')
            requirements = args.get('requirements', '')

            if not goal:
                return {
                    "success": False,
                    "error": "Goal cannot be empty"
                }

            # Transition from chat mode to execution mode
            self.chat_mode_active = False

            # Trigger on_goal_set event (SubAgentModeBehavior will handle initialization)
            if agent:
                # Fire on_goal_set event to all behaviors
                for behavior in agent.behaviors:
                    if hasattr(behavior, 'on_goal_set'):
                        behavior.on_goal_set(
                            agent=agent,
                            goal=goal,
                            workspace=None,  # Create new workspace
                        )

            return {
                "success": True,
                "result": f"Goal set: {goal}",
                "goal": goal,
                "requirements": requirements,
                "mode_transition": "chat → execution"
            }

        return super().dispatch_tool(tool_name, args, **kwargs)

    def enhance_context(
        self,
        context: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Enhance context with chat mode instructions.

        Args:
            context: Current context (system + messages)
            **kwargs: Additional context

        Returns:
            Modified context with chat mode info
        """
        if self.chat_mode_active:
            # Insert chat mode instructions after system prompt
            chat_instructions = {
                "role": "user",
                "content": """CHAT MODE ACTIVE:

No goal has been provided yet. Your job is to:
1. Engage in conversation with the user
2. Ask clarifying questions to understand their requirements
3. Extract a clear goal from the conversation
4. Call set_goal(goal, requirements) when you have sufficient clarity

Guidelines:
- Be conversational and helpful
- Ask questions to clarify ambiguous requests
- Suggest approaches based on user's needs
- When requirements are clear, form a concise goal statement
- Use set_goal tool to transition to execution mode
"""
            }

            if len(context) > 0:
                context.insert(1, chat_instructions)

        return context

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
            # Check via context_manager if available
            if hasattr(self.agent, 'context_manager') and self.agent.context_manager:
                if self.agent.context_manager.state.goal:
                    # Goal already set - don't provide chat instructions
                    return ""

            # Check if SubAgentModeBehavior has set a goal (for delegated agents)
            for behavior in self.agent._behaviors:
                if behavior.get_name() in ['subagent_mode', 'subagent_context']:
                    if hasattr(behavior, 'goal') and behavior.goal:
                        # Goal set by delegation - don't provide chat instructions
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

    def on_agent_start(self, agent: Any, **kwargs: Any) -> None:
        """
        Handle agent start event.

        Check if agent has a goal. If not, activate chat mode.

        Args:
            agent: Agent instance
            **kwargs: Additional context
        """
        # Check if agent has a goal already set
        has_goal = False

        if hasattr(agent, 'context_manager') and agent.context_manager:
            if hasattr(agent.context_manager, 'state') and agent.context_manager.state.goal:
                has_goal = True

        # Check if SubAgentModeBehavior has set a goal (for delegated agents)
        if not has_goal and hasattr(agent, '_behaviors'):
            for behavior in agent._behaviors:
                if behavior.get_name() in ['subagent_mode', 'subagent_context']:
                    if hasattr(behavior, 'goal') and behavior.goal:
                        has_goal = True
                        break

        # Activate chat mode if no goal
        if not has_goal:
            self.chat_mode_active = True
            print("[chatbot] Chat mode activated (no goal provided)")
        else:
            self.chat_mode_active = False
            print("[chatbot] Execution mode (goal already set)")

    def run_chat_loop(self, agent: Any) -> str | None:
        """
        Run interactive chat loop until goal is set.

        This is a blocking function that:
        1. Enters chat loop
        2. Gets user input
        3. Calls LLM with conversation history
        4. Displays responses
        5. Exits when set_goal is called

        Args:
            agent: Agent instance with LLM capabilities

        Returns:
            Goal string if extracted, None if chat ended without goal
        """
        print("\n" + "=" * 60)
        print("CHAT MODE")
        print("=" * 60)
        print("Hi! I'm ready to help. What would you like to work on?")
        print("(Type 'quit' or 'exit' to end chat)")
        print()

        goal_extracted = None

        while self.chat_mode_active:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nEnding chat session...")
                    break

                # Add user message to agent history
                agent.add_message({
                    "role": "user",
                    "content": user_input
                })

                # Call LLM
                response = agent.call_llm(
                    model=agent.config.llm.model if hasattr(agent, 'config') else "gpt-oss:20b",
                    temperature=0.7,  # Higher temp for more natural conversation
                )

                if "message" in response:
                    msg = response["message"]

                    # Show content if present
                    if msg.get("content"):
                        print(f"\nAgent: {msg['content']}\n")

                    # Handle tool calls
                    if "tool_calls" in msg and msg["tool_calls"]:
                        for tool_call in msg["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            args = tool_call["function"]["arguments"]

                            # Dispatch tool call
                            result = agent.dispatch_tool(tool_call)

                            # Check if set_goal was called
                            if tool_name == "set_goal" and result.get("success"):
                                goal_extracted = result.get("goal")
                                print(f"\n✅ Goal extracted: {goal_extracted}")
                                print("Transitioning to execution mode...")
                                return goal_extracted

                            # Add tool result to history
                            agent.add_message({
                                "role": "tool",
                                "content": str(result)
                            })

            except KeyboardInterrupt:
                print("\n\nChat interrupted. Exiting...")
                break
            except Exception as e:
                print(f"\nError in chat loop: {e}")
                import traceback
                traceback.print_exc()
                break

        return goal_extracted

    def run_orchestrator_chat_loop(
        self,
        agent: Any,
        execute_task_callback: Any,
        initial_message: str | None = None
    ) -> None:
        """
        Run interactive chat loop for orchestrator (multi-task mode).

        Unlike run_chat_loop which extracts a single goal and exits,
        this loop continues indefinitely, allowing multiple tasks in sequence.

        Each user input is treated as a task request that gets executed immediately
        via the execute_task_callback, then returns to prompt for next task.

        Args:
            agent: Orchestrator agent instance
            execute_task_callback: Function(user_input) that executes a task and returns when done
            initial_message: Optional first message to execute before entering loop

        Workflow:
            1. User enters task description
            2. Callback executes task (delegates, waits for completion)
            3. Show completion message
            4. Return to prompt for next task
            5. Repeat until user types 'quit'
        """
        # Execute initial message if provided
        if initial_message:
            execute_task_callback(initial_message)
            print("\n✅ Task completed. Ready for next request.\n")

        # Interactive loop
        print("=" * 60)
        print("ORCHESTRATOR CHAT MODE")
        print("=" * 60)
        print("Enter task descriptions and I'll execute them.")
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

                # Show completion
                print("\n✅ Task completed. Ready for next request.\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Shutting down...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                print("\nReturning to prompt...\n")
