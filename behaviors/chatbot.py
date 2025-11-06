"""
ChatbotBehavior - Enables interactive chat mode when no goal is provided.

This behavior allows agents to enter chat mode when invoked without a goal:
- Activated when NO goal provided to the agent
- Provides set_goal tool for transitioning to execution mode
- Manages chat mode state

Key features:
- Provides set_goal and clarify_with_user tools
- Injects chat instructions once when chat mode activates
- Transitions to execution mode when set_goal is called
- Works with any agent type (TaskExecutor, Orchestrator, Architect)

COMPOSITION:
- This behavior does NOT handle execution mode (core BaseAgent functionality)
- This behavior does NOT handle delegation (use DelegationBehavior)
- This behavior does NOT implement its own chat loop (agent handles input/output)
- This behavior ONLY manages chat mode state and provides chat tools

Usage:
    When agent runs without a goal, this behavior:
    1. Provides set_goal tool
    2. Injects chat instructions once
    3. Waits for agent to call set_goal
    4. Transitions to execution mode when set_goal is called
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
    - Provides set_goal tool for transitioning to execution mode
    - Conversational interface for clarifying ambiguous requests
    - Supports multi-turn conversations to refine requirements

    This behavior is COMPOSABLE:
    - Does NOT handle execution (core BaseAgent functionality)
    - Does NOT handle delegation (delegate to DelegationBehavior)
    - Does NOT implement its own chat loop (agent handles I/O)
    - ONLY manages chat mode state and provides chat tools
    """

    def __init__(self):
        """Initialize chatbot behavior."""
        self.chat_mode_active = False
        self.chat_instructions_injected = False  # Track if chat instructions already injected

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

            # Check if agent.goal is set (core agent functionality)
            if hasattr(self.agent, 'goal') and self.agent.goal:
                # Goal set - don't provide chatbot tools
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
            },
            {
                "type": "function",
                "function": {
                    "name": "clarify_with_user",
                    "description": "Ask the user a clarifying question. The question will be displayed in the assistant's response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask the user"
                            }
                        },
                        "required": ["question"]
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

            # Transition from chat mode to execution mode
            self.chat_mode_active = False

            # Trigger onGoalSet event (BaseAgent will handle core initialization)
            if agent:
                # Fire onGoalSet event to all behaviors
                for behavior in agent.behaviors:
                    if hasattr(behavior, 'onGoalSet'):
                        behavior.onGoalSet(
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

        elif tool_name == "clarify_with_user":
            # Question already displayed in assistant message content
            # Just acknowledge internally
            question = args.get('question', '')
            return {
                "success": True,
                "message": "Question posed to user",
                "question": question
            }

        return super().dispatch_tool(agent, tool_name, args)

    def on_round_start(
        self,
        agent: Any,
        round_number: int,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject chat mode instructions once when chat mode becomes active.

        Args:
            agent: Agent instance
            round_number: Current round number
            context: Current context

        Returns:
            Modified context with chat instructions
        """
        # Inject chat instructions ONCE when chat mode first becomes active
        if self.chat_mode_active and not self.chat_instructions_injected:
            chat_instructions = """CHAT MODE ACTIVE:

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
            # Append to messages (not inject after system)
            context.append({
                "role": "user",
                "content": chat_instructions
            })
            self.chat_instructions_injected = True

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
