"""
Orchestrator agent - manages user conversation and delegates to TaskExecutor.

Uses append-until-full context management: keeps all messages until near token limit,
then performs compaction pass to summarize old messages.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import json

from base_agent import BaseAgent
from agent_config import config


class OrchestratorAgent(BaseAgent):
    """
    Agent specialized for user interaction and task delegation.

    Context strategy: Append until full, then compact
    Tools: Delegation, clarification, planning
    """

    def __init__(self, workspace: Path):
        """
        Initialize Orchestrator agent.

        Args:
            workspace: Working directory
        """
        super().__init__(
            name="orchestrator",
            role="User interface and task coordinator",
            workspace=workspace,
            config=config,
        )

        # Track delegated tasks
        self.delegated_tasks: list[dict[str, Any]] = []

        # Context compaction threshold (tokens)
        # Using max_tokens from config, compact at 80% full
        self.token_threshold = int(config.context.max_tokens * 0.8)

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tools available to Orchestrator.

        Tools:
        - delegate_to_executor: Send a task to TaskExecutor
        - clarify_with_user: Ask user for clarification
        - create_task_plan: Break down user request into tasks
        - get_executor_status: Check TaskExecutor progress
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_executor",
                    "description": "Delegate a coding task to the TaskExecutor agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Clear description of coding task to execute",
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context or requirements",
                            },
                            "workspace": {
                                "type": "string",
                                "description": "Optional: Path to existing workspace to continue work in. Use this when updating/modifying existing projects. Omit to create new isolated workspace.",
                            },
                        },
                        "required": ["task_description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "clarify_with_user",
                    "description": "Ask the user a clarifying question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to ask the user",
                            },
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_task_plan",
                    "description": "Create a structured plan for completing user request",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {"type": "string"},
                                        "estimated_complexity": {"type": "string"},
                                    },
                                },
                                "description": "List of tasks to complete",
                            },
                        },
                        "required": ["tasks"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_executor_status",
                    "description": "Check the current status of the TaskExecutor",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_workspaces",
                    "description": "List all existing workspaces in .agent_workspace/ directory",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]

    def get_system_prompt(self) -> str:
        """Return Orchestrator-specific system prompt."""
        return """You are an orchestrator agent that helps users plan and execute software projects.

Your workflow:
1. When user makes a request, check if you need clarification
2. If clear, IMMEDIATELY delegate to TaskExecutor using delegate_to_executor
3. For complex requests, you can optionally create_task_plan first, then delegate_to_executor
4. After delegation completes, report results back to user WITH FILE LOCATIONS

WORKSPACE BEHAVIOR (CRITICAL):
- By DEFAULT, TaskExecutor creates a NEW isolated workspace for each delegation
- New workspace path is derived from task description (e.g., "create calculator" → .agent_workspace/create-calculator/)
- When user says "update X" or "modify X" or "add to X", they mean work in EXISTING workspace
- To work in existing workspace, use the "workspace" parameter in delegate_to_executor
- Use list_workspaces tool to see all existing workspaces
- When user references existing project, ALWAYS check workspaces and specify correct one

WORKSPACE DECISION TREE:
1. User says "create/make/build NEW thing" → Omit workspace (creates new)
2. User says "update/modify/add to EXISTING thing" → Call list_workspaces, then specify workspace parameter
3. User says "fix bug in X" or "improve X" → Find X's workspace and specify it
4. Not sure? → Ask user or call list_workspaces to check

IMPORTANT RULES:
- Delegate ONCE per user request (unless user explicitly asks for more work)
- After delegation completes successfully, REPORT to user - do NOT delegate again
- When reporting completion, tell user WHERE files are located
- Do NOT delegate tasks to "retrieve" or "return" file contents - just tell user the workspace path
- If user wants to see file content, tell them the path so they can read it themselves

Guidelines:
- Be conversational and brief
- Ask clarifying questions ONLY if truly needed
- For simple requests (e.g., "create a file"), delegate immediately without planning
- For complex requests (e.g., "build a website"), you may create_task_plan then delegate_to_executor
- ALWAYS delegate to TaskExecutor for any coding work - don't just plan
- Don't write code yourself - delegate to TaskExecutor

Tools available:
- delegate_to_executor: Send coding tasks to TaskExecutor (workspace param optional - use for existing projects)
- list_workspaces: List all existing project workspaces
- clarify_with_user: Ask user questions (use sparingly)
- create_task_plan: Structure complex requests (optional, use before delegation)
- get_executor_status: Check TaskExecutor progress
"""

    def get_context_strategy(self) -> str:
        """Orchestrator uses append-until-full strategy."""
        return "append_until_full"

    def build_context(self) -> list[dict[str, Any]]:
        """
        Build context using append-until-full strategy.

        Strategy:
        1. Always include system prompt
        2. Append all messages until approaching token limit
        3. When near limit, compact old messages into summary

        Returns:
            [system_prompt, ...all_messages_or_compacted...]
        """
        context = [{"role": "system", "content": self.get_system_prompt()}]

        messages = self.state.messages

        # Estimate token count (rough: 1 token ≈ 4 chars)
        current_tokens = self._estimate_tokens(context + messages)

        # If under threshold, return all messages
        if current_tokens < self.token_threshold:
            context.extend(messages)
            return context

        # Otherwise, compact old messages
        compacted = self._compact_messages(messages)
        context.extend(compacted)
        return context

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """
        Estimate token count for messages.

        Uses rough heuristic: 1 token ≈ 4 characters

        Args:
            messages: List of message dicts

        Returns:
            Estimated token count
        """
        total_chars = 0
        for msg in messages:
            # Count content
            if "content" in msg:
                total_chars += len(str(msg["content"]))

            # Count tool calls
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total_chars += len(tc["function"]["name"])
                    total_chars += len(str(tc["function"]["arguments"]))

        return total_chars // 4

    def _compact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Compact old messages to reduce token usage.

        Strategy:
        1. Keep recent 20 messages intact (preserve recent context)
        2. Summarize older messages into a single summary message
        3. Return: [summary, ...recent_messages...]

        Args:
            messages: Full message history

        Returns:
            Compacted message list
        """
        if len(messages) <= 20:
            return messages

        # Split: old messages vs recent messages
        recent_count = 20
        old_messages = messages[:-recent_count]
        recent_messages = messages[-recent_count:]

        # Summarize old messages
        summary = self._summarize_messages(old_messages)

        # Return: summary + recent
        return [
            {
                "role": "user",
                "content": f"[Earlier conversation summary]\n{summary}\n[End of summary - recent messages follow]",
            }
        ] + recent_messages

    def _summarize_messages(self, messages: list[dict[str, Any]]) -> str:
        """
        Create a text summary of messages.

        Args:
            messages: Messages to summarize

        Returns:
            Summary string
        """
        summary_parts = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                summary_parts.append(f"User: {content[:100]}...")
            elif role == "assistant":
                # Check for tool calls
                if "tool_calls" in msg:
                    tools = [tc["function"]["name"] for tc in msg["tool_calls"]]
                    summary_parts.append(f"Assistant called: {', '.join(tools)}")
                elif content:
                    summary_parts.append(f"Assistant: {content[:100]}...")

        return "\n".join(summary_parts)

    def execute_round(self, model: str, temperature: float) -> dict[str, Any]:
        """
        Execute one round of orchestration.

        Returns:
            LLM response with tool calls
        """
        self.increment_round()
        response = self.call_llm(model, temperature)

        # Add assistant message to history
        if "message" in response:
            self.add_message(response["message"])

        return response

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to conversation.

        Args:
            content: User message content
        """
        self.add_message({"role": "user", "content": content})

    def get_conversation_summary(self) -> dict[str, Any]:
        """
        Get summary of current conversation.

        Returns:
            Dict with message count, token estimate, delegated tasks
        """
        messages = self.state.messages
        tokens = self._estimate_tokens(messages)

        return {
            "total_messages": len(messages),
            "estimated_tokens": tokens,
            "token_threshold": self.token_threshold,
            "delegated_tasks": len(self.delegated_tasks),
            "rounds": self.state.total_rounds,
        }
