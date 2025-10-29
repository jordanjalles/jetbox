"""
Orchestrator agent - manages user conversation and delegates to TaskExecutor.

Uses append-until-full context management: keeps all messages until near token limit,
then performs compaction pass to summarize old messages.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import os

from base_agent import BaseAgent
from agent_config import config
from llm_utils import chat_with_inactivity_timeout


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

        # Get model name for LLM calls
        self.model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

        # Get actual context window from model
        self.context_window = self._get_model_context_window()

        # Context compaction threshold (tokens)
        # Compact at 75% to account for system prompt overhead
        self.token_threshold = int(self.context_window * 0.75)

        print(f"[orchestrator] Model: {self.model}, Context window: {self.context_window} tokens, Threshold: {self.token_threshold} tokens")

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
                    "description": "Delegate a coding task to the TaskExecutor agent. IMPORTANT: You MUST specify workspace_mode to indicate whether this is new work or continuation of existing work.",
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
                            "workspace_mode": {
                                "type": "string",
                                "enum": ["new", "existing"],
                                "description": "REQUIRED: 'new' to create fresh isolated workspace, 'existing' to continue work in existing workspace. Use 'existing' when user says 'update X', 'add to X', 'modify X', 'fix X'. Use 'new' when user says 'create X', 'build X', 'make X'.",
                            },
                            "workspace_path": {
                                "type": "string",
                                "description": "REQUIRED when workspace_mode='existing': Path to existing workspace (use find_workspace tool to get this). MUST be omitted when workspace_mode='new'.",
                            },
                        },
                        "required": ["task_description", "workspace_mode"],
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
            {
                "type": "function",
                "function": {
                    "name": "find_workspace",
                    "description": "Find the best matching workspace for a given project reference (e.g., 'calculator', 'blog', 'todo app'). Use this when user says 'add to X' or 'update X' to find which workspace contains X.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Name or description of the project to find (e.g., 'calculator', 'blog post manager', 'todo')",
                            },
                        },
                        "required": ["project_name"],
                    },
                },
            },
        ]

    def get_system_prompt(self) -> str:
        """Return Orchestrator-specific system prompt."""
        return """You are an orchestrator agent that helps users plan and execute software projects.

Your workflow:
1. When user makes a request, check if you need clarification
2. Determine if this is NEW work or EXISTING work (MANDATORY STEP)
3. Delegate to TaskExecutor with explicit workspace_mode
4. After delegation completes, report results back to user WITH FILE LOCATIONS

WORKSPACE MODE (CRITICAL - YOU MUST ALWAYS SPECIFY):
The delegate_to_executor tool REQUIRES a workspace_mode parameter with value "new" or "existing".

DECISION PROCESS - YOU MUST FOLLOW THIS:

Step 1: Analyze the user request for keywords:
- NEW work indicators: "create", "make", "build", "new", "start"
- EXISTING work indicators: "update", "modify", "add to", "change", "fix", "improve", "enhance"

Step 2: Choose workspace_mode:
- If NEW work → workspace_mode="new" (no workspace_path needed)
- If EXISTING work → MUST call find_workspace first, then workspace_mode="existing" with workspace_path

Step 3: Delegate with correct parameters

EXAMPLE WORKFLOWS:

Example 1 - NEW PROJECT:
User: "create a calculator package"
→ Analyze: "create" = NEW work
→ delegate_to_executor(
    task_description="create a calculator package",
    workspace_mode="new"
  )

Example 2 - UPDATE EXISTING:
User: "add a square root function to the calculator"
→ Analyze: "add to" = EXISTING work
→ find_workspace(project_name="calculator")
→ [tool returns workspace_path: ".agent_workspace/create-calculator"]
→ delegate_to_executor(
    task_description="add a square root function to the calculator",
    workspace_mode="existing",
    workspace_path=".agent_workspace/create-calculator"
  )

Example 3 - AMBIGUOUS (user just says "calculator"):
User: "add square root to calculator"
→ Analyze: "add to" = EXISTING work, but which calculator?
→ find_workspace(project_name="calculator")
→ If found: use existing workspace
→ If not found: ask user or use workspace_mode="new"

MANDATORY RULES:
1. ALWAYS specify workspace_mode in delegate_to_executor - no exceptions
2. If workspace_mode="existing", MUST provide workspace_path
3. To get workspace_path, MUST call find_workspace first
4. If workspace_mode="new", do NOT provide workspace_path
5. When user references existing project by name, ALWAYS use find_workspace
6. Delegate ONCE per user request (unless user explicitly asks for more work)
7. After delegation completes, REPORT file locations to user

REPORTING RESULTS:
- Tell user WHERE files are located (workspace path)
- Do NOT delegate again just to "retrieve" or "show" files
- If user wants file content, tell them the path

Guidelines:
- Be conversational and brief
- Ask clarifying questions ONLY if truly needed
- For simple requests, delegate immediately without planning
- For complex requests, you may create_task_plan first
- ALWAYS delegate to TaskExecutor for coding work

Tools available:
- delegate_to_executor: Send coding tasks (REQUIRES workspace_mode: "new" or "existing")
- find_workspace: Find workspace by project name (REQUIRED before workspace_mode="existing")
- list_workspaces: List all existing workspaces
- clarify_with_user: Ask user questions
- create_task_plan: Structure complex requests (optional)
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
        4. CRITICAL: Update self.state.messages to compacted version

        Returns:
            [system_prompt, ...all_messages_or_compacted...]
        """
        context = [{"role": "system", "content": self.get_system_prompt()}]

        # Estimate token count (rough: 1 token ≈ 4 chars)
        current_tokens = self._estimate_tokens(context + self.state.messages)

        # If we need to compact, do it NOW and update state
        if current_tokens >= self.token_threshold:
            # Compact and UPDATE state.messages to prevent infinite growth
            compacted = self._compact_messages(self.state.messages)
            old_count = len(self.state.messages)
            self.state.messages = compacted
            new_count = len(self.state.messages)

            # Persist the compacted state
            self.persist_state()

            print(f"[context_compaction] Compacted messages: {old_count} → {new_count} (saved {old_count - new_count} messages)")

        # Now just use state.messages (which is compacted if needed)
        context.extend(self.state.messages)
        return context

    def _get_model_context_window(self) -> int:
        """
        Get actual context window size from Ollama model.

        Queries Ollama API for model metadata to get num_ctx parameter.
        Falls back to config value if query fails.

        Returns:
            Context window size in tokens
        """
        try:
            import ollama

            # Query model info
            model_info = ollama.show(self.model)

            # Extract num_ctx from model_info (in modelfile or parameters)
            if "modelfile" in model_info:
                # Parse modelfile for num_ctx
                modelfile = model_info["modelfile"]
                for line in modelfile.split("\n"):
                    if "num_ctx" in line.lower():
                        # Extract number (format: "PARAMETER num_ctx 8192")
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.lower() == "num_ctx" and i + 1 < len(parts):
                                return int(parts[i + 1])

            # Try parameters dict
            if "parameters" in model_info:
                params = model_info["parameters"]
                if "num_ctx" in params:
                    return int(params["num_ctx"])

            # Default from config
            print(
                "⚠️  [orchestrator] Could not find num_ctx in model info, using config default"
            )
            return config.context.max_tokens

        except Exception as e:
            print(
                f"⚠️  [orchestrator] Failed to query model context window: {e}, using config default"
            )
            return config.context.max_tokens

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:  # noqa: C901
        """
        Estimate token count for messages.

        Uses tiktoken if available, otherwise improved heuristic:
        - Code/structured data: chars/3 (denser)
        - Prose/natural text: chars/4 (sparser)

        Args:
            messages: List of message dicts

        Returns:
            Estimated token count
        """
        # Try tiktoken first (accurate)
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model("gpt-4")  # Use gpt-4 as proxy

            total_tokens = 0
            for msg in messages:
                # Count content
                if "content" in msg:
                    total_tokens += len(enc.encode(str(msg["content"])))

                # Count tool calls
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        total_tokens += len(enc.encode(tc["function"]["name"]))
                        total_tokens += len(
                            enc.encode(str(tc["function"]["arguments"]))
                        )

            return total_tokens

        except ImportError:
            # Fallback: Improved heuristic
            total_chars = 0
            code_chars = 0

            for msg in messages:
                # Count content
                if "content" in msg:
                    content = str(msg["content"])
                    total_chars += len(content)

                    # Detect code (contains braces, semicolons, indentation patterns)
                    if any(
                        marker in content for marker in ["{", "}", ";", "    ", "\t"]
                    ):
                        code_chars += len(content)

                # Tool calls are structured (like code)
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tool_str = tc["function"]["name"] + str(
                            tc["function"]["arguments"]
                        )
                        total_chars += len(tool_str)
                        code_chars += len(tool_str)

            # Estimate: code is denser (chars/3), prose is sparser (chars/4)
            prose_chars = total_chars - code_chars
            estimated_tokens = (code_chars // 3) + (prose_chars // 4)

            return estimated_tokens

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
        Summarize messages using LLM (preferred) or crude fallback.

        Tries LLM-driven summarization first for intelligent context preservation.
        Falls back to crude text truncation if LLM fails.

        Args:
            messages: Messages to summarize (usually 80-100 messages)

        Returns:
            Concise summary text (target: 150-200 words)
        """
        # Try LLM summarization first
        try:
            print("🔄 [compaction] Attempting LLM-driven summarization...")
            summary = self._summarize_with_llm(messages)

            if summary and len(summary) > 100:  # Sanity check
                print(f"✅ [compaction] LLM summary generated: {len(summary)} chars")
                return summary

        except Exception as e:
            print(f"❌ [compaction] LLM SUMMARIZATION FAILED: {e}")
            print("⚠️  [compaction] Falling back to crude text truncation")
            print(
                "⚠️  [compaction] This will preserve less context - consider checking LLM health"
            )

        # Fallback to crude method
        print("🔄 [compaction] Using crude summarization fallback")
        return self._summarize_crude(messages)

    def _summarize_with_llm(self, messages: list[dict[str, Any]]) -> str:
        """
        Use LLM to create intelligent summary of conversation.

        Uses agent-agnostic prompt that works for any conversation type.

        Args:
            messages: Messages to summarize

        Returns:
            Summary text (150-200 words)
        """
        # Format conversation for summarization
        conversation_text = self._format_messages_for_summary(messages)

        # Build agent-agnostic summarization prompt
        prompt = """You are summarizing an earlier conversation to preserve context while reducing size.

Create a concise summary in 150-200 WORDS (not tokens).

Focus on:
- Key requests and goals
- Important decisions made
- Actions taken and their outcomes
- Problems encountered and solutions
- Current state or status
- Critical information needed for continuing this conversation

Format: Bullet points, factual, no flowery language.

EARLIER CONVERSATION:
{conversation}

CONCISE SUMMARY (150-200 words, bullet points):""".format(
            conversation=conversation_text
        )

        # Call LLM with low temperature for factual summary
        response = chat_with_inactivity_timeout(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
            inactivity_timeout=30,
        )

        # Extract summary
        if "message" in response and "content" in response["message"]:
            summary = response["message"]["content"].strip()
            return summary

        raise Exception("No content in LLM response")

    def _format_messages_for_summary(self, messages: list[dict[str, Any]]) -> str:
        """
        Format messages into readable text for LLM summarization.

        Limits output to ~15000 chars to keep LLM summarization fast.

        Args:
            messages: Messages to format

        Returns:
            Formatted conversation text (max ~15000 chars)
        """
        lines = []
        total_chars = 0
        max_chars = 15000  # Keep input manageable for fast summarization

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Format based on role
            if role == "user":
                # Truncate long user messages
                truncated = content[:200] + "..." if len(content) > 200 else content
                line = f"USER: {truncated}"
            elif role == "assistant":
                # Check for tool calls
                if "tool_calls" in msg:
                    # Summarize tool calls
                    tool_names = [
                        tc["function"]["name"] for tc in msg["tool_calls"]
                    ]
                    line = f"ASSISTANT: [called tools: {', '.join(tool_names)}]"
                elif content:
                    # Truncate long assistant messages
                    truncated = content[:200] + "..." if len(content) > 200 else content
                    line = f"ASSISTANT: {truncated}"
                else:
                    continue
            elif role == "tool":
                # Summarize tool results briefly
                line = f"TOOL_RESULT: {content[:100]}..."
            else:
                continue

            # Check if adding this line would exceed limit
            if total_chars + len(line) > max_chars:
                lines.append("[...earlier messages truncated for brevity...]")
                break

            lines.append(line)
            total_chars += len(line) + 1  # +1 for newline

        return "\n".join(lines)

    def _summarize_crude(self, messages: list[dict[str, Any]]) -> str:
        """
        Fallback crude summarization (text truncation).

        Used when LLM summarization fails.

        Args:
            messages: Messages to summarize

        Returns:
            Crude summary text
        """
        summary_parts = []

        for msg in messages[:50]:  # Limit to first 50
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
