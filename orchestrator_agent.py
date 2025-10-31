"""
Orchestrator agent - manages user conversation and delegates to TaskExecutor.

Uses append-until-full context management: keeps all messages until near token limit,
then performs compaction pass to summarize old messages.

Can switch to task management strategy when working with structured tasks.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import os

from base_agent import BaseAgent
from agent_config import config
from llm_utils import chat_with_inactivity_timeout
from context_strategies import AppendUntilFullStrategy, TaskManagementEnhancement, ContextStrategy
from workspace_manager import WorkspaceManager


class OrchestratorAgent(BaseAgent):
    """
    Agent specialized for user interaction and task delegation.

    Context strategy: Append until full, then compact
    Can switch to TaskManagementStrategy when task breakdown exists
    Tools: Delegation, clarification, planning
    """

    def __init__(self, workspace: Path, context_strategy: ContextStrategy | None = None):
        """
        Initialize Orchestrator agent.

        Args:
            workspace: Working directory
            context_strategy: Optional context strategy (defaults to AppendUntilFullStrategy)
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

        # Workspace manager (initialized when needed for task management)
        self.workspace_manager = None

        # Primary context strategy (always AppendUntilFull for orchestrator)
        self.context_strategy = context_strategy or AppendUntilFullStrategy()

        # Context enhancements (composable plugins)
        self.enhancements = []

        print(f"[orchestrator] Context strategy: {self.context_strategy.get_name()}")

        # Initialize simple context manager (for strategy compatibility)
        from context_manager import ContextManager
        self.context_manager = ContextManager()
        # Set a default goal (will be updated when user makes a request)
        self.context_manager.load_or_init("User conversation")

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tools available to Orchestrator.

        Tools:
        - delegate_to_executor: Send a task to TaskExecutor
        - clarify_with_user: Ask user for clarification
        - create_task_plan: Break down user request into tasks
        - get_executor_status: Check TaskExecutor progress
        + strategy-specific tools (e.g., task management tools)
        """
        base_tools = [
            {
                "type": "function",
                "function": {
                    "name": "consult_architect",
                    "description": "Consult the Architect agent for complex project design. Use this for multi-component systems, architecture decisions, technology recommendations, or when you need structured breakdown of a complex project. The architect will produce architecture documents, module specifications, and a task breakdown.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_description": {
                                "type": "string",
                                "description": "What the user wants to build (high-level description)",
                            },
                            "requirements": {
                                "type": "string",
                                "description": "Functional and non-functional requirements (performance, scale, features, etc.)",
                            },
                            "constraints": {
                                "type": "string",
                                "description": "Constraints and context (team size, tech stack preferences, timeline, existing infrastructure, etc.)",
                            },
                            "workspace_path": {
                                "type": "string",
                                "description": "Optional: Path to existing workspace if consulting on existing project (use find_workspace to get this)",
                            },
                        },
                        "required": ["project_description"],
                    },
                },
            },
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

        # Add strategy-specific tools if available
        strategy_tools = []
        if self.context_strategy:
            strategy_tools = self.context_strategy.get_strategy_tools()

        # Add enhancement tools
        enhancement_tools = []
        for enhancement in self.enhancements:
            enhancement_tools.extend(enhancement.get_enhancement_tools())

        # Merge base tools + strategy tools + enhancement tools
        return base_tools + strategy_tools + enhancement_tools

    def get_system_prompt(self) -> str:
        """Return Orchestrator-specific system prompt."""
        base_text = """You are an orchestrator agent that helps users plan and execute software projects.

Your workflow:
1. When user makes a request, check if you need clarification
2. Assess complexity: SIMPLE (direct implementation) vs COMPLEX (needs architecture)
3. For COMPLEX projects: Consult Architect first, get architecture + task breakdown
4. For SIMPLE projects or after architect: Delegate to TaskExecutor with explicit workspace_mode
5. After delegation completes, report results back to user WITH FILE LOCATIONS

COMPLEXITY ASSESSMENT (NEW STEP):
Before delegating, determine if project needs architecture design:

CONSULT ARCHITECT when:
- Multi-component/multi-service systems (microservices, distributed systems)
- Complex data flows or processing pipelines
- Technology stack decisions needed (which database? which framework?)
- Performance/scaling concerns (handle X req/sec, support Y users)
- Multiple modules with interfaces/dependencies
- User explicitly asks for architecture/design
- Refactoring large existing codebases

SKIP ARCHITECT (delegate directly) when:
- Simple single-file scripts or utilities
- Small feature additions to existing code
- Bug fixes
- Straightforward CRUD applications
- User explicitly asks for quick implementation

WORKSPACE MODE (CRITICAL - YOU MUST ALWAYS SPECIFY):
The delegate_to_executor tool REQUIRES a workspace_mode parameter with value "new" or "existing".

DECISION PROCESS - YOU MUST FOLLOW THIS:

Step 1: Analyze the user request for keywords:
- NEW work indicators: "create", "make", "build", "new", "start"
- EXISTING work indicators: "update", "modify", "add to", "change", "fix", "improve", "enhance"

Step 2: Choose workspace_mode:
- If NEW work → workspace_mode="new" (no workspace_path needed)
- If EXISTING work → MUST call find_workspace first, then workspace_mode="existing" with workspace_path

Step 3: Delegate with correct parameters (to Architect or TaskExecutor)

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

Example 4 - COMPLEX PROJECT (with Architect):
User: "build a real-time analytics platform for 1M events/sec"
→ Assess complexity: multi-component, scaling concerns → COMPLEX
→ consult_architect(
    project_description="real-time analytics platform",
    requirements="1M events/sec ingestion, multi-tenant isolation, schema evolution",
    constraints="AWS + Kubernetes, 4-person team, Go/Python"
  )
→ [Architect returns: task list with 5 tasks, workspace path]
→ ITERATE through each task in the returned task list:
  - Task 1: delegate_to_executor(
      task_description="Implement ingestion module per architecture/modules/ingestion.md",
      workspace_mode="existing",
      workspace_path=".agent_workspace/real-time-analytics-platform"
    )
  - Task 2: delegate_to_executor(
      task_description="Implement processing module per architecture/modules/processing.md",
      workspace_mode="existing",
      workspace_path=".agent_workspace/real-time-analytics-platform"
    )
  - [Continue for all tasks...]
→ Report: "Completed 5 tasks. Architecture in workspace/architecture/, all modules implemented"

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
- For simple requests, delegate immediately to TaskExecutor
- For complex requests, consult Architect first
- ALWAYS delegate to TaskExecutor for coding work (after Architect if needed)
- **CRITICAL**: After Architect consultation, you MUST delegate EACH task from the returned task list
  - Architect result includes "tasks" array with full task details
  - Make ONE delegate_to_executor call PER task
  - Use workspace_mode="existing" and the workspace_path from architect result
  - Include reference to architecture module spec in task_description

Tools available:
- consult_architect: Get architecture design for complex projects (produces docs, specs, task breakdown)
- delegate_to_executor: Send coding tasks (REQUIRES workspace_mode: "new" or "existing")
- find_workspace: Find workspace by project name (REQUIRED before workspace_mode="existing")
- list_workspaces: List all existing workspaces
- clarify_with_user: Ask user questions
- create_task_plan: Structure complex requests (optional, architect does this better for complex projects)
- get_executor_status: Check TaskExecutor progress
"""

        # Add strategy-specific instructions if available
        base_prompt = base_text
        if self.context_strategy:
            strategy_instructions = self.context_strategy.get_strategy_instructions()
            if strategy_instructions:
                base_prompt = base_prompt + "\n" + strategy_instructions

        # Add enhancement instructions
        for enhancement in self.enhancements:
            enhancement_instructions = enhancement.get_enhancement_instructions()
            if enhancement_instructions:
                base_prompt = base_prompt + "\n" + enhancement_instructions

        return base_prompt

    def get_context_strategy(self) -> str:
        """Orchestrator uses append-until-full strategy (or task management if tasks exist)."""
        return self.context_strategy.get_name()

    def add_task_management(self, workspace_path: Path | str) -> None:
        """
        Add TaskManagementEnhancement for structured task tracking.

        Call this after architect consultation creates task breakdown.

        Args:
            workspace_path: Path to workspace containing architecture/task-breakdown.json
        """
        workspace_path = Path(workspace_path)

        # Create workspace manager for this workspace (use generic goal)
        self.workspace_manager = WorkspaceManager(
            goal="orchestrator-task-management",
            workspace_path=workspace_path
        )

        # Add task management enhancement
        task_enhancement = TaskManagementEnhancement(workspace_manager=self.workspace_manager)
        self.enhancements.append(task_enhancement)

        print(f"[orchestrator] Added task management enhancement (workspace: {workspace_path})")

    def build_context(self) -> list[dict[str, Any]]:
        """
        Build context using configured context strategy + enhancements.

        Uses context_strategy.build_context() which handles compaction automatically,
        then injects enhancement context sections.

        Returns:
            [system_prompt, ...enhancements..., ...messages...]
        """
        # Use configured context strategy to build base context
        context = self.context_strategy.build_context(
            context_manager=self.context_manager,
            messages=self.state.messages,
            system_prompt=self.get_system_prompt(),
            config=self.config,
            workspace=self.workspace,
        )

        # Inject enhancements after system prompt (index 1)
        enhancement_index = 1
        for enhancement in self.enhancements:
            enhancement_context = enhancement.get_context_injection(
                context_manager=self.context_manager,
                workspace=self.workspace
            )
            if enhancement_context:
                context.insert(enhancement_index, enhancement_context)
                enhancement_index += 1

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
