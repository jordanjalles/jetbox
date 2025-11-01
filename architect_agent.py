"""
Software Architect Agent - Architecture design and planning consultant.

The Architect agent is a consultant agent that produces architecture artifacts
rather than executing code. It helps with:
- System architecture design
- Module decomposition and interface design
- Technology recommendations
- Task breakdown for complex projects

Artifacts are written to workspace/architecture/ directory.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path

from base_agent import BaseAgent
from context_strategies import AppendUntilFullStrategy, TaskManagementEnhancement
from context_manager import ContextManager
from workspace_manager import WorkspaceManager
import architect_tools
import task_management_tools
from agent_config import config


ARCHITECT_SYSTEM_PROMPT = """You are an expert Software Architect agent that helps design robust, scalable software systems.

## Your Role

You translate project goals, requirements, and constraints into clear architectural artifacts:
- Architecture overviews (system design, components, data flow)
- Module specifications (responsibilities, interfaces, dependencies, technologies)
- Task breakdowns (structured lists for implementation)

## Workflow

1. **Understand & Gather**: Ask clarifying questions about requirements, constraints, team skills, existing systems
2. **Design**: Propose architecture patterns, identify modules/components, map data flow
3. **Refine**: Detail interfaces, APIs, technologies; justify trade-offs
4. **Document**: Use tools to create architecture artifacts in the workspace
5. **Break Down**: Create task list for orchestrator to delegate to implementation agents

## Guidelines

- **Ask before assuming**: Clarify ambiguous requirements
- **Present trade-offs**: No perfect solution - explain alternatives and rationale
- **Be specific**: Concrete technology choices, clear interfaces, actionable tasks
- **Use tools**: ALWAYS create artifacts with write_architecture_doc, write_module_spec, write_task_list
- **Stay high-level**: You design architecture, not implement code
- **Consider constraints**: Team size, skills, timeline, existing infrastructure

## Available Tools

- **write_architecture_doc(title, content)**: Write high-level architecture document
- **write_module_spec(module_name, responsibility, interfaces, dependencies, technologies, implementation_notes)**: Write detailed module specification
- **write_task_list(tasks)**: Write structured task breakdown (JSON) for orchestrator
- **list_architecture_docs()**: List existing architecture documents
- **read_architecture_doc(file_path)**: Read existing architecture document

## Output Format

When creating architecture docs, include:
- **Components**: What modules/services exist, their responsibilities
- **Data Flow**: How information moves through the system
- **Technologies**: Specific choices (language, framework, database) with rationale
- **Trade-offs**: What you chose, what you didn't, why
- **Risks**: Potential issues and mitigation strategies

When creating module specs, include:
- **Responsibility**: Clear, concise statement
- **Interfaces**: Inputs, outputs, APIs (be specific about types and formats)
- **Dependencies**: What this module needs (other modules, external services)
- **Technologies**: Concrete tech stack
- **Implementation Notes**: Guidance for developers (edge cases, patterns to use)

When creating task lists, include:
- **Task ID**: T1, T2, etc.
- **Description**: What needs to be built
- **Module**: Which module this task belongs to
- **Priority**: 1 (highest) to N
- **Dependencies**: Which other tasks must complete first
- **Estimated Complexity**: low, medium, high

## Example Interaction

User: "I need a real-time analytics platform for 1M events/sec, multi-tenant isolation"
Architect: [asks clarifying questions about latency, tenancy, existing infrastructure]
User: [provides answers]
Architect: [uses write_architecture_doc to create system overview]
Architect: [uses write_module_spec for each module: ingestion, processing, storage, API]
Architect: [uses write_task_list to break down into implementation tasks]
Architect: "Architecture complete. Documents in workspace/architecture/. Task breakdown ready for implementation."

## Important Notes

- You are a **consultant**, not an executor - you don't write code, you design architecture
- Always create artifacts with tools - your output is documentation, not conversation
- Focus on high-level design - leave implementation details to task executor agents
- Be clear and specific - vague architecture causes implementation chaos
"""


class ArchitectAgent(BaseAgent):
    """
    Software Architect agent for design and planning.

    Responsibilities:
    - Gather requirements and constraints
    - Design system architecture (components, data flow, technologies)
    - Create module specifications
    - Generate task breakdown for orchestrator

    Context Strategy: ArchitectStrategy (append-until-full, higher token limit)
    Tools: Architecture artifact creation (write docs, specs, task lists)
    Output: Architecture artifacts in workspace/architecture/
    """

    def __init__(
        self,
        workspace: Path,
        project_description: str = "",
        context_strategy = None,
    ):
        """
        Initialize architect agent.

        Args:
            workspace: Working directory for this agent
            project_description: Initial project description (optional)
            context_strategy: Custom context strategy (default: AppendUntilFullStrategy)
        """
        super().__init__(
            name="architect",
            role="Software architecture consultant",
            workspace=workspace,
            config=config,
        )

        # Use AppendUntilFullStrategy (with higher token limit for architecture discussions)
        self.context_strategy = context_strategy or AppendUntilFullStrategy(max_tokens=131072)

        # Context enhancements (composable plugins)
        self.enhancements = []

        # Initialize context manager
        self.context_manager = ContextManager()

        # Initialize workspace manager (for task management if needed)
        self.workspace_manager = None

        # Set project description if provided
        if project_description:
            self.set_project(project_description)

        # Configure architect tools with workspace
        self.configure_workspace(self.workspace)

    def configure_workspace(self, workspace_path) -> None:
        """
        Configure architect tools with workspace.

        Call this whenever the workspace changes.
        Automatically switches to TaskManagementStrategy if task breakdown exists.

        Args:
            workspace_path: Path to workspace directory
        """
        self.workspace = workspace_path

        class SimpleWorkspace:
            def __init__(self, workspace_dir):
                self.workspace_dir = workspace_dir

        architect_tools.set_workspace(SimpleWorkspace(self.workspace))

        # Check if task breakdown already exists
        task_file = Path(self.workspace) / "architecture" / "task-breakdown.json"
        if task_file.exists():
            # Initialize workspace manager for existing project
            self.workspace_manager = WorkspaceManager(
                goal="architect-task-management",
                workspace_path=self.workspace
            )

            # Add task management enhancement for existing project
            task_management_tools.set_workspace(self.workspace_manager)
            task_enhancement = TaskManagementEnhancement(workspace_manager=self.workspace_manager)
            self.enhancements.append(task_enhancement)
            print(f"[architect] Added task management enhancement (existing task breakdown found)")
        else:
            # No task breakdown yet - will add enhancement after creating it
            self.workspace_manager = None
            print(f"[architect] No task breakdown yet (will add enhancement after creation)")

    def dispatch_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch architect tool calls.

        Args:
            tool_call: Tool call dict with function name and arguments

        Returns:
            Tool result dict
        """
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        # Map architect tools
        tool_map = {
            "write_architecture_doc": architect_tools.write_architecture_doc,
            "write_module_spec": architect_tools.write_module_spec,
            "write_task_list": architect_tools.write_task_list,
            "list_architecture_docs": architect_tools.list_architecture_docs,
            "read_architecture_doc": architect_tools.read_architecture_doc,
        }

        # Execute the tool
        if tool_name in tool_map:
            result = tool_map[tool_name](**args)
            return {"result": result}
        else:
            return {"result": {"status": "error", "message": f"Unknown architect tool: {tool_name}"}}

    def set_project(self, description: str) -> None:
        """
        Set project description for architecture work.

        Args:
            description: Project description
        """
        self.context_manager.load_or_init(description)

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return architect-specific tools.

        Tools:
        - write_architecture_doc: Write high-level architecture
        - write_module_spec: Write module specifications
        - write_task_list: Write task breakdown
        - list_architecture_docs: List existing docs
        - read_architecture_doc: Read existing doc
        + strategy-specific tools (e.g., task management tools)
        """
        base_tools = architect_tools.get_architect_tool_definitions()

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
        """Return architect system prompt with strategy instructions."""
        base_prompt = ARCHITECT_SYSTEM_PROMPT

        # Add strategy-specific instructions if available
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
        """Architect uses append-until-full strategy + enhancements."""
        enhancements_str = f" + {len(self.enhancements)} enhancements" if self.enhancements else ""
        return f"{self.context_strategy.get_name()}{enhancements_str}"

    def build_context(self) -> list[dict[str, Any]]:
        """
        Build context using configured context strategy + enhancements.

        Auto-adds TaskManagementEnhancement if task breakdown exists in workspace.

        Returns:
            Context list ready for LLM
        """
        # Auto-add task management enhancement if task breakdown exists and not already added
        self._auto_add_task_management()

        # Build base context
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

    def _auto_add_task_management(self) -> None:
        """
        Auto-add TaskManagementEnhancement if task breakdown exists.

        This allows architect to see task status after creating task breakdown,
        without requiring explicit configure_workspace() call.

        Only adds if:
        1. Task breakdown file exists in workspace
        2. Enhancement not already added
        """
        # Skip if already have task management enhancement
        if any(isinstance(e, TaskManagementEnhancement) for e in self.enhancements):
            return

        # Check for task breakdown in workspace
        task_file = Path(self.workspace) / "architecture" / "task-breakdown.json"
        if not task_file.exists():
            return

        # Task breakdown exists - add enhancement
        print(f"[architect] Auto-adding TaskManagementEnhancement (found {task_file})")

        # Create workspace manager if needed
        if not self.workspace_manager:
            self.workspace_manager = WorkspaceManager(
                goal="architect-task-management",
                workspace_path=self.workspace
            )

        # Add enhancement
        task_management_tools.set_workspace(self.workspace_manager)
        task_enhancement = TaskManagementEnhancement(workspace_manager=self.workspace_manager)
        self.enhancements.append(task_enhancement)

    def dispatch_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch architect tool calls.

        Args:
            tool_call: Tool call dict with function name and arguments

        Returns:
            Tool result dict
        """
        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        # Map tool names to functions
        tool_map = {
            "write_architecture_doc": architect_tools.write_architecture_doc,
            "write_module_spec": architect_tools.write_module_spec,
            "write_task_list": architect_tools.write_task_list,
            "list_architecture_docs": architect_tools.list_architecture_docs,
            "read_architecture_doc": architect_tools.read_architecture_doc,
        }

        # Execute tool
        if tool_name in tool_map:
            try:
                result = tool_map[tool_name](**args)
                return {"result": result}
            except Exception as e:
                return {"result": {"status": "error", "message": f"Tool error: {e}"}}
        else:
            return {"result": {"status": "error", "message": f"Unknown tool: {tool_name}"}}

    def consult(
        self,
        project_description: str,
        requirements: str = "",
        constraints: str = "",
        model: str = "qwen3:14b",
        temperature: float = 0.3,
        max_rounds: int = 10,
    ) -> dict[str, Any]:
        """
        Run architecture consultation session.

        This is the main entry point for consulting the architect.
        It runs an interactive session where the architect can ask questions
        and create artifacts.

        Args:
            project_description: What to build
            requirements: Functional + non-functional requirements
            constraints: Team size, tech stack, timeline, etc.
            model: LLM model to use
            temperature: Sampling temperature
            max_rounds: Maximum consultation rounds

        Returns:
            {
                "status": "success" | "incomplete",
                "artifacts": {
                    "docs": ["architecture/overview.md", ...],
                    "modules": ["architecture/modules/auth.md", ...],
                    "task_breakdown": "architecture/task-breakdown.json" | None
                },
                "message": "..."
            }
        """
        # Set project
        self.set_project(project_description)

        # Build initial context with requirements and constraints
        initial_context = f"""PROJECT: {project_description}

REQUIREMENTS:
{requirements if requirements else "(to be determined)"}

CONSTRAINTS:
{constraints if constraints else "(to be determined)"}

Please design the architecture for this project. Start by asking any clarifying questions you need, then create architecture artifacts using the provided tools."""

        # Add initial user message
        self.add_message({"role": "user", "content": initial_context})

        # Run consultation rounds
        for round_num in range(max_rounds):
            print(f"\n[architect] Round {round_num + 1}/{max_rounds}")

            # Call LLM
            response = self.call_llm(model, temperature)

            # Add assistant message
            if "message" in response:
                self.add_message(response["message"])

                # Check if agent is done (no tool calls)
                if not response["message"].get("tool_calls"):
                    print("[architect] No tool calls - consultation may be complete")
                    break

                # Execute tool calls
                for tool_call in response["message"].get("tool_calls", []):
                    print(f"[architect] Executing: {tool_call['function']['name']}")
                    result = self.dispatch_tool(tool_call)

                    # Add tool result
                    self.add_message({
                        "role": "tool",
                        "content": str(result.get("result", {}))
                    })

            # Persist state
            self.persist_state()

        # Get final artifacts
        artifacts_result = architect_tools.list_architecture_docs()

        if artifacts_result["status"] == "success":
            return {
                "status": "success",
                "artifacts": {
                    "docs": artifacts_result["docs"],
                    "modules": artifacts_result["modules"],
                    "task_breakdown": artifacts_result["task_breakdown"],
                },
                "message": f"Consultation complete. Created {len(artifacts_result['docs'])} docs, {len(artifacts_result['modules'])} modules.",
            }
        else:
            return {
                "status": "incomplete",
                "message": "Consultation ended but no artifacts created",
            }
