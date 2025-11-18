"""
Architect tools behavior - provides architecture artifact creation.

Extracts architect tools from architect_tools.py into a composable AgentBehavior:
- write_architecture_doc: Create architecture overview documents
- write_module_spec: Create module specifications
- write_task_list: Create task breakdown JSON
- list_architecture_docs: List existing architecture artifacts
- read_architecture_doc: Read existing architecture documents

Features:
- Structured artifact creation
- Workspace-aware file operations
- Metadata injection (timestamps)
- Consistent formatting

Now uses @tool decorator for automatic tool registration!"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from behaviors.base import AgentBehavior
from behaviors.tool_decorator import tool


class ArchitectToolsBehavior(AgentBehavior):
    """
    Provides architecture artifact creation tools.

    Creates structured documentation in workspace/architecture/ directory.

    Security: [] None
    - Writes architecture documentation files locally (internal state change)
    - Does NOT read untrusted input (no [A])
    - Does NOT access sensitive data (no [B])
    - Does NOT communicate externally via network (no [C])
    """

    # Rule of Two: [] - local file writes are internal state changes, not network communication
    rule_of_two_properties = set()

    def __init__(
        self,
        workspace_manager=None,
        **kwargs
    ):
        """
        Initialize ArchitectToolsBehavior.

        Args:
            workspace_manager: WorkspaceManager instance
            **kwargs: Additional parameters (ignored)
        """
        self.workspace_manager = workspace_manager

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "architect_tools"

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject tool documentation for architect tools.

        Called once during agent initialization to document available tools.

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Context with tool documentation injected
        """
        tools = self.get_tools()
        if not tools:
            return context

        # Build tool documentation
        tool_docs = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            # Build parameter signature
            param_strs = []
            for param_name, param_spec in params.items():
                param_type = param_spec.get("type", "any")
                is_required = param_name in required
                if is_required:
                    param_strs.append(f"{param_name}: {param_type}")
                else:
                    param_strs.append(f"{param_name}?: {param_type}")

            param_sig = ", ".join(param_strs) if param_strs else ""
            tool_docs.append(f"  - {name}({param_sig}): {desc}")

        if tool_docs:
            tool_message = f"\n{self.get_name()} tools:\n" + "\n".join(tool_docs)
            # Use default role="system" for framework tool documentation
            return self.inject_message_after_system(context, tool_message)

        return context

    @tool
    def write_architecture_doc(
        self,
        title: str,
        content: str
    ) -> dict[str, Any]:
        """Write a comprehensive architecture document (overview, components, data flow, implementation notes) to the workspace. Creates a single markdown file in architecture/ directory. Include ALL module details in this one document.

        Args:
            title: Document title (e.g., 'Blog System Architecture', 'Todo App Design')
            content: Markdown content with complete architecture: system overview, all components/modules with their responsibilities and interfaces, data flow, technology choices, and implementation guidance. Keep it in ONE document.

        Returns:
            Dict with file path and success status
        """
        workspace_manager = getattr(self.agent, 'workspace_manager', self.workspace_manager)
        return self._write_architecture_doc(title, content, workspace_manager=workspace_manager)

    @tool
    def write_task_list(
        self,
        tasks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Write a structured task breakdown for the orchestrator. Creates a JSON file in architecture/ directory that the orchestrator can use to delegate tasks to executors.

        Args:
            tasks: List of tasks in priority order. Each task should have: id (e.g., 'T1', 'T2'), description, module, priority (1=highest), dependencies (list of task IDs), estimated_complexity ('low', 'medium', 'high')

        Returns:
            Dict with file path, task count, and success status
        """
        workspace_manager = getattr(self.agent, 'workspace_manager', self.workspace_manager)
        return self._write_task_list(tasks, workspace_manager=workspace_manager)

    @tool
    def list_architecture_docs(self) -> dict[str, Any]:
        """List all architecture documents currently in the workspace. Useful to see what's already been created before writing more docs.

        Returns:
            Dict with lists of docs, modules, and task breakdown file
        """
        workspace_manager = getattr(self.agent, 'workspace_manager', self.workspace_manager)
        return self._list_architecture_docs(workspace_manager=workspace_manager)

    @tool
    def read_architecture_doc(
        self,
        file_path: str
    ) -> dict[str, Any]:
        """Read an existing architecture document from the workspace. Useful for reviewing or updating previous work.

        Args:
            file_path: Relative path to document (e.g., 'architecture/overview.md')

        Returns:
            Dict with file content and success status
        """
        workspace_manager = getattr(self.agent, 'workspace_manager', self.workspace_manager)
        return self._read_architecture_doc(file_path, workspace_manager=workspace_manager)

    def _slugify(self, text: str) -> str:
        """Convert text to filesystem-safe slug."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text[:50]

    def _format_list(self, items: list) -> str:
        """Format list items as markdown bullets."""
        if not items:
            return "*(none)*"
        return "\n".join(f"- {item}" for item in items)

    def _format_dict(self, d: dict) -> str:
        """Format dict as markdown table or list."""
        if not d:
            return "*(none)*"
        return "\n".join(f"- **{key}**: {value}" for key, value in d.items())

    def _write_architecture_doc(
        self,
        title: str,
        content: str,
        workspace_manager=None
    ) -> dict[str, Any]:
        """Write high-level architecture document."""
        if not workspace_manager:
            return {"status": "error", "message": "No workspace manager configured"}

        # Create architecture directory
        arch_dir = workspace_manager.workspace_dir / "architecture"
        arch_dir.mkdir(exist_ok=True)

        # Write file
        file_name = f"{self._slugify(title)}.md"
        file_path = arch_dir / file_name

        # Add metadata header
        full_content = f"""# {title}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

{content}
"""

        file_path.write_text(full_content, encoding="utf-8")

        rel_path = file_path.relative_to(workspace_manager.workspace_dir)

        return {
            "status": "success",
            "file_path": str(rel_path),
            "absolute_path": str(file_path),
            "message": f"Architecture doc written: {title}",
        }

    def _write_module_spec(
        self,
        module_name: str,
        responsibility: str,
        interfaces: dict[str, Any],
        dependencies: list[str],
        technologies: dict[str, str],
        implementation_notes: str = "",
        workspace_manager=None
    ) -> dict[str, Any]:
        """Write detailed module specification."""
        if not workspace_manager:
            return {"status": "error", "message": "No workspace manager configured"}

        # Create modules directory
        modules_dir = workspace_manager.workspace_dir / "architecture" / "modules"
        modules_dir.mkdir(parents=True, exist_ok=True)

        # Build markdown content
        content = f"""# Module: {module_name}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Responsibility
{responsibility}

## Interfaces

### Inputs
{self._format_list(interfaces.get('inputs', []))}

### Outputs
{self._format_list(interfaces.get('outputs', []))}

### APIs
{self._format_list(interfaces.get('apis', []))}

## Dependencies
{self._format_list(dependencies)}

## Technologies
{self._format_dict(technologies)}
"""

        if implementation_notes:
            content += f"""
## Implementation Notes
{implementation_notes}
"""

        # Write file
        file_name = f"{self._slugify(module_name)}.md"
        file_path = modules_dir / file_name
        file_path.write_text(content, encoding="utf-8")

        rel_path = file_path.relative_to(workspace_manager.workspace_dir)

        return {
            "status": "success",
            "file_path": str(rel_path),
            "absolute_path": str(file_path),
            "message": f"Module spec written: {module_name}",
        }

    def _write_task_list(
        self,
        tasks: list[dict[str, Any]],
        workspace_manager=None
    ) -> dict[str, Any]:
        """Write structured task breakdown for orchestrator."""
        if not workspace_manager:
            return {"status": "error", "message": "No workspace manager configured"}

        # Create architecture directory
        arch_dir = workspace_manager.workspace_dir / "architecture"
        arch_dir.mkdir(exist_ok=True)

        # Initialize status tracking fields for each task
        for task in tasks:
            # Only initialize if not already present (allows for updating existing breakdowns)
            if "status" not in task:
                task["status"] = "pending"
            if "started_at" not in task:
                task["started_at"] = None
            if "completed_at" not in task:
                task["completed_at"] = None
            if "result" not in task:
                task["result"] = None
            if "attempts" not in task:
                task["attempts"] = 0
            if "notes" not in task:
                task["notes"] = []

        # Build task breakdown structure
        breakdown = {
            "generated_at": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "tasks": tasks,
        }

        # Write JSON file
        file_path = arch_dir / "task-breakdown.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(breakdown, f, indent=2)

        rel_path = file_path.relative_to(workspace_manager.workspace_dir)

        return {
            "status": "success",
            "file_path": str(rel_path),
            "absolute_path": str(file_path),
            "task_count": len(tasks),
            "message": f"Task breakdown written: {len(tasks)} tasks",
        }

    def _list_architecture_docs(
        self,
        workspace_manager=None
    ) -> dict[str, Any]:
        """List all architecture documents in the workspace."""
        if not workspace_manager:
            return {"status": "error", "message": "No workspace manager configured"}

        arch_dir = workspace_manager.workspace_dir / "architecture"

        if not arch_dir.exists():
            return {
                "status": "success",
                "docs": [],
                "modules": [],
                "task_breakdown": None,
                "message": "No architecture documents yet",
            }

        # Find all docs
        docs = []
        for md_file in arch_dir.glob("*.md"):
            rel_path = md_file.relative_to(workspace_manager.workspace_dir)
            docs.append(str(rel_path))

        # Find module specs
        modules = []
        modules_dir = arch_dir / "modules"
        if modules_dir.exists():
            for md_file in modules_dir.glob("*.md"):
                rel_path = md_file.relative_to(workspace_manager.workspace_dir)
                modules.append(str(rel_path))

        # Check for task breakdown
        task_breakdown = None
        task_file = arch_dir / "task-breakdown.json"
        if task_file.exists():
            rel_path = task_file.relative_to(workspace_manager.workspace_dir)
            task_breakdown = str(rel_path)

        return {
            "status": "success",
            "docs": sorted(docs),
            "modules": sorted(modules),
            "task_breakdown": task_breakdown,
            "message": f"Found {len(docs)} docs, {len(modules)} modules",
        }

    def _read_architecture_doc(
        self,
        file_path: str,
        workspace_manager=None
    ) -> dict[str, Any]:
        """Read an existing architecture document."""
        if not workspace_manager:
            return {"status": "error", "message": "No workspace manager configured"}

        full_path = workspace_manager.workspace_dir / file_path

        if not full_path.exists():
            return {"status": "error", "message": f"Document not found: {file_path}"}

        try:
            content = full_path.read_text(encoding="utf-8")
            return {
                "status": "success",
                "content": content,
                "file_path": file_path,
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to read {file_path}: {e}"}
