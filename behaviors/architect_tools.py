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
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from behaviors.base import AgentBehavior


class ArchitectToolsBehavior(AgentBehavior):
    """
    Provides architecture artifact creation tools.

    Creates structured documentation in workspace/architecture/ directory.
    """

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

    def get_tools(self) -> list[dict[str, Any]]:
        """Return architecture tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_architecture_doc",
                    "description": "Write a high-level architecture document (overview, data flow, component diagram, etc.) to the workspace. Creates a markdown file in architecture/ directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Document title (e.g., 'System Overview', 'Data Flow Architecture')"
                            },
                            "content": {
                                "type": "string",
                                "description": "Markdown content with diagrams, component descriptions, architecture decisions"
                            },
                        },
                        "required": ["title", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_module_spec",
                    "description": "Write a detailed module specification to the workspace. Creates a markdown file in architecture/modules/ directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "Module identifier (e.g., 'auth-service', 'data-pipeline')"
                            },
                            "responsibility": {
                                "type": "string",
                                "description": "What this module does (clear, concise description)"
                            },
                            "interfaces": {
                                "type": "object",
                                "description": "Module interfaces with inputs, outputs, APIs",
                                "properties": {
                                    "inputs": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of inputs (format: 'name: type - description')"
                                    },
                                    "outputs": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of outputs (format: 'name: type - description')"
                                    },
                                    "apis": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of APIs (format: 'METHOD /path - description')"
                                    }
                                }
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of dependencies (other modules, external services, databases)"
                            },
                            "technologies": {
                                "type": "object",
                                "description": "Technologies used (language, framework, database, etc.)",
                                "additionalProperties": {"type": "string"}
                            },
                            "implementation_notes": {
                                "type": "string",
                                "description": "Optional: Specific guidance for implementation (edge cases, patterns, etc.)"
                            }
                        },
                        "required": ["module_name", "responsibility", "interfaces", "dependencies", "technologies"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_task_list",
                    "description": "Write a structured task breakdown for the orchestrator. Creates a JSON file in architecture/ directory that the orchestrator can use to delegate tasks to executors.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string", "description": "Task ID (e.g., 'T1', 'T2')"},
                                        "description": {"type": "string", "description": "Task description"},
                                        "module": {"type": "string", "description": "Module this task belongs to"},
                                        "priority": {"type": "integer", "description": "Priority (1=highest)"},
                                        "dependencies": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "List of task IDs this depends on"
                                        },
                                        "estimated_complexity": {
                                            "type": "string",
                                            "enum": ["low", "medium", "high"],
                                            "description": "Estimated complexity"
                                        }
                                    },
                                    "required": ["id", "description", "module", "priority"]
                                },
                                "description": "List of tasks in priority order"
                            }
                        },
                        "required": ["tasks"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_architecture_docs",
                    "description": "List all architecture documents currently in the workspace. Useful to see what's already been created before writing more docs.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_architecture_doc",
                    "description": "Read an existing architecture document from the workspace. Useful for reviewing or updating previous work.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Relative path to document (e.g., 'architecture/overview.md')"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
        ]

    def dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Dispatch architecture tool calls.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            **kwargs: Additional context (workspace_manager)

        Returns:
            Tool result dict
        """
        # Allow runtime override
        workspace_manager = kwargs.get('workspace_manager', self.workspace_manager)

        if tool_name == "write_architecture_doc":
            return self._write_architecture_doc(
                args.get("title"),
                args.get("content"),
                workspace_manager=workspace_manager
            )
        elif tool_name == "write_module_spec":
            return self._write_module_spec(
                args.get("module_name"),
                args.get("responsibility"),
                args.get("interfaces"),
                args.get("dependencies"),
                args.get("technologies"),
                args.get("implementation_notes", ""),
                workspace_manager=workspace_manager
            )
        elif tool_name == "write_task_list":
            return self._write_task_list(
                args.get("tasks"),
                workspace_manager=workspace_manager
            )
        elif tool_name == "list_architecture_docs":
            return self._list_architecture_docs(workspace_manager=workspace_manager)
        elif tool_name == "read_architecture_doc":
            return self._read_architecture_doc(
                args.get("file_path"),
                workspace_manager=workspace_manager
            )
        else:
            return super().dispatch_tool(tool_name, args, **kwargs)

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
