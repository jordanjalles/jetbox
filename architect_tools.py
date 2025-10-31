"""
Architect-specific tools for creating architecture artifacts.

These tools allow the Architect agent to create structured documentation
in the workspace: architecture overviews, module specifications, and task breakdowns.
"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import json
from datetime import datetime
import re


# Global workspace manager (set by ArchitectAgent)
_workspace_manager = None


def set_workspace(workspace_manager) -> None:
    """Set the workspace manager for architect tools."""
    global _workspace_manager
    _workspace_manager = workspace_manager


def _slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50]


def write_architecture_doc(
    title: str,
    content: str,
) -> dict[str, Any]:
    """
    Write high-level architecture document.

    Args:
        title: Document title (e.g., "System Overview", "Data Flow Architecture")
        content: Markdown content with diagrams, component descriptions

    Returns:
        {
            "status": "success",
            "file_path": "architecture/system-overview.md",
            "message": "Architecture doc written: System Overview"
        }
    """
    if not _workspace_manager:
        return {"status": "error", "message": "No workspace manager configured"}

    # Create architecture directory
    arch_dir = _workspace_manager.workspace_dir / "architecture"
    arch_dir.mkdir(exist_ok=True)

    # Write file
    file_name = f"{_slugify(title)}.md"
    file_path = arch_dir / file_name

    # Add metadata header
    full_content = f"""# {title}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

{content}
"""

    file_path.write_text(full_content, encoding="utf-8")

    rel_path = file_path.relative_to(_workspace_manager.workspace_dir)

    return {
        "status": "success",
        "file_path": str(rel_path),
        "absolute_path": str(file_path),
        "message": f"Architecture doc written: {title}",
    }


def write_module_spec(
    module_name: str,
    responsibility: str,
    interfaces: dict[str, Any],
    dependencies: list[str],
    technologies: dict[str, str],
    implementation_notes: str = "",
) -> dict[str, Any]:
    """
    Write detailed module specification.

    Args:
        module_name: Module identifier (e.g., "auth-service", "data-pipeline")
        responsibility: What this module does
        interfaces: {
            "inputs": ["input1: type - description", ...],
            "outputs": ["output1: type - description", ...],
            "apis": ["POST /api/resource - description", ...]
        }
        dependencies: ["module-a", "module-b", "PostgreSQL", ...]
        technologies: {"language": "Python", "framework": "FastAPI", ...}
        implementation_notes: Specific guidance for implementation

    Returns:
        {
            "status": "success",
            "file_path": "architecture/modules/auth-service.md",
        }
    """
    if not _workspace_manager:
        return {"status": "error", "message": "No workspace manager configured"}

    # Create modules directory
    modules_dir = _workspace_manager.workspace_dir / "architecture" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    # Build markdown content
    content = f"""# Module: {module_name}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Responsibility
{responsibility}

## Interfaces

### Inputs
{_format_list(interfaces.get('inputs', []))}

### Outputs
{_format_list(interfaces.get('outputs', []))}

### APIs
{_format_list(interfaces.get('apis', []))}

## Dependencies
{_format_list(dependencies)}

## Technologies
{_format_dict(technologies)}
"""

    if implementation_notes:
        content += f"""
## Implementation Notes
{implementation_notes}
"""

    # Write file
    file_name = f"{_slugify(module_name)}.md"
    file_path = modules_dir / file_name
    file_path.write_text(content, encoding="utf-8")

    rel_path = file_path.relative_to(_workspace_manager.workspace_dir)

    return {
        "status": "success",
        "file_path": str(rel_path),
        "absolute_path": str(file_path),
        "message": f"Module spec written: {module_name}",
    }


def write_task_list(
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Write structured task breakdown for orchestrator.

    Args:
        tasks: [
            {
                "id": "T1",
                "description": "Implement authentication module",
                "module": "auth-service",
                "priority": 1,
                "dependencies": [],
                "estimated_complexity": "medium"
            },
            ...
        ]

    Returns:
        {
            "status": "success",
            "file_path": "architecture/task-breakdown.json",
            "task_count": 5
        }
    """
    if not _workspace_manager:
        return {"status": "error", "message": "No workspace manager configured"}

    # Create architecture directory
    arch_dir = _workspace_manager.workspace_dir / "architecture"
    arch_dir.mkdir(exist_ok=True)

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

    rel_path = file_path.relative_to(_workspace_manager.workspace_dir)

    return {
        "status": "success",
        "file_path": str(rel_path),
        "absolute_path": str(file_path),
        "task_count": len(tasks),
        "message": f"Task breakdown written: {len(tasks)} tasks",
    }


def list_architecture_docs() -> dict[str, Any]:
    """
    List all architecture documents in the workspace.

    Returns:
        {
            "status": "success",
            "docs": ["architecture/overview.md", ...],
            "modules": ["architecture/modules/auth.md", ...],
            "task_breakdown": "architecture/task-breakdown.json" or None
        }
    """
    if not _workspace_manager:
        return {"status": "error", "message": "No workspace manager configured"}

    arch_dir = _workspace_manager.workspace_dir / "architecture"

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
        rel_path = md_file.relative_to(_workspace_manager.workspace_dir)
        docs.append(str(rel_path))

    # Find module specs
    modules = []
    modules_dir = arch_dir / "modules"
    if modules_dir.exists():
        for md_file in modules_dir.glob("*.md"):
            rel_path = md_file.relative_to(_workspace_manager.workspace_dir)
            modules.append(str(rel_path))

    # Check for task breakdown
    task_breakdown = None
    task_file = arch_dir / "task-breakdown.json"
    if task_file.exists():
        rel_path = task_file.relative_to(_workspace_manager.workspace_dir)
        task_breakdown = str(rel_path)

    return {
        "status": "success",
        "docs": sorted(docs),
        "modules": sorted(modules),
        "task_breakdown": task_breakdown,
        "message": f"Found {len(docs)} docs, {len(modules)} modules",
    }


def read_architecture_doc(file_path: str) -> dict[str, Any]:
    """
    Read an existing architecture document.

    Args:
        file_path: Relative path to document (e.g., "architecture/overview.md")

    Returns:
        {"status": "success", "content": "...", "file_path": "..."}
    """
    if not _workspace_manager:
        return {"status": "error", "message": "No workspace manager configured"}

    full_path = _workspace_manager.workspace_dir / file_path

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


# ============================================================================
# Helper Functions
# ============================================================================

def _format_list(items: list) -> str:
    """Format list items as markdown bullets."""
    if not items:
        return "*(none)*"
    return "\n".join(f"- {item}" for item in items)


def _format_dict(d: dict) -> str:
    """Format dict as markdown table or list."""
    if not d:
        return "*(none)*"
    return "\n".join(f"- **{key}**: {value}" for key, value in d.items())


# ============================================================================
# Tool Definitions (for LLM)
# ============================================================================

def get_architect_tool_definitions() -> list[dict[str, Any]]:
    """
    Get tool definitions for architect agent.

    Returns:
        List of tool definitions in Ollama format
    """
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
