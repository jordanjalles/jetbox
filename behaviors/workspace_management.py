"""
WorkspaceManagementBehavior - Manages workspace strategy and discovery.

This behavior is ONLY for orchestrator-level agents that need to:
1. Detect whether user wants to edit existing code or create new project
2. List and search available workspaces
3. Provide guidance on workspace selection

Other agents (TaskExecutor, Architect) work WITHIN workspaces but don't manage them.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from pathlib import Path
from behaviors.base import AgentBehavior

if TYPE_CHECKING:
    pass


class WorkspaceManagementBehavior(AgentBehavior):
    """
    Behavior for workspace strategy detection and management.

    Provides:
    - Intent detection guidance (edit vs create)
    - Workspace listing and search tools
    - Strategy recommendations based on user goals

    Only used by orchestrator-level agents, not task executors.
    """

    def __init__(self, workspaces_root: str = ".agent_workspaces"):
        """
        Initialize workspace management behavior.

        Args:
            workspaces_root: Root directory for isolated workspaces (default: .agent_workspaces)
        """
        self.workspaces_root = Path(workspaces_root)

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "workspace_management"

    def get_instructions(self) -> str:
        """
        Provide workspace strategy detection guidance.

        Returns:
            Instructions for detecting workspace intent and selecting strategy
        """
        return """
## Workspace Strategy Detection

Before delegating tasks, analyze user intent to choose appropriate workspace:

### Edit/Modify Mode (use current directory)

**Keywords**: edit, modify, update, refactor, fix, improve, enhance, extend, add to
**Indicators**:
- References existing files: "update app.py", "fix the bug in calculator.py"
- Mentions "my code", "this project", "current app"
- Asks to improve/extend existing functionality
- User is clearly working on something already

**Action**: Use current directory (workspace=".")
- Code changes are made in place
- User can see results immediately
- No isolation needed - they're editing their own project

### Create Mode (use isolated workspace)

**Keywords**: create, build, generate, make, new, from scratch
**Indicators**:
- "Create a new...", "Build a fresh...", "Start a new project"
- No reference to existing files
- Describes complete new system/package/application
- User wants deliverable code artifact

**Action**: Create isolated workspace in .agent_workspaces/
- Clean environment for new project
- Easy to find and export results
- Doesn't pollute current directory

### Available Tools

Use these tools to help decide:
- `list_workspaces()`: See all existing isolated workspaces
- `search_workspaces(query)`: Find workspaces by name/description
- `get_workspace_info(name)`: Get details about a workspace

### Examples

**Edit Mode**:
- "Refactor my calculator code" → workspace="."
- "Add error handling to the API" → workspace="."
- "Fix the bug in utils.py" → workspace="."

**Create Mode**:
- "Create a calculator package" → new isolated workspace
- "Build a Flask REST API" → new isolated workspace
- "Generate a data structures library" → new isolated workspace

**When Unclear**: Ask the user!
- "Are you working on existing code or starting fresh?"
- "Should I work in your current directory or create a new workspace?"
"""

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Provide workspace management tools.

        Returns:
            Tool definitions for workspace listing and search
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_workspaces",
                    "description": "List all isolated workspaces in .agent_workspaces/ directory. Shows workspace names, creation times, and file counts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of workspaces to return (default: 20)"
                            },
                            "sort_by": {
                                "type": "string",
                                "description": "Sort order: 'newest', 'oldest', 'name' (default: newest)",
                                "enum": ["newest", "oldest", "name"]
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_workspaces",
                    "description": "Search for workspaces by name pattern. Useful for finding previous work.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (workspace names containing this string)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_workspace_info",
                    "description": "Get detailed information about a specific workspace (files, size, notes).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "workspace_name": {
                                "type": "string",
                                "description": "Name of the workspace directory"
                            }
                        },
                        "required": ["workspace_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_workspace",
                    "description": "Find best matching workspace for a project name. Returns the workspace path that best matches the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_name": {
                                "type": "string",
                                "description": "Project name or description to search for"
                            }
                        },
                        "required": ["project_name"]
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
        Handle workspace management tool calls.

        Args:
            agent: Agent instance
            tool_name: Tool being called
            args: Tool arguments

        Returns:
            Tool result dict
        """
        if tool_name == "list_workspaces":
            return self._list_workspaces(
                limit=args.get("limit", 20),
                sort_by=args.get("sort_by", "newest")
            )
        elif tool_name == "search_workspaces":
            return self._search_workspaces(args["query"])
        elif tool_name == "get_workspace_info":
            return self._get_workspace_info(args["workspace_name"])
        elif tool_name == "find_workspace":
            return self._find_workspace(args["project_name"])

        return super().dispatch_tool(agent, tool_name, args)

    def _list_workspaces(self, limit: int = 20, sort_by: str = "newest") -> dict[str, Any]:
        """
        List available workspaces.

        Args:
            limit: Maximum number to return
            sort_by: Sort order (newest/oldest/name)

        Returns:
            Dict with workspaces list
        """
        if not self.workspaces_root.exists():
            return {
                "success": True,
                "workspaces": [],
                "message": f"No workspaces found (directory {self.workspaces_root} doesn't exist)"
            }

        try:
            workspaces = []
            for workspace_dir in self.workspaces_root.iterdir():
                if not workspace_dir.is_dir():
                    continue

                # Get workspace info
                info = self._get_workspace_summary(workspace_dir)
                workspaces.append(info)

            # Sort workspaces
            if sort_by == "newest":
                workspaces.sort(key=lambda x: x["modified_time"], reverse=True)
            elif sort_by == "oldest":
                workspaces.sort(key=lambda x: x["modified_time"])
            elif sort_by == "name":
                workspaces.sort(key=lambda x: x["name"])

            # Limit results
            workspaces = workspaces[:limit]

            return {
                "success": True,
                "workspaces": workspaces,
                "count": len(workspaces),
                "message": f"Found {len(workspaces)} workspace(s)"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error listing workspaces: {e}"
            }

    def _search_workspaces(self, query: str) -> dict[str, Any]:
        """
        Search for workspaces by name.

        Args:
            query: Search query string

        Returns:
            Dict with matching workspaces
        """
        if not self.workspaces_root.exists():
            return {
                "success": True,
                "workspaces": [],
                "message": "No workspaces directory found"
            }

        try:
            query_lower = query.lower()
            matches = []

            for workspace_dir in self.workspaces_root.iterdir():
                if not workspace_dir.is_dir():
                    continue

                # Check if query matches workspace name
                if query_lower in workspace_dir.name.lower():
                    info = self._get_workspace_summary(workspace_dir)
                    matches.append(info)

            # Sort by modification time (newest first)
            matches.sort(key=lambda x: x["modified_time"], reverse=True)

            return {
                "success": True,
                "workspaces": matches,
                "count": len(matches),
                "query": query,
                "message": f"Found {len(matches)} workspace(s) matching '{query}'"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error searching workspaces: {e}"
            }

    def _get_workspace_info(self, workspace_name: str) -> dict[str, Any]:
        """
        Get detailed information about a workspace.

        Args:
            workspace_name: Name of the workspace

        Returns:
            Dict with workspace details
        """
        workspace_path = self.workspaces_root / workspace_name

        if not workspace_path.exists():
            return {
                "success": False,
                "error": f"Workspace '{workspace_name}' not found"
            }

        try:
            info = self._get_workspace_summary(workspace_path)

            # Add file list
            files = []
            for item in workspace_path.rglob("*"):
                if item.is_file() and not item.name.startswith("."):
                    rel_path = item.relative_to(workspace_path)
                    files.append({
                        "path": str(rel_path),
                        "size": item.stat().st_size
                    })

            info["files"] = files
            info["file_count"] = len(files)

            # Check for notes file
            notes_file = workspace_path / "workspace_task_notes.md"
            if notes_file.exists():
                try:
                    info["notes"] = notes_file.read_text(encoding="utf-8")
                except Exception:
                    info["notes"] = "(unable to read notes file)"
            else:
                info["notes"] = None

            return {
                "success": True,
                "workspace": info
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting workspace info: {e}"
            }

    def _get_workspace_summary(self, workspace_path: Path) -> dict[str, Any]:
        """
        Get summary information about a workspace.

        Args:
            workspace_path: Path to workspace directory

        Returns:
            Dict with summary info
        """
        import time

        # Get modification time
        mtime = workspace_path.stat().st_mtime

        # Count files (non-hidden)
        file_count = sum(
            1 for item in workspace_path.rglob("*")
            if item.is_file() and not item.name.startswith(".")
        )

        return {
            "name": workspace_path.name,
            "path": str(workspace_path),
            "modified_time": mtime,
            "modified_time_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
            "file_count": file_count
        }

    def _find_workspace(self, project_name: str) -> dict[str, Any]:
        """
        Find best matching workspace for a project name.

        Uses fuzzy matching to score workspaces by how well they match the query.

        Args:
            project_name: Project name or description to search for

        Returns:
            Dict with best matching workspace or error if none found
        """
        project_name_lower = project_name.lower()

        if not self.workspaces_root.exists():
            return {
                "success": False,
                "message": f"No workspaces found. Cannot find workspace for '{project_name}'."
            }

        try:
            # Get all workspaces
            workspaces = []
            for item in self.workspaces_root.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    workspaces.append({
                        "name": item.name,
                        "path": str(item),
                        "modified": item.stat().st_mtime,
                    })

            if not workspaces:
                return {
                    "success": False,
                    "message": f"No workspaces found. Cannot find workspace for '{project_name}'."
                }

            # Score each workspace by how well it matches project_name
            def score_match(workspace_name: str, query: str) -> int:
                """Score how well a workspace name matches a query. Higher is better."""
                ws_lower = workspace_name.lower()
                query_lower = query.lower()

                # Exact match
                if query_lower in ws_lower:
                    # Bonus for matching at word boundaries
                    words = ws_lower.split('-')
                    for word in words:
                        if word == query_lower:
                            return 100  # Exact word match
                        if word.startswith(query_lower):
                            return 80  # Word starts with query
                    return 60  # Contains query

                # Fuzzy match - check if all characters appear in order
                query_idx = 0
                for char in ws_lower:
                    if query_idx < len(query_lower) and char == query_lower[query_idx]:
                        query_idx += 1
                if query_idx == len(query_lower):
                    return 30  # All chars present in order

                # Check individual words
                query_words = query_lower.split()
                ws_words = ws_lower.split('-')
                matches = sum(1 for qw in query_words if any(qw in wsw for wsw in ws_words))
                if matches > 0:
                    return 20 * matches

                return 0

            # Score all workspaces
            scored = []
            for ws in workspaces:
                score = score_match(ws["name"], project_name)
                if score > 0:
                    scored.append((score, ws))

            if not scored:
                # No matches - return list of available workspaces
                ws_list = "\n".join(f"  - {ws['name']}" for ws in workspaces[:10])
                return {
                    "success": False,
                    "message": f"No workspace found matching '{project_name}'.\n\nAvailable workspaces:\n{ws_list}"
                }

            # Sort by score (descending), then by recency
            scored.sort(key=lambda x: (x[0], x[1]["modified"]), reverse=True)

            best_match = scored[0][1]
            best_score = scored[0][0]

            # If we have multiple good matches, show them
            other_matches = [ws for score, ws in scored[1:3] if score >= 30]

            msg = f"Found workspace for '{project_name}':\n"
            msg += f"  Best match: {best_match['name']}\n"
            msg += f"  Path: {best_match['path']}\n"

            if other_matches:
                msg += "\nOther possible matches:\n"
                for ws in other_matches:
                    msg += f"  - {ws['name']}\n"

            return {
                "success": True,
                "workspace": best_match["path"],
                "workspace_name": best_match["name"],
                "message": msg,
                "confidence": "high" if best_score >= 60 else "medium" if best_score >= 30 else "low",
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Error finding workspace: {e}"}
