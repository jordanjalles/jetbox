"""Workspace isolation for messy workspace environments."""
from __future__ import annotations
import re
import shutil
from pathlib import Path
from typing import Any

def slugify(text: str, max_length: int = 50) -> str:
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')[:max_length].rstrip('-')
    return slug or 'workspace'

class WorkspaceManager:
    def __init__(self, goal: str, base_dir: Path | None = None, auto_cleanup: bool = False,
                 workspace_path: Path | str | None = None) -> None:
        """
        Initialize workspace manager.

        Args:
            goal: The goal/task description
            base_dir: Base directory for isolated workspaces (default: .agent_workspace)
            auto_cleanup: Whether to auto-cleanup on destruction
            workspace_path: If provided, use this existing directory (edit mode)
                           If None, create isolated workspace (isolate mode)
        """
        self.goal = goal
        self.auto_cleanup = auto_cleanup

        # Edit mode: use specified existing directory
        if workspace_path is not None:
            self.workspace_dir = Path(workspace_path).resolve()
            self.is_edit_mode = True
            self.base_dir = self.workspace_dir.parent
            self.workspace_name = self.workspace_dir.name

            # Ensure directory exists
            if not self.workspace_dir.exists():
                raise ValueError(f"Edit mode workspace path does not exist: {workspace_path}")

        # Isolate mode: create new isolated workspace under .agent_workspace
        else:
            self.is_edit_mode = False
            self.base_dir = base_dir or Path(".agent_workspace")
            self.workspace_name = slugify(goal)
            self.workspace_dir = self.base_dir / self.workspace_name
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.created_files: list[str] = []

    def resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        if str(path).startswith('.agent_workspace/'):
            return Path(path)
        return self.workspace_dir / path

    def relative_path(self, path: str | Path) -> str:
        path = Path(path)
        try:
            return str(path.relative_to(self.workspace_dir))
        except ValueError:
            return str(path)

    def list_files(self, pattern: str = "**/*") -> list[str]:
        files = []
        for p in self.workspace_dir.glob(pattern):
            if p.is_file():
                files.append(str(p.relative_to(self.workspace_dir)))
        return sorted(files)

    def get_test_command(self) -> list[str] | None:
        test_dirs = list(self.workspace_dir.glob("tests/"))
        if test_dirs:
            # Use python -m pytest to ensure proper PYTHONPATH handling
            return ["python", "-m", "pytest", str(test_dirs[0].relative_to(self.workspace_dir)), "-q"]
        test_files = list(self.workspace_dir.glob("test_*.py")) + list(self.workspace_dir.glob("*_test.py"))
        if test_files:
            return ["python", "-m", "pytest"] + [str(f.relative_to(self.workspace_dir)) for f in test_files] + ["-q"]
        return None

    def get_lint_command(self) -> list[str] | None:
        if list(self.workspace_dir.glob("**/*.py")):
            return ["ruff", "check", "."]
        return None

    def track_file(self, path: str | Path) -> None:
        rel_path = self.relative_path(path)
        if rel_path not in self.created_files:
            self.created_files.append(rel_path)

    def export_to_root(self, dest_dir: Path | None = None) -> dict[str, str]:
        dest_dir = dest_dir or Path.cwd()
        exported = {}
        for file_path in self.created_files:
            src = self.workspace_dir / file_path
            dst = dest_dir / file_path
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                exported[str(src)] = str(dst)
        return exported

    def cleanup(self) -> None:
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)

    def get_probe_state(self) -> dict[str, Any]:
        state = {
            "workspace_dir": str(self.workspace_dir),
            "files_created": self.created_files.copy(),
            "files_exist": [],
            "files_missing": [],
            "test_command": self.get_test_command(),
            "lint_command": self.get_lint_command(),
        }
        for file_path in self.created_files:
            full_path = self.workspace_dir / file_path
            if full_path.exists():
                state["files_exist"].append(file_path)
            else:
                state["files_missing"].append(file_path)
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "workspace_name": self.workspace_name,
            "workspace_dir": str(self.workspace_dir),
            "base_dir": str(self.base_dir),
            "created_files": self.created_files,
            "auto_cleanup": self.auto_cleanup,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WorkspaceManager':
        workspace = cls(goal=data["goal"], base_dir=Path(data["base_dir"]), auto_cleanup=data.get("auto_cleanup", False))
        workspace.created_files = data.get("created_files", [])
        return workspace
