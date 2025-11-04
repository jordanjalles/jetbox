"""
Workspace cleanup utility for Jetbox agents.

Provides tools to clean up old agent workspaces based on age criteria.
"""

import os
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta


def parse_time_delta(time_str: str) -> timedelta:
    """
    Parse time delta string like '7d', '2w', '3h'.

    Supported units:
    - h: hours
    - d: days
    - w: weeks
    - m: months (30 days)

    Args:
        time_str: Time string like '7d', '2w', '3h'

    Returns:
        timedelta object

    Raises:
        ValueError: If format is invalid
    """
    if not time_str:
        raise ValueError("Time string cannot be empty")

    # Extract number and unit
    unit = time_str[-1]
    try:
        value = int(time_str[:-1])
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Expected format: <number><unit> (e.g., '7d')")

    # Convert to timedelta
    if unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    elif unit == 'w':
        return timedelta(weeks=value)
    elif unit == 'm':
        return timedelta(days=value * 30)
    else:
        raise ValueError(f"Invalid time unit: {unit}. Supported: h, d, w, m")


def get_workspace_age(workspace_path: Path) -> float:
    """
    Get age of workspace in seconds based on most recent modification.

    Args:
        workspace_path: Path to workspace directory

    Returns:
        Age in seconds (time since last modification)
    """
    if not workspace_path.exists():
        return 0.0

    # Get most recent modification time in the workspace
    latest_mtime = workspace_path.stat().st_mtime

    # Check all files recursively
    for root, dirs, files in os.walk(workspace_path):
        for name in files + dirs:
            path = Path(root) / name
            try:
                mtime = path.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
            except (OSError, FileNotFoundError):
                # Skip files that can't be accessed
                continue

    # Return age in seconds
    current_time = time.time()
    return current_time - latest_mtime


def find_old_workspaces(
    base_dir: Path,
    older_than: timedelta,
) -> list[tuple[Path, float]]:
    """
    Find workspaces older than the specified age.

    Args:
        base_dir: Base directory containing workspaces (e.g., .agent_workspaces)
        older_than: Minimum age (workspaces older than this will be returned)

    Returns:
        List of (workspace_path, age_in_days) tuples, sorted by age (oldest first)
    """
    if not base_dir.exists():
        return []

    old_workspaces = []
    threshold_seconds = older_than.total_seconds()

    # Scan all subdirectories
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        # Skip special directories
        if item.name.startswith('.'):
            continue

        # Get workspace age
        age_seconds = get_workspace_age(item)

        # Check if older than threshold
        if age_seconds > threshold_seconds:
            age_days = age_seconds / 86400  # Convert to days
            old_workspaces.append((item, age_days))

    # Sort by age (oldest first)
    old_workspaces.sort(key=lambda x: x[1], reverse=True)

    return old_workspaces


def clean_workspaces(
    base_dir: Path = Path(".agent_workspaces"),
    older_than: str | None = None,
    dry_run: bool = False,
) -> dict[str, any]:
    """
    Clean up old agent workspaces.

    Args:
        base_dir: Base directory containing workspaces (default: .agent_workspaces)
        older_than: Time delta string (e.g., '7d', '2w', '3h'). REQUIRED.
        dry_run: If True, only show what would be deleted (default: False)

    Returns:
        Dict with cleanup results:
        - found: Number of old workspaces found
        - deleted: Number of workspaces deleted
        - space_freed: Estimated space freed in bytes
        - workspaces: List of workspace details

    Raises:
        ValueError: If older_than is not specified
    """
    if not older_than:
        raise ValueError(
            "Must specify --older-than criteria (e.g., '7d', '2w', '30d'). "
            "Will not clean workspaces without explicit criteria."
        )

    # Parse time delta
    time_delta = parse_time_delta(older_than)

    # Find old workspaces
    old_workspaces = find_old_workspaces(base_dir, time_delta)

    # Prepare results
    results = {
        "found": len(old_workspaces),
        "deleted": 0,
        "space_freed": 0,
        "workspaces": []
    }

    # Process each workspace
    for workspace_path, age_days in old_workspaces:
        # Calculate workspace size
        size_bytes = 0
        for root, dirs, files in os.walk(workspace_path):
            for name in files:
                try:
                    size_bytes += (Path(root) / name).stat().st_size
                except (OSError, FileNotFoundError):
                    pass

        workspace_info = {
            "path": str(workspace_path),
            "age_days": round(age_days, 1),
            "size_mb": round(size_bytes / 1024 / 1024, 2),
        }

        if dry_run:
            workspace_info["action"] = "would_delete"
        else:
            try:
                shutil.rmtree(workspace_path)
                workspace_info["action"] = "deleted"
                results["deleted"] += 1
                results["space_freed"] += size_bytes
            except Exception as e:
                workspace_info["action"] = "failed"
                workspace_info["error"] = str(e)

        results["workspaces"].append(workspace_info)

    return results


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., '1.5 GB', '256 MB')
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def print_cleanup_report(results: dict) -> None:
    """
    Print cleanup results in a readable format.

    Args:
        results: Results dict from clean_workspaces()
    """
    print(f"\n{'=' * 60}")
    print("WORKSPACE CLEANUP REPORT")
    print(f"{'=' * 60}\n")

    if results["found"] == 0:
        print("✅ No old workspaces found matching criteria.")
        return

    print(f"Found: {results['found']} old workspace(s)")
    print(f"Deleted: {results['deleted']} workspace(s)")
    print(f"Space freed: {format_size(results['space_freed'])}\n")

    if results["workspaces"]:
        print("Details:")
        print(f"{'─' * 60}")
        for ws in results["workspaces"]:
            action_icon = {
                "deleted": "✓",
                "would_delete": "→",
                "failed": "✗"
            }.get(ws["action"], "?")

            print(f"{action_icon} {ws['path']}")
            print(f"  Age: {ws['age_days']} days | Size: {ws['size_mb']} MB | Action: {ws['action']}")
            if "error" in ws:
                print(f"  Error: {ws['error']}")
            print()


if __name__ == "__main__":
    import sys

    # Simple CLI for testing
    if len(sys.argv) < 2:
        print("Usage: python workspace_cleanup.py --older-than=7d [--dry-run]")
        sys.exit(1)

    older_than = None
    dry_run = False

    for arg in sys.argv[1:]:
        if arg.startswith("--older-than="):
            older_than = arg.split("=", 1)[1]
        elif arg == "--dry-run":
            dry_run = True

    try:
        results = clean_workspaces(older_than=older_than, dry_run=dry_run)
        print_cleanup_report(results)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
