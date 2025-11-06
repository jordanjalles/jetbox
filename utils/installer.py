"""
Installation utilities with rollback capability for self-extensibility.

Functions:
- install_with_rollback: Install file with automatic backup
- rollback_from_backup: Restore file from backup
- list_backups: List available backups
- cleanup_old_backups: Remove backups older than N days
"""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def install_with_rollback(
    source_file: str | Path,
    dest_file: str | Path,
    backup_dir: str | Path = ".agent_generated/backups"
) -> dict[str, Any]:
    """
    Install file with rollback capability.

    Creates backup of existing file (if any) before overwriting.
    If installation fails, automatically restores from backup.

    Args:
        source_file: Path to source file to install
        dest_file: Path to destination file
        backup_dir: Directory to store backups (default: .agent_generated/backups)

    Returns:
        Dict with:
            - success: bool (True if installed successfully)
            - backup_file: Path to backup (if existing file was backed up)
            - message: str (status message)
            - error: str (if success=False)
    """
    source_path = Path(source_file)
    dest_path = Path(dest_file)
    backup_path = Path(backup_dir)

    # Validate source exists
    if not source_path.exists():
        return {
            "success": False,
            "error": f"Source file does not exist: {source_file}"
        }

    # Create backup directory
    backup_path.mkdir(parents=True, exist_ok=True)

    backup_file = None

    try:
        # Backup existing file if present
        if dest_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"{dest_path.name}.{timestamp}.backup"
            shutil.copy2(dest_path, backup_file)
            print(f"[installer] Backed up existing file: {backup_file}")

        # Create destination directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Install new file
        shutil.copy2(source_path, dest_path)
        print(f"[installer] Installed: {dest_path}")

        return {
            "success": True,
            "backup_file": str(backup_file) if backup_file else None,
            "message": f"Successfully installed {dest_path.name}"
        }

    except Exception as e:
        # Installation failed - restore from backup if we made one
        error_msg = f"Installation failed: {str(e)}"
        print(f"[installer] {error_msg}")

        if backup_file and backup_file.exists():
            try:
                shutil.copy2(backup_file, dest_path)
                print(f"[installer] Restored from backup: {backup_file}")
                error_msg += " (restored from backup)"
            except Exception as restore_error:
                error_msg += f" (restore also failed: {restore_error})"

        return {
            "success": False,
            "error": error_msg,
            "backup_file": str(backup_file) if backup_file else None
        }


def rollback_from_backup(
    backup_file: str | Path,
    dest_file: str | Path | None = None
) -> dict[str, Any]:
    """
    Restore file from a specific backup.

    Args:
        backup_file: Path to backup file
        dest_file: Destination path (if None, infers from backup filename)

    Returns:
        Dict with success status and message
    """
    backup_path = Path(backup_file)

    if not backup_path.exists():
        return {
            "success": False,
            "error": f"Backup file does not exist: {backup_file}"
        }

    # Infer destination from backup filename if not provided
    if dest_file is None:
        # Example: my_behavior.py.20251106_120000.backup -> my_behavior.py
        parts = backup_path.stem.split('.')
        if len(parts) >= 2:
            # Remove timestamp part
            filename = '.'.join(parts[:-1]) if len(parts) > 2 else parts[0]
            dest_file = backup_path.parent.parent / Path(backup_path.name).suffix.lstrip('.') / filename
        else:
            return {
                "success": False,
                "error": "Could not infer destination path from backup filename"
            }

    dest_path = Path(dest_file)

    try:
        # Create destination directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Restore from backup
        shutil.copy2(backup_path, dest_path)
        print(f"[installer] Restored from backup: {dest_path}")

        return {
            "success": True,
            "message": f"Successfully restored {dest_path.name} from backup"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Rollback failed: {str(e)}"
        }


def list_backups(
    backup_dir: str | Path = ".agent_generated/backups",
    file_pattern: str | None = None
) -> list[dict[str, Any]]:
    """
    List available backup files.

    Args:
        backup_dir: Directory containing backups
        file_pattern: Optional pattern to filter files (e.g., "*.py")

    Returns:
        List of dicts with backup info (file, timestamp, size)
    """
    backup_path = Path(backup_dir)

    if not backup_path.exists():
        return []

    backups = []

    pattern = file_pattern if file_pattern else "*.backup"
    for backup_file in backup_path.glob(pattern):
        if backup_file.is_file():
            # Extract timestamp from filename
            parts = backup_file.stem.split('.')
            timestamp_str = parts[-1] if len(parts) > 1 else "unknown"

            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                timestamp = None

            backups.append({
                "file": str(backup_file),
                "name": backup_file.name,
                "timestamp": timestamp,
                "size": backup_file.stat().st_size,
                "age_hours": (datetime.now() - timestamp).total_seconds() / 3600 if timestamp else None
            })

    # Sort by timestamp (newest first)
    backups.sort(key=lambda x: x["timestamp"] if x["timestamp"] else datetime.min, reverse=True)

    return backups


def cleanup_old_backups(
    backup_dir: str | Path = ".agent_generated/backups",
    max_age_days: int = 30,
    keep_latest: int = 5
) -> dict[str, Any]:
    """
    Remove old backup files.

    Args:
        backup_dir: Directory containing backups
        max_age_days: Remove backups older than this many days
        keep_latest: Always keep this many most recent backups per file

    Returns:
        Dict with cleanup statistics
    """
    backup_path = Path(backup_dir)

    if not backup_path.exists():
        return {
            "removed": 0,
            "kept": 0,
            "message": "Backup directory does not exist"
        }

    cutoff_date = datetime.now() - timedelta(days=max_age_days)

    # Group backups by original filename
    backups_by_file = {}
    for backup_file in backup_path.glob("*.backup"):
        if backup_file.is_file():
            # Extract original filename (remove timestamp and .backup)
            parts = backup_file.name.split('.')
            original_name = '.'.join(parts[:-2]) if len(parts) > 2 else parts[0]

            if original_name not in backups_by_file:
                backups_by_file[original_name] = []

            backups_by_file[original_name].append(backup_file)

    removed_count = 0
    kept_count = 0

    for original_name, backup_files in backups_by_file.items():
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        for i, backup_file in enumerate(backup_files):
            # Keep if in latest N
            if i < keep_latest:
                kept_count += 1
                continue

            # Check age
            mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if mtime < cutoff_date:
                try:
                    backup_file.unlink()
                    print(f"[installer] Removed old backup: {backup_file.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"[installer] Failed to remove {backup_file.name}: {e}")
            else:
                kept_count += 1

    return {
        "removed": removed_count,
        "kept": kept_count,
        "message": f"Removed {removed_count} old backups, kept {kept_count}"
    }
