"""
Generation Metadata Utilities

Provides metadata tracking for all self-extensibility generated files.
Enables queries like "remove behaviors created today" or "list files by author".

Metadata Format:
    # META: GENERATED_BY=MetaProgrammer
    # META: AUTHOR=user@example.com
    # META: TIMESTAMP=2025-11-07T12:34:56
    # META: GENERATOR=CreateBehaviorBehavior
    # META: VERSION=1.0.0
    # META: PARENT_REQUEST="Create a calculator behavior"
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import re


def generate_metadata_header(
    generator: str,
    author: str = "MetaProgrammer",
    parent_request: str = "",
    version: str = "1.0.0"
) -> str:
    """
    Generate metadata header for a generated file.

    Args:
        generator: Tool that generated the file (e.g., "CreateBehaviorBehavior")
        author: Who requested the generation (default: "MetaProgrammer")
        parent_request: The original user request (optional)
        version: Version of the generator (default: "1.0.0")

    Returns:
        Formatted metadata header as string
    """
    timestamp = datetime.utcnow().isoformat()

    lines = [
        "# META: GENERATED_BY=MetaProgrammer",
        f"# META: GENERATOR={generator}",
        f"# META: AUTHOR={author}",
        f"# META: TIMESTAMP={timestamp}",
        f"# META: VERSION={version}",
    ]

    if parent_request:
        # Escape quotes and newlines in request
        safe_request = parent_request.replace('"', '\\"').replace('\n', ' ')
        lines.append(f'# META: PARENT_REQUEST="{safe_request}"')

    return "\n".join(lines) + "\n"


def parse_metadata(file_path: Path | str) -> dict[str, Any]:
    """
    Parse metadata from a generated file.

    Args:
        file_path: Path to file with metadata header

    Returns:
        Dict with metadata fields (generator, author, timestamp, etc.)
        Empty dict if no metadata found
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {}

    metadata = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Stop at first non-comment line
                if not line.startswith('#'):
                    break

                # Parse META lines
                if line.startswith('# META:'):
                    match = re.match(r'# META: (\w+)=(.*)', line)
                    if match:
                        key = match.group(1).lower()
                        value = match.group(2).strip()

                        # Remove quotes from string values
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]

                        metadata[key] = value
    except Exception:
        return {}

    return metadata


def find_generated_files(
    directory: Path | str,
    generator: str = None,
    author: str = None,
    since: datetime = None,
    before: datetime = None
) -> list[Path]:
    """
    Find generated files matching criteria.

    Args:
        directory: Directory to search (recursively)
        generator: Filter by generator name (e.g., "CreateBehaviorBehavior")
        author: Filter by author
        since: Filter files created after this datetime
        before: Filter files created before this datetime

    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    matching_files = []

    # Search for Python files and YAML files
    for pattern in ['**/*.py', '**/*.yaml', '**/*.yml']:
        for file_path in directory.glob(pattern):
            metadata = parse_metadata(file_path)

            # Skip files without metadata
            if not metadata:
                continue

            # Apply filters
            if generator and metadata.get('generator') != generator:
                continue

            if author and metadata.get('author') != author:
                continue

            # Parse timestamp if present
            if 'timestamp' in metadata:
                try:
                    file_timestamp = datetime.fromisoformat(metadata['timestamp'])

                    if since and file_timestamp < since:
                        continue

                    if before and file_timestamp > before:
                        continue
                except (ValueError, TypeError):
                    continue

            matching_files.append(file_path)

    return matching_files


def get_files_by_date(
    directory: Path | str,
    date: str
) -> list[Path]:
    """
    Get all generated files from a specific date.

    Args:
        directory: Directory to search
        date: Date string in YYYY-MM-DD format

    Returns:
        List of file paths generated on that date
    """
    # Parse date string to datetime range
    try:
        date_obj = datetime.fromisoformat(date)
        since = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        before = date_obj.replace(hour=23, minute=59, second=59, microsecond=999999)

        return find_generated_files(directory, since=since, before=before)
    except ValueError:
        return []


def get_files_today(directory: Path | str) -> list[Path]:
    """
    Get all generated files from today.

    Args:
        directory: Directory to search

    Returns:
        List of file paths generated today
    """
    today = datetime.now().date().isoformat()
    return get_files_by_date(directory, today)


def format_metadata_summary(metadata: dict[str, Any]) -> str:
    """
    Format metadata as human-readable summary.

    Args:
        metadata: Metadata dict from parse_metadata()

    Returns:
        Formatted string summary
    """
    if not metadata:
        return "No metadata found"

    lines = []

    if 'generated_by' in metadata:
        lines.append(f"Generated by: {metadata['generated_by']}")

    if 'generator' in metadata:
        lines.append(f"Generator: {metadata['generator']}")

    if 'author' in metadata:
        lines.append(f"Author: {metadata['author']}")

    if 'timestamp' in metadata:
        try:
            dt = datetime.fromisoformat(metadata['timestamp'])
            lines.append(f"Created: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        except (ValueError, TypeError):
            lines.append(f"Created: {metadata['timestamp']}")

    if 'version' in metadata:
        lines.append(f"Version: {metadata['version']}")

    if 'parent_request' in metadata:
        lines.append(f"Request: {metadata['parent_request']}")

    return "\n".join(lines)


def remove_generated_files(
    file_paths: list[Path],
    backup_dir: Path | str = ".agent_generated/removed"
) -> dict[str, Any]:
    """
    Remove generated files with backup.

    Args:
        file_paths: List of files to remove
        backup_dir: Where to backup files before removal

    Returns:
        Dict with removal results:
        {
            "removed": [list of removed paths],
            "backed_up": [list of backup paths],
            "errors": [list of error messages]
        }
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "removed": [],
        "backed_up": [],
        "errors": []
    }

    for file_path in file_paths:
        file_path = Path(file_path)

        if not file_path.exists():
            results["errors"].append(f"File not found: {file_path}")
            continue

        try:
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name

            # Copy to backup
            import shutil
            shutil.copy2(file_path, backup_path)
            results["backed_up"].append(str(backup_path))

            # Remove original
            file_path.unlink()
            results["removed"].append(str(file_path))

        except Exception as e:
            results["errors"].append(f"Error removing {file_path}: {e}")

    return results
