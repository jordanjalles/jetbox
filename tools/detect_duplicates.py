#!/usr/bin/env python3
"""
Duplicate Code Detector for Jetbox

Detects duplicate code blocks across the codebase using AST-based comparison.
Helps identify consolidation opportunities and reduce code duplication.

Usage:
    python tools/detect_duplicates.py --min-lines 5 --path behaviors/
    python tools/detect_duplicates.py --min-lines 10 --ignore-tests --output duplicates.md
    python tools/detect_duplicates.py --path src/ behaviors/ --min-lines 7
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.code_analysis_utils import (
    scan_python_files,
    extract_code_blocks,
    CodeBlock,
    format_file_location,
    create_markdown_report,
    print_progress
)


class DuplicateGroup:
    """Group of duplicate code blocks."""

    def __init__(self, blocks: List[CodeBlock]):
        self.blocks = blocks
        self.line_count = blocks[0].line_count if blocks else 0
        self.total_duplicated_lines = self.line_count * len(blocks)

    @property
    def duplication_count(self) -> int:
        """Number of duplicates (excluding original)."""
        return len(self.blocks)

    @property
    def impact_score(self) -> int:
        """Impact score for sorting (total wasted lines)."""
        return self.total_duplicated_lines

    def __repr__(self) -> str:
        return (f"DuplicateGroup({self.duplication_count} copies, "
                f"{self.line_count} lines each, "
                f"impact={self.impact_score})")


def find_duplicates(file_paths: List[Path], min_lines: int) -> List[DuplicateGroup]:
    """
    Find duplicate code blocks across files.

    Args:
        file_paths: List of Python files to analyze
        min_lines: Minimum block size to consider

    Returns:
        List of DuplicateGroup objects, sorted by impact
    """
    print(f"📊 Analyzing {len(file_paths)} files for duplicates (min {min_lines} lines)...",
          file=sys.stderr)

    # Extract all code blocks
    all_blocks = []
    for i, file_path in enumerate(file_paths):
        print_progress(i + 1, len(file_paths), "Scanning files")
        blocks = extract_code_blocks(file_path, min_lines)
        all_blocks.extend(blocks)

    print(f"\n✓ Found {len(all_blocks)} code blocks to analyze", file=sys.stderr)

    # Group by AST hash
    hash_groups: Dict[str, List[CodeBlock]] = defaultdict(list)
    for block in all_blocks:
        hash_groups[block.ast_hash].append(block)

    # Filter groups with duplicates (2+ blocks with same hash)
    duplicate_groups = []
    for blocks in hash_groups.values():
        if len(blocks) >= 2:
            duplicate_groups.append(DuplicateGroup(blocks))

    # Sort by impact (most duplicated lines first)
    duplicate_groups.sort(key=lambda g: g.impact_score, reverse=True)

    return duplicate_groups


def format_duplicate_report(duplicate_groups: List[DuplicateGroup],
                           workspace_root: Path,
                           output_format: str = 'console') -> str:
    """
    Format duplicate detection results.

    Args:
        duplicate_groups: List of DuplicateGroup objects
        workspace_root: Root directory for relative paths
        output_format: 'console' or 'markdown'

    Returns:
        Formatted report string
    """
    if not duplicate_groups:
        return "✓ No duplicates found!"

    if output_format == 'markdown':
        return _format_markdown_report(duplicate_groups, workspace_root)
    else:
        return _format_console_report(duplicate_groups, workspace_root)


def _format_console_report(duplicate_groups: List[DuplicateGroup],
                           workspace_root: Path) -> str:
    """Format report for console output."""
    lines = []
    lines.append("=" * 80)
    lines.append("DUPLICATE CODE REPORT")
    lines.append("=" * 80)
    lines.append("")

    total_duplicated_lines = sum(g.total_duplicated_lines for g in duplicate_groups)
    lines.append(f"Found {len(duplicate_groups)} duplicate code patterns")
    lines.append(f"Total duplicated lines: {total_duplicated_lines}")
    lines.append("")

    for i, group in enumerate(duplicate_groups, 1):
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Duplicate #{i} (Impact: {group.impact_score} lines)")
        lines.append(f"  {group.duplication_count} copies × {group.line_count} lines each")
        lines.append(f"{'─' * 80}")

        lines.append("\nLocations:")
        for block in group.blocks:
            location = format_file_location(block.file_path, block.start_line, workspace_root)
            lines.append(f"  • {location} (lines {block.start_line}-{block.end_line})")

        # Show code preview (first 10 lines)
        lines.append("\nCode preview:")
        code_lines = group.blocks[0].code.split('\n')[:10]
        for line in code_lines:
            lines.append(f"  {line.rstrip()}")
        if len(group.blocks[0].code.split('\n')) > 10:
            lines.append("  ...")

        lines.append("\n💡 Suggestion: Extract to shared function or base class")

    lines.append("\n" + "=" * 80)
    return '\n'.join(lines)


def _format_markdown_report(duplicate_groups: List[DuplicateGroup],
                            workspace_root: Path) -> str:
    """Format report as markdown."""
    sections = []

    # Summary section
    total_duplicated_lines = sum(g.total_duplicated_lines for g in duplicate_groups)
    summary = [
        f"**Found {len(duplicate_groups)} duplicate code patterns**",
        "",
        f"- Total duplicated lines: {total_duplicated_lines}",
        f"- Files analyzed: {len(set(b.file_path for g in duplicate_groups for b in g.blocks))}",
    ]
    sections.append({
        'title': 'Summary',
        'items': summary
    })

    # Duplicates section
    duplicate_items = []
    for i, group in enumerate(duplicate_groups, 1):
        duplicate_items.append(f"### Duplicate #{i} (Impact: {group.impact_score} lines)")
        duplicate_items.append("")
        duplicate_items.append(f"**{group.duplication_count} copies × {group.line_count} lines each**")
        duplicate_items.append("")
        duplicate_items.append("**Locations:**")

        for block in group.blocks:
            location = format_file_location(block.file_path, block.start_line, workspace_root)
            duplicate_items.append(f"- `{location}` (lines {block.start_line}-{block.end_line})")

        duplicate_items.append("")
        duplicate_items.append("**Code preview:**")
        duplicate_items.append("```python")
        code_lines = group.blocks[0].code.split('\n')[:15]
        duplicate_items.append('\n'.join(code_lines))
        if len(group.blocks[0].code.split('\n')) > 15:
            duplicate_items.append("...")
        duplicate_items.append("```")
        duplicate_items.append("")
        duplicate_items.append("💡 **Suggestion:** Extract to shared function or base class")
        duplicate_items.append("")

    sections.append({
        'title': 'Duplicate Code Blocks',
        'items': duplicate_items
    })

    return create_markdown_report("Duplicate Code Report", sections)


def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate code blocks in Python codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan behaviors directory with 5+ line minimum
  %(prog)s --min-lines 5 --path behaviors/

  # Scan multiple directories, ignore tests, save to markdown
  %(prog)s --path src/ behaviors/ --ignore-tests --output duplicates.md

  # Scan entire codebase with 10+ line minimum
  %(prog)s --min-lines 10

  # Scan specific file
  %(prog)s --path behaviors/delegation.py --min-lines 7
        """
    )

    parser.add_argument('--path', nargs='+', default=['.'],
                       help='Paths to scan (files or directories). Default: current directory')
    parser.add_argument('--min-lines', type=int, default=5,
                       help='Minimum number of lines for duplicate detection (default: 5)')
    parser.add_argument('--ignore-tests', action='store_true',
                       help='Ignore test files (test_*.py, *_test.py, tests/)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file (markdown format). If not specified, prints to console')
    parser.add_argument('--ignore-patterns', nargs='+',
                       default=['archive/', '__pycache__/', '.git/', 'venv/', '.venv/'],
                       help='Patterns to ignore (default: archive, cache, venv)')

    args = parser.parse_args()

    # Resolve paths
    workspace_root = Path.cwd()
    paths = [Path(p).resolve() for p in args.path]

    # Validate paths
    for path in paths:
        if not path.exists():
            print(f"❌ Error: Path does not exist: {path}", file=sys.stderr)
            sys.exit(1)

    # Scan for Python files
    python_files = scan_python_files(
        paths,
        ignore_tests=args.ignore_tests,
        ignore_patterns=args.ignore_patterns
    )

    if not python_files:
        print("❌ No Python files found to analyze", file=sys.stderr)
        sys.exit(1)

    # Find duplicates
    duplicate_groups = find_duplicates(python_files, args.min_lines)

    # Format report
    output_format = 'markdown' if args.output else 'console'
    report = format_duplicate_report(duplicate_groups, workspace_root, output_format)

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\n✓ Report saved to: {output_path}", file=sys.stderr)
        print(f"  Found {len(duplicate_groups)} duplicate patterns", file=sys.stderr)
    else:
        print(report)

    # Exit with appropriate code
    sys.exit(0 if not duplicate_groups else 1)


if __name__ == '__main__':
    main()
