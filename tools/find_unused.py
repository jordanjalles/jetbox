#!/usr/bin/env python3
"""
Unused Code Finder for Jetbox

Finds functions and methods that aren't called anywhere in the codebase.
Helps identify dead code that can be safely removed.

Usage:
    python tools/find_unused.py behaviors/workspace_task_notes.py
    python tools/find_unused.py behaviors/ --output unused_report.md
    python tools/find_unused.py base_agent.py --include-private
    python tools/find_unused.py src/ --show-used
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.code_analysis_utils import (
    scan_python_files,
    extract_functions,
    find_references,
    FunctionInfo,
    format_file_location,
    create_markdown_report,
    print_progress
)


# Event system patterns that indicate usage even without direct calls
EVENT_PATTERNS = [
    'on_',      # Event handlers (on_task_complete, on_before_llm)
    'get_',     # Getters called via introspection
    'set_',     # Setters called via introspection
    'dispatch_',  # Event dispatchers
    'handle_',  # Event handlers
]


def is_likely_used_by_framework(func_info: FunctionInfo) -> bool:
    """
    Check if function is likely used by framework even without direct calls.

    Returns True for:
    - Event handlers (on_*)
    - Getters/setters (get_*, set_*)
    - Magic methods (__init__, __str__, etc.)
    - Common override methods (run, execute, etc.)
    """
    if func_info.is_dunder:
        return True

    for pattern in EVENT_PATTERNS:
        if func_info.name.startswith(pattern):
            return True

    # Common override methods
    common_overrides = ['run', 'execute', 'call', 'apply', 'process']
    if func_info.name in common_overrides:
        return True

    return False


def analyze_unused_code(target_paths: List[Path],
                       search_paths: List[Path],
                       include_private: bool = False,
                       include_dunder: bool = False) -> Dict[str, List[FunctionInfo]]:
    """
    Analyze codebase for unused functions.

    Args:
        target_paths: Files to extract functions from
        search_paths: Files to search for references
        include_private: Include private methods (_method)
        include_dunder: Include dunder methods (__method__)

    Returns:
        Dict with 'unused', 'used', 'framework' lists of FunctionInfo
    """
    print(f"📊 Extracting functions from {len(target_paths)} files...", file=sys.stderr)

    # Extract all functions from target files
    all_functions = []
    for i, file_path in enumerate(target_paths):
        print_progress(i + 1, len(target_paths), "Extracting functions")
        functions = extract_functions(file_path)
        all_functions.extend(functions)

    print(f"\n✓ Found {len(all_functions)} functions to analyze", file=sys.stderr)

    # Filter based on options
    filtered_functions = []
    for func in all_functions:
        if not include_dunder and func.is_dunder:
            continue
        if not include_private and func.is_private:
            continue
        filtered_functions.append(func)

    print(f"✓ Analyzing {len(filtered_functions)} functions (after filters)", file=sys.stderr)

    # Find references for each function
    unused = []
    used = []
    framework = []

    for i, func_info in enumerate(filtered_functions):
        print_progress(i + 1, len(filtered_functions), "Finding references")

        references = find_references(
            func_info.name,
            search_paths,
            func_info.class_name
        )

        # Filter out self-references (definition in same file)
        external_refs = [
            (path, line) for path, line in references
            if path != func_info.file_path or line != func_info.line_number
        ]

        func_info.references = external_refs

        if not external_refs:
            # No references found - check if framework usage
            if is_likely_used_by_framework(func_info):
                framework.append(func_info)
            else:
                unused.append(func_info)
        else:
            used.append(func_info)

    print(file=sys.stderr)  # New line after progress

    return {
        'unused': unused,
        'used': used,
        'framework': framework
    }


def format_unused_report(analysis: Dict[str, List[FunctionInfo]],
                        workspace_root: Path,
                        output_format: str = 'console',
                        show_used: bool = False) -> str:
    """
    Format unused code analysis results.

    Args:
        analysis: Dict with 'unused', 'used', 'framework' lists
        workspace_root: Root directory for relative paths
        output_format: 'console' or 'markdown'
        show_used: Include information about used functions

    Returns:
        Formatted report string
    """
    if output_format == 'markdown':
        return _format_markdown_report(analysis, workspace_root, show_used)
    else:
        return _format_console_report(analysis, workspace_root, show_used)


def _format_console_report(analysis: Dict[str, List[FunctionInfo]],
                           workspace_root: Path,
                           show_used: bool) -> str:
    """Format report for console output."""
    lines = []
    lines.append("=" * 80)
    lines.append("UNUSED CODE REPORT")
    lines.append("=" * 80)
    lines.append("")

    unused = analysis['unused']
    used = analysis['used']
    framework = analysis['framework']

    lines.append(f"Summary:")
    lines.append(f"  • Unused functions: {len(unused)}")
    lines.append(f"  • Framework/event functions: {len(framework)}")
    lines.append(f"  • Used functions: {len(used)}")
    lines.append("")

    # Unused functions
    if unused:
        lines.append(f"\n{'═' * 80}")
        lines.append("UNUSED FUNCTIONS (No references found)")
        lines.append(f"{'═' * 80}\n")

        # Group by file
        by_file: Dict[Path, List[FunctionInfo]] = {}
        for func in unused:
            by_file.setdefault(func.file_path, []).append(func)

        for file_path in sorted(by_file.keys()):
            funcs = by_file[file_path]
            rel_path = format_file_location(file_path, workspace_root=workspace_root)
            lines.append(f"\n📄 {rel_path} ({len(funcs)} unused)")
            lines.append("─" * 80)

            for func in sorted(funcs, key=lambda f: f.line_number):
                location = format_file_location(func.file_path, func.line_number, workspace_root)
                lines.append(f"  • {func.full_name}")
                lines.append(f"    Location: {location}")
                if func.is_private:
                    lines.append(f"    Type: Private method")
                elif func.is_method:
                    lines.append(f"    Type: Method")
                else:
                    lines.append(f"    Type: Function")
                lines.append(f"    💡 Consider removing if truly unused")
                lines.append("")

    else:
        lines.append("\n✓ No unused functions found!")

    # Framework functions
    if framework:
        lines.append(f"\n{'═' * 80}")
        lines.append("FRAMEWORK/EVENT FUNCTIONS (May be used via introspection)")
        lines.append(f"{'═' * 80}\n")

        for func in sorted(framework, key=lambda f: (str(f.file_path), f.line_number)):
            location = format_file_location(func.file_path, func.line_number, workspace_root)
            lines.append(f"  • {func.full_name}")
            lines.append(f"    Location: {location}")
            lines.append(f"    ℹ️  Likely called via event system or introspection")
            lines.append("")

    # Used functions (optional)
    if show_used and used:
        lines.append(f"\n{'═' * 80}")
        lines.append("USED FUNCTIONS (With references)")
        lines.append(f"{'═' * 80}\n")

        for func in sorted(used, key=lambda f: len(f.references), reverse=True)[:20]:
            location = format_file_location(func.file_path, func.line_number, workspace_root)
            lines.append(f"  • {func.full_name} ({len(func.references)} references)")
            lines.append(f"    Location: {location}")

            # Show first few references
            for ref_path, ref_line in func.references[:3]:
                ref_location = format_file_location(ref_path, ref_line, workspace_root)
                lines.append(f"      - {ref_location}")

            if len(func.references) > 3:
                lines.append(f"      ... and {len(func.references) - 3} more")
            lines.append("")

    lines.append("=" * 80)
    return '\n'.join(lines)


def _format_markdown_report(analysis: Dict[str, List[FunctionInfo]],
                            workspace_root: Path,
                            show_used: bool) -> str:
    """Format report as markdown."""
    sections = []

    unused = analysis['unused']
    used = analysis['used']
    framework = analysis['framework']

    # Summary section
    summary = [
        f"- **Unused functions:** {len(unused)}",
        f"- **Framework/event functions:** {len(framework)}",
        f"- **Used functions:** {len(used)}",
    ]
    sections.append({
        'title': 'Summary',
        'items': summary
    })

    # Unused functions section
    if unused:
        unused_items = []

        # Group by file
        by_file: Dict[Path, List[FunctionInfo]] = {}
        for func in unused:
            by_file.setdefault(func.file_path, []).append(func)

        for file_path in sorted(by_file.keys()):
            funcs = by_file[file_path]
            rel_path = format_file_location(file_path, workspace_root=workspace_root)
            unused_items.append(f"### {rel_path} ({len(funcs)} unused)")
            unused_items.append("")

            for func in sorted(funcs, key=lambda f: f.line_number):
                location = format_file_location(func.file_path, func.line_number, workspace_root)
                unused_items.append(f"#### `{func.full_name}`")
                unused_items.append("")
                unused_items.append(f"- **Location:** `{location}`")

                if func.is_private:
                    unused_items.append(f"- **Type:** Private method")
                elif func.is_method:
                    unused_items.append(f"- **Type:** Method")
                else:
                    unused_items.append(f"- **Type:** Function")

                unused_items.append(f"- 💡 **Suggestion:** Consider removing if truly unused")
                unused_items.append("")

        sections.append({
            'title': 'Unused Functions',
            'items': unused_items
        })

    # Framework functions section
    if framework:
        framework_items = []

        for func in sorted(framework, key=lambda f: (str(f.file_path), f.line_number)):
            location = format_file_location(func.file_path, func.line_number, workspace_root)
            framework_items.append(f"- **`{func.full_name}`**")
            framework_items.append(f"  - Location: `{location}`")
            framework_items.append(f"  - ℹ️ Likely called via event system or introspection")

        sections.append({
            'title': 'Framework/Event Functions',
            'items': framework_items
        })

    # Used functions section (optional)
    if show_used and used:
        used_items = []
        used_items.append("Showing top 20 most-referenced functions:")
        used_items.append("")

        for func in sorted(used, key=lambda f: len(f.references), reverse=True)[:20]:
            location = format_file_location(func.file_path, func.line_number, workspace_root)
            used_items.append(f"#### `{func.full_name}` ({len(func.references)} references)")
            used_items.append("")
            used_items.append(f"- **Location:** `{location}`")
            used_items.append(f"- **References:**")

            # Show first few references
            for ref_path, ref_line in func.references[:5]:
                ref_location = format_file_location(ref_path, ref_line, workspace_root)
                used_items.append(f"  - `{ref_location}`")

            if len(func.references) > 5:
                used_items.append(f"  - ... and {len(func.references) - 5} more")
            used_items.append("")

        sections.append({
            'title': 'Used Functions',
            'items': used_items
        })

    return create_markdown_report("Unused Code Report", sections)


def main():
    parser = argparse.ArgumentParser(
        description="Find unused functions and methods in Python codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check single file for unused methods
  %(prog)s behaviors/workspace_task_notes.py

  # Check directory, save to markdown
  %(prog)s behaviors/ --output unused_report.md

  # Include private methods, show what IS used
  %(prog)s base_agent.py --include-private --show-used

  # Check multiple directories
  %(prog)s src/ behaviors/ --output unused.md

  # Check file and search entire codebase
  %(prog)s behaviors/delegation.py --search-path .
        """
    )

    parser.add_argument('paths', nargs='+',
                       help='Files or directories to analyze for unused code')
    parser.add_argument('--search-path', nargs='+',
                       help='Directories to search for references (default: same as paths)')
    parser.add_argument('--include-private', action='store_true',
                       help='Include private methods (starting with _)')
    parser.add_argument('--include-dunder', action='store_true',
                       help='Include dunder methods (__init__, __str__, etc.)')
    parser.add_argument('--show-used', action='store_true',
                       help='Show information about used functions too')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file (markdown format). If not specified, prints to console')
    parser.add_argument('--ignore-patterns', nargs='+',
                       default=['archive/', '__pycache__/', '.git/', 'venv/', '.venv/'],
                       help='Patterns to ignore (default: archive, cache, venv)')

    args = parser.parse_args()

    # Resolve paths
    workspace_root = Path.cwd()
    target_paths = [Path(p).resolve() for p in args.paths]

    # Validate target paths
    for path in target_paths:
        if not path.exists():
            print(f"❌ Error: Path does not exist: {path}", file=sys.stderr)
            sys.exit(1)

    # Determine search paths
    if args.search_path:
        search_root_paths = [Path(p).resolve() for p in args.search_path]
    else:
        search_root_paths = target_paths

    # Scan target files
    target_files = scan_python_files(
        target_paths,
        ignore_tests=False,  # Include tests in target
        ignore_patterns=args.ignore_patterns
    )

    if not target_files:
        print("❌ No Python files found to analyze", file=sys.stderr)
        sys.exit(1)

    # Scan search files (broader search)
    search_files = scan_python_files(
        search_root_paths,
        ignore_tests=False,
        ignore_patterns=args.ignore_patterns
    )

    print(f"✓ Target files: {len(target_files)}", file=sys.stderr)
    print(f"✓ Search files: {len(search_files)}", file=sys.stderr)

    # Analyze unused code
    analysis = analyze_unused_code(
        target_files,
        search_files,
        include_private=args.include_private,
        include_dunder=args.include_dunder
    )

    # Format report
    output_format = 'markdown' if args.output else 'console'
    report = format_unused_report(
        analysis,
        workspace_root,
        output_format,
        show_used=args.show_used
    )

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\n✓ Report saved to: {output_path}", file=sys.stderr)
        print(f"  Unused: {len(analysis['unused'])}", file=sys.stderr)
        print(f"  Framework: {len(analysis['framework'])}", file=sys.stderr)
        print(f"  Used: {len(analysis['used'])}", file=sys.stderr)
    else:
        print(report)

    # Exit with code based on findings
    sys.exit(0 if not analysis['unused'] else 1)


if __name__ == '__main__':
    main()
