#!/usr/bin/env python3
"""
Shared utilities for code analysis tools.

Provides common functionality for scanning Python files, parsing AST,
detecting code patterns, and generating reports.
"""

import ast
import hashlib
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
import sys


class CodeBlock:
    """Represents a block of code for comparison."""

    def __init__(self, file_path: Path, start_line: int, end_line: int,
                 code: str, ast_hash: str):
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.code = code
        self.ast_hash = ast_hash
        self.line_count = end_line - start_line + 1

    def __repr__(self) -> str:
        return (f"CodeBlock({self.file_path}:{self.start_line}-{self.end_line}, "
                f"{self.line_count} lines)")


class FunctionInfo:
    """Information about a function or method definition."""

    def __init__(self, name: str, file_path: Path, line_number: int,
                 is_method: bool = False, class_name: Optional[str] = None,
                 is_private: bool = False, is_dunder: bool = False):
        self.name = name
        self.file_path = file_path
        self.line_number = line_number
        self.is_method = is_method
        self.class_name = class_name
        self.is_private = is_private
        self.is_dunder = is_dunder
        self.references: List[Tuple[Path, int]] = []

    @property
    def full_name(self) -> str:
        """Return fully qualified name (Class.method or function)."""
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name

    def __repr__(self) -> str:
        return f"FunctionInfo({self.full_name} at {self.file_path}:{self.line_number})"


def scan_python_files(paths: List[Path],
                      ignore_tests: bool = False,
                      ignore_patterns: Optional[List[str]] = None) -> List[Path]:
    """
    Scan for Python files in given paths.

    Args:
        paths: List of files or directories to scan
        ignore_tests: Skip files matching test patterns
        ignore_patterns: Additional patterns to ignore (e.g., ['archive/', '__pycache__'])

    Returns:
        List of Python file paths
    """
    if ignore_patterns is None:
        ignore_patterns = ['archive/', '__pycache__/', '.git/', 'venv/', '.venv/']

    python_files = []

    for path in paths:
        if path.is_file():
            if path.suffix == '.py':
                python_files.append(path)
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                # Check ignore patterns
                if any(pattern in str(py_file) for pattern in ignore_patterns):
                    continue

                # Check test files
                if ignore_tests and is_test_file(py_file):
                    continue

                python_files.append(py_file)

    return sorted(set(python_files))


def is_test_file(file_path: Path) -> bool:
    """Check if file is a test file."""
    name = file_path.name.lower()
    return (name.startswith('test_') or
            name.endswith('_test.py') or
            'tests' in file_path.parts)


def parse_ast_safe(file_path: Path) -> Optional[ast.AST]:
    """
    Parse Python file to AST, handling errors gracefully.

    Returns:
        AST node or None if parsing failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        print(f"⚠️  Syntax error in {file_path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"⚠️  Error parsing {file_path}: {e}", file=sys.stderr)
        return None


def normalize_ast(node: ast.AST) -> str:
    """
    Convert AST to normalized string for comparison.

    Strips line numbers, column offsets, and other location info
    to focus on structural similarity.
    """
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def hash_ast(node: ast.AST) -> str:
    """Generate hash of AST node for quick comparison."""
    normalized = normalize_ast(node)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def extract_code_blocks(file_path: Path, min_lines: int = 5) -> List[CodeBlock]:
    """
    Extract code blocks from a Python file.

    Extracts function bodies, class bodies, and top-level statements
    that meet the minimum line threshold.

    Args:
        file_path: Path to Python file
        min_lines: Minimum number of lines for a block

    Returns:
        List of CodeBlock objects
    """
    tree = parse_ast_safe(file_path)
    if not tree:
        return []

    blocks = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}", file=sys.stderr)
        return []

    # Extract function and class definitions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not (hasattr(node, 'lineno') and hasattr(node, 'end_lineno')):
                continue

            start = node.lineno
            end = node.end_lineno

            if end - start + 1 >= min_lines:
                code = ''.join(lines[start-1:end])
                ast_hash_val = hash_ast(node)

                blocks.append(CodeBlock(
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    code=code,
                    ast_hash=ast_hash_val
                ))

    return blocks


def extract_functions(file_path: Path) -> List[FunctionInfo]:
    """
    Extract all function and method definitions from a Python file.

    Returns:
        List of FunctionInfo objects
    """
    tree = parse_ast_safe(file_path)
    if not tree:
        return []

    functions = []

    # Track class context for methods
    class ClassVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class = None
            self.functions = []

        def visit_ClassDef(self, node):
            old_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = old_class

        def visit_FunctionDef(self, node):
            self._handle_function(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            self._handle_function(node)
            self.generic_visit(node)

        def _handle_function(self, node):
            is_private = node.name.startswith('_') and not node.name.startswith('__')
            is_dunder = node.name.startswith('__') and node.name.endswith('__')

            func_info = FunctionInfo(
                name=node.name,
                file_path=file_path,
                line_number=node.lineno,
                is_method=self.current_class is not None,
                class_name=self.current_class,
                is_private=is_private,
                is_dunder=is_dunder
            )
            self.functions.append(func_info)

    visitor = ClassVisitor()
    visitor.visit(tree)

    return visitor.functions


def find_references(function_name: str, search_paths: List[Path],
                   class_name: Optional[str] = None) -> List[Tuple[Path, int]]:
    """
    Find references to a function/method in codebase.

    Searches for direct calls, imports, and getattr patterns.

    Args:
        function_name: Name of function to search for
        search_paths: Files to search in
        class_name: If method, the class name

    Returns:
        List of (file_path, line_number) tuples
    """
    references = []

    # Build search patterns
    patterns = [
        f"{function_name}(",  # Direct call
        f"self.{function_name}",  # Method call
        f'"{function_name}"',  # String reference (getattr, etc.)
        f"'{function_name}'",  # String reference (getattr, etc.)
    ]

    if class_name:
        patterns.append(f"{class_name}.{function_name}")

    for file_path in search_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                # Check if any pattern appears in line
                if any(pattern in line for pattern in patterns):
                    # Avoid counting the definition itself
                    if f"def {function_name}(" not in line:
                        references.append((file_path, line_num))

        except Exception as e:
            print(f"⚠️  Error searching {file_path}: {e}", file=sys.stderr)

    return references


def format_file_location(file_path: Path, line_number: Optional[int] = None,
                        workspace_root: Optional[Path] = None) -> str:
    """
    Format file location for display.

    Args:
        file_path: Absolute path to file
        line_number: Optional line number
        workspace_root: Optional root to make path relative

    Returns:
        Formatted string like "behaviors/foo.py:123"
    """
    if workspace_root:
        try:
            rel_path = file_path.relative_to(workspace_root)
        except ValueError:
            rel_path = file_path
    else:
        rel_path = file_path

    if line_number:
        return f"{rel_path}:{line_number}"
    return str(rel_path)


def create_markdown_report(title: str, sections: List[Dict]) -> str:
    """
    Create markdown report from sections.

    Args:
        title: Report title
        sections: List of dicts with 'title', 'content', 'items' keys

    Returns:
        Markdown formatted string
    """
    lines = [
        f"# {title}",
        "",
        f"Generated: {Path.cwd()}",
        "",
    ]

    for section in sections:
        lines.append(f"## {section['title']}")
        lines.append("")

        if 'content' in section:
            lines.append(section['content'])
            lines.append("")

        if 'items' in section:
            for item in section['items']:
                lines.append(item)
            lines.append("")

    return '\n'.join(lines)


def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Print progress bar to stderr."""
    if total == 0:
        return

    percent = int(100 * current / total)
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f"\r{prefix}: [{bar}] {percent}% ({current}/{total})",
          end='', file=sys.stderr)

    if current == total:
        print(file=sys.stderr)  # New line when done
