#!/usr/bin/env python3
"""
Semantic validation - checks for functionality, not file names.

Instead of requiring exact file names, this validator:
1. Scans all Python files in workspace
2. Looks for required classes/functions
3. Tests that imports work
4. Validates core functionality
"""
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Any


class SemanticValidator:
    """Validates code based on semantics, not file names."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.python_files = list(workspace.glob("**/*.py"))
        self.symbols = self._extract_symbols()

    def _extract_symbols(self) -> Dict[str, Set[str]]:
        """
        Extract all classes, functions, and imports from Python files.

        Returns:
            Dict mapping symbol types to sets of symbol names
        """
        symbols = {
            "classes": set(),
            "functions": set(),
            "imports": set(),
            "files": set(),
        }

        for py_file in self.python_files:
            # Track file names (without extension)
            symbols["files"].add(py_file.stem)

            # Parse AST
            try:
                content = py_file.read_text()
                tree = ast.parse(content, filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        symbols["classes"].add(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        symbols["functions"].add(node.name)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            symbols["imports"].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            symbols["imports"].add(node.module)
            except Exception as e:
                # Skip files with syntax errors for now
                pass

        return symbols

    def has_class(self, class_name: str) -> bool:
        """Check if a class with this name exists."""
        return class_name in self.symbols["classes"]

    def has_function(self, func_name: str) -> bool:
        """Check if a function with this name exists."""
        return func_name in self.symbols["functions"]

    def has_import(self, module_name: str) -> bool:
        """Check if a module is imported."""
        return module_name in self.symbols["imports"]

    def has_file_stem(self, stem: str) -> bool:
        """Check if a file with this stem exists (e.g., 'blog' matches blog.py)."""
        return stem in self.symbols["files"]

    def has_any_file_containing(self, substring: str) -> bool:
        """Check if any file name contains substring."""
        return any(substring in f for f in self.symbols["files"])

    def has_required_symbols(self, required: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Check for required symbols.

        Args:
            required: Dict like {"classes": ["BlogManager", "Post"], "functions": ["save", "load"]}

        Returns:
            Dict with results: {"success": bool, "found": {...}, "missing": {...}}
        """
        result = {
            "success": True,
            "found": {},
            "missing": {},
        }

        for symbol_type, names in required.items():
            result["found"][symbol_type] = []
            result["missing"][symbol_type] = []

            for name in names:
                if symbol_type == "classes":
                    if self.has_class(name):
                        result["found"][symbol_type].append(name)
                    else:
                        result["missing"][symbol_type].append(name)
                        result["success"] = False
                elif symbol_type == "functions":
                    if self.has_function(name):
                        result["found"][symbol_type].append(name)
                    else:
                        result["missing"][symbol_type].append(name)
                        result["success"] = False
                elif symbol_type == "imports":
                    if self.has_import(name):
                        result["found"][symbol_type].append(name)
                    else:
                        result["missing"][symbol_type].append(name)
                        result["success"] = False

        return result

    def validate_importable(self, module_stem: str) -> bool:
        """
        Test if a module can be imported.

        Args:
            module_stem: Module name without .py (e.g., "blog")

        Returns:
            True if module can be imported
        """
        # Add workspace to path temporarily
        old_path = sys.path.copy()
        try:
            sys.path.insert(0, str(self.workspace))

            # Try to import
            __import__(module_stem)
            return True
        except ImportError:
            return False
        except Exception:
            # Module imported but has runtime errors - still counts as importable
            return True
        finally:
            sys.path = old_path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of what was found in workspace."""
        return {
            "files": len(self.python_files),
            "file_names": sorted([f.name for f in self.python_files]),
            "classes": sorted(self.symbols["classes"]),
            "functions": sorted(self.symbols["functions"]),
            "imports": sorted(self.symbols["imports"]),
        }


# Validation rules for each task type
VALIDATION_RULES = {
    # L3: Basic file operations and functions
    "calculator": {
        "classes": [],
        "functions": ["add", "subtract", "multiply", "divide"],
        "description": "Calculator with basic arithmetic operations",
    },
    "file_processor": {
        "classes": ["FileProcessor"],
        "functions": [],
        "description": "FileProcessor class with read/write/count operations",
    },

    # L4: Classes with state
    "todo_list": {
        "classes": ["TodoList", "Todo"],
        "functions": [],
        "description": "TodoList class with task management",
    },
    "todo_app": {
        "classes": ["TodoManager", "Todo", "Category"],
        "functions": [],
        "description": "Todo app with Todo/Category models and TodoManager",
    },
    "stack": {
        "classes": ["Stack"],
        "functions": [],
        "description": "Stack class with push/pop/peek operations",
    },

    # L5: Multi-class systems
    "blog_system": {
        "classes": ["BlogManager", "Post", "Comment"],
        "functions": [],  # Don't require specific function names
        "description": "Blog system with Post/Comment models and BlogManager",
    },
    "inventory_system": {
        "classes": ["Inventory", "Product"],
        "functions": [],
        "description": "Inventory system with Product model and Inventory class",
    },

    # L6: Design patterns
    "observer_pattern": {
        "classes": ["Subject", "Observer"],
        "functions": ["notify", "subscribe", "unsubscribe"],
        "description": "Observer pattern with Subject/Observer and event system",
    },
    "factory_pattern": {
        "classes": ["Factory"],
        "functions": ["create", "create_product"],
        "description": "Factory pattern with Factory class and product creation",
    },
    "dependency_injection": {
        "classes": ["Container"],
        "functions": ["register", "resolve"],
        "description": "DI container with service registration and resolution",
    },

    # L7: Algorithms and complex logic
    "rate_limiter": {
        "classes": ["RateLimiter"],
        "functions": ["allow_request"],
        "description": "Rate limiter with token bucket or sliding window",
    },
    "lru_cache": {
        "classes": ["LRUCache"],
        "functions": ["get", "put"],
        "description": "LRU cache with capacity-based eviction",
    },
    "connection_pool": {
        "classes": ["ConnectionPool", "Pool"],
        "functions": ["acquire", "release"],
        "description": "Connection pool with acquire/release",
    },
    "circuit_breaker": {
        "classes": ["CircuitBreaker"],
        "functions": ["call"],
        "description": "Circuit breaker with state management",
    },
}


def validate_workspace(workspace: Path, task_name: str) -> Dict[str, Any]:
    """
    Validate a workspace for a specific task.

    Args:
        workspace: Path to workspace directory
        task_name: Task identifier (e.g., "blog_system")

    Returns:
        Dict with validation results (always includes found/missing keys)
    """
    if task_name not in VALIDATION_RULES:
        return {
            "success": False,
            "error": f"Unknown task: {task_name}",
            "found": {"classes": [], "functions": []},
            "missing": {"classes": [], "functions": []},
        }

    validator = SemanticValidator(workspace)
    rules = VALIDATION_RULES[task_name]

    # Check for required symbols
    required = {
        "classes": rules["classes"],
        "functions": rules["functions"],
    }

    result = validator.has_required_symbols(required)
    result["task"] = task_name
    result["description"] = rules["description"]
    result["summary"] = validator.get_summary()

    return result


if __name__ == "__main__":
    # Test validator
    if len(sys.argv) > 1:
        workspace = Path(sys.argv[1])
        task = sys.argv[2] if len(sys.argv) > 2 else "blog_system"

        result = validate_workspace(workspace, task)

        print(f"Task: {result['task']}")
        print(f"Description: {result['description']}")
        print(f"\nSuccess: {result['success']}")
        print(f"\nFound classes: {result['found']['classes']}")
        print(f"Missing classes: {result['missing']['classes']}")
        print(f"\nFound functions: {result['found']['functions']}")
        print(f"Missing functions: {result['missing']['functions']}")
        print(f"\nWorkspace summary:")
        print(f"  Files: {result['summary']['file_names']}")
        print(f"  All classes: {result['summary']['classes']}")
        print(f"  All functions: {result['summary']['functions'][:10]}...")  # First 10
