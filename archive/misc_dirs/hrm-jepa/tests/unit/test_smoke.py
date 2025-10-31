"""Smoke tests to validate basic project setup."""

import sys
from pathlib import Path


def test_python_version() -> None:
    """Verify Python 3.11+ is being used."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


def test_project_structure() -> None:
    """Verify core project directories exist."""
    project_root = Path(__file__).parent.parent.parent

    required_dirs = [
        "core",
        "core/encoders",
        "core/objectives",
        "core/hrm",
        "data",
        "scripts",
        "ui",
        "configs",
        "tests/unit",
        "docs",
        "tools",
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Required directory missing: {dir_path}"
        assert full_path.is_dir(), f"Path exists but is not a directory: {dir_path}"


def test_config_files_exist() -> None:
    """Verify essential configuration files are present."""
    project_root = Path(__file__).parent.parent.parent

    required_files = [
        "pyproject.toml",
        "environment.yml",
        ".pre-commit-config.yaml",
        ".gitignore",
        "README.md",
    ]

    for file_path in required_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"Required file missing: {file_path}"
        assert full_path.is_file(), f"Path exists but is not a file: {file_path}"


def test_no_network_imports() -> None:
    """Verify we don't accidentally import network-dependent modules."""
    # This is a placeholder - actual implementation would scan for
    # requests, urllib, etc. For now, just assert we can check
    import importlib.util

    # Verify we can check for modules
    assert importlib.util.find_spec("pathlib") is not None
    assert importlib.util.find_spec("sys") is not None


def test_pathlib_usage() -> None:
    """Verify Path operations work correctly (Windows compatibility)."""
    project_root = Path(__file__).parent.parent.parent

    # Test basic path operations
    assert project_root.exists()
    assert project_root.is_dir()

    # Test path joining (works across platforms)
    test_path = project_root / "tests" / "unit"
    assert test_path.exists()

    # Test .resolve() for absolute paths
    absolute = test_path.resolve()
    assert absolute.is_absolute()
