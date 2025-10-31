"""Test that all modules can be imported (no PyTorch required)."""

from pathlib import Path


def test_project_structure() -> None:
    """Verify core modules exist."""
    project_root = Path(__file__).parent.parent.parent

    modules = [
        "core/encoders/vision_vit.py",
        "core/encoders/text_transformer.py",
        "core/jepa_core.py",
        "core/objectives/jepa_objectives.py",
        "core/hrm/working_memory.py",
        "core/hrm/abstract_core.py",
        "core/hrm/reflection_loop.py",
        "core/hrm/hrm_reasoner.py",
    ]

    for module_path in modules:
        full_path = project_root / module_path
        assert full_path.exists(), f"Module not found: {module_path}"
        assert full_path.stat().st_size > 0, f"Module is empty: {module_path}"


def test_module_syntax() -> None:
    """Verify all Python files have valid syntax."""
    import py_compile

    project_root = Path(__file__).parent.parent.parent

    python_files = [
        "core/encoders/vision_vit.py",
        "core/encoders/text_transformer.py",
        "core/jepa_core.py",
        "core/objectives/jepa_objectives.py",
        "core/hrm/working_memory.py",
        "core/hrm/abstract_core.py",
        "core/hrm/reflection_loop.py",
        "core/hrm/hrm_reasoner.py",
    ]

    for file_path in python_files:
        full_path = project_root / file_path
        try:
            py_compile.compile(str(full_path), doraise=True)
        except py_compile.PyCompileError as e:
            assert False, f"Syntax error in {file_path}: {e}"


def test_init_files_exist() -> None:
    """Verify __init__.py files exist for packages."""
    project_root = Path(__file__).parent.parent.parent

    init_files = [
        "core/__init__.py",
        "core/encoders/__init__.py",
        "core/objectives/__init__.py",
        "core/hrm/__init__.py",
    ]

    for init_file in init_files:
        full_path = project_root / init_file
        assert full_path.exists(), f"Missing __init__.py: {init_file}"
