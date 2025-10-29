#!/usr/bin/env python3
"""
Systematic check for missing integrations between Python modules.

Compares agent_legacy.py (reference implementation) with task_executor_agent.py
to find features that were implemented but not integrated.
"""
import ast
import re
from pathlib import Path
from typing import Set, Dict, List
from collections import defaultdict


def extract_imports(file_path: Path) -> Set[str]:
    """Extract all imported module names from a Python file."""
    with open(file_path) as f:
        content = f.read()

    tree = ast.parse(content)
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    return imports


def extract_function_calls(file_path: Path) -> Set[str]:
    """Extract all function call names from a Python file."""
    with open(file_path) as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return set()

    calls = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)

    return calls


def find_defined_but_unused_modules() -> Dict[str, List[str]]:
    """Find Python modules that are defined but never imported."""
    workspace = Path(".")
    all_modules = {f.stem for f in workspace.glob("*.py") if not f.name.startswith("_")}

    # Collect all imports from all files
    imported_modules = set()
    for py_file in workspace.glob("*.py"):
        imported_modules.update(extract_imports(py_file))

    # Also check imports in subdirectories
    for py_file in workspace.glob("tests/*.py"):
        imported_modules.update(extract_imports(py_file))

    unused = all_modules - imported_modules

    # Filter out scripts (files with __name__ == "__main__")
    unused_non_scripts = []
    for module in unused:
        module_path = workspace / f"{module}.py"
        if module_path.exists():
            with open(module_path) as f:
                content = f.read()
                if '__name__ == "__main__"' not in content:
                    unused_non_scripts.append(module)

    return {"unused_modules": unused_non_scripts}


def compare_agent_implementations() -> Dict[str, any]:
    """Compare agent_legacy.py with task_executor_agent.py."""
    legacy_path = Path("agent_legacy.py")
    new_path = Path("task_executor_agent.py")

    if not legacy_path.exists() or not new_path.exists():
        return {"error": "Missing agent files"}

    legacy_imports = extract_imports(legacy_path)
    new_imports = extract_imports(new_path)

    legacy_calls = extract_function_calls(legacy_path)
    new_calls = extract_function_calls(new_path)

    missing_imports = legacy_imports - new_imports
    missing_calls = legacy_calls - new_calls

    # Filter to relevant differences (exclude standard library)
    stdlib_modules = {
        "os", "sys", "re", "time", "json", "pathlib", "subprocess",
        "tempfile", "shutil", "datetime", "typing", "dataclasses",
        "collections", "itertools", "functools", "copy"
    }

    relevant_missing_imports = missing_imports - stdlib_modules

    return {
        "missing_imports": list(relevant_missing_imports),
        "missing_calls": list(missing_calls),
    }


def check_tool_definitions() -> Dict[str, List[str]]:
    """Check if all tools in tools.py are exported and used."""
    tools_path = Path("tools.py")
    if not tools_path.exists():
        return {"error": "tools.py not found"}

    with open(tools_path) as f:
        content = f.read()

    # Find function definitions
    tree = ast.parse(content)
    defined_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith("_"):  # Public functions
                defined_functions.append(node.name)

    # Check get_tool_definitions
    tool_defs_pattern = r'"name":\s*"(\w+)"'
    advertised_tools = set(re.findall(tool_defs_pattern, content))

    # Check dispatch maps
    agent_path = Path("agent.py")
    task_executor_path = Path("task_executor_agent.py")

    dispatched_tools = set()
    for path in [agent_path, task_executor_path]:
        if path.exists():
            with open(path) as f:
                agent_content = f.read()
                # Look for tool_map or dispatch patterns
                tool_map_pattern = r'"(\w+)":\s*tools\.\w+'
                dispatched_tools.update(re.findall(tool_map_pattern, agent_content))

    defined_set = set(defined_functions)

    return {
        "defined_tools": sorted(defined_functions),
        "advertised_tools": sorted(advertised_tools),
        "dispatched_tools": sorted(dispatched_tools),
        "not_advertised": sorted(defined_set - advertised_tools),
        "not_dispatched": sorted(defined_set - dispatched_tools),
    }


def check_utility_integrations() -> Dict[str, any]:
    """Check if utility modules are properly integrated."""
    utilities = {
        "completion_detector": ["analyze_llm_response", "should_nudge_completion"],
        "jetbox_notes": ["append_to_jetbox_notes", "prompt_for_goal_summary"],
        "workspace_manager": ["WorkspaceManager"],
        "status_display": ["StatusDisplay"],
        "server_manager": ["ServerManager"],
        "prompt_loader": ["load_prompts"],
        "agent_config": ["config", "AgentConfig"],
        "llm_utils": ["chat", "chat_with_inactivity_timeout"],
    }

    results = {}

    for module, expected_usage in utilities.items():
        module_path = Path(f"{module}.py")
        if not module_path.exists():
            continue

        # Find where this module is imported
        imported_in = []
        for py_file in Path(".").glob("*.py"):
            if py_file.stem == module:
                continue

            imports = extract_imports(py_file)
            if module in imports:
                imported_in.append(py_file.stem)

        # Find which functions/classes are actually called
        used_functions = []
        for py_file in Path(".").glob("*.py"):
            if py_file.stem == module:
                continue

            calls = extract_function_calls(py_file)
            used_functions.extend([fn for fn in expected_usage if fn in calls])

        results[module] = {
            "imported_in": imported_in,
            "expected_usage": expected_usage,
            "actually_used": list(set(used_functions)),
            "unused": [fn for fn in expected_usage if fn not in used_functions],
        }

    return results


def main():
    print("="*80)
    print("SYSTEMATIC INTEGRATION CHECK")
    print("="*80)
    print()

    # 1. Compare legacy vs new implementation
    print("1. COMPARING AGENT IMPLEMENTATIONS")
    print("-" * 80)
    comparison = compare_agent_implementations()

    if "error" in comparison:
        print(f"   ERROR: {comparison['error']}")
    else:
        print(f"   Missing imports in new agent: {len(comparison['missing_imports'])}")
        for imp in sorted(comparison['missing_imports']):
            print(f"      - {imp}")

        if not comparison['missing_imports']:
            print("      ✓ All imports present")

    print()

    # 2. Check tool definitions
    print("2. CHECKING TOOL DEFINITIONS")
    print("-" * 80)
    tool_check = check_tool_definitions()

    if "error" in tool_check:
        print(f"   ERROR: {tool_check['error']}")
    else:
        print(f"   Defined tools: {len(tool_check['defined_tools'])}")
        print(f"   Advertised to LLM: {len(tool_check['advertised_tools'])}")
        print(f"   Dispatched in agents: {len(tool_check['dispatched_tools'])}")

        if tool_check['not_advertised']:
            print(f"\n   ⚠ Tools not advertised to LLM:")
            for tool in tool_check['not_advertised']:
                print(f"      - {tool}")
        else:
            print(f"\n   ✓ All tools advertised to LLM")

        if tool_check['not_dispatched']:
            print(f"\n   ⚠ Tools not in dispatch map:")
            for tool in tool_check['not_dispatched']:
                print(f"      - {tool}")
        else:
            print(f"\n   ✓ All tools in dispatch map")

    print()

    # 3. Check utility integrations
    print("3. CHECKING UTILITY MODULE INTEGRATIONS")
    print("-" * 80)
    utility_check = check_utility_integrations()

    for module, info in sorted(utility_check.items()):
        print(f"\n   {module}.py:")
        print(f"      Imported in: {', '.join(info['imported_in']) or 'NOWHERE ⚠'}")

        if info['unused']:
            print(f"      ⚠ Unused functions:")
            for fn in info['unused']:
                print(f"         - {fn}")
        else:
            print(f"      ✓ All expected functions used")

    print()

    # 4. Find unused modules
    print("4. CHECKING FOR UNUSED MODULES")
    print("-" * 80)
    unused = find_defined_but_unused_modules()

    if unused['unused_modules']:
        print(f"   ⚠ Modules defined but never imported:")
        for module in sorted(unused['unused_modules']):
            print(f"      - {module}.py")
    else:
        print("   ✓ All modules are imported somewhere")

    print()
    print("="*80)
    print("CHECK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
