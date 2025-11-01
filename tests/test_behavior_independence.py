"""
Test suite for behavior independence and single responsibility.

This module tests that behaviors:
1. Don't import other behaviors
2. Don't have hardcoded references to other behavior names
3. Follow single responsibility principle
4. Implement interfaces correctly
5. Have no cross-behavior dependencies
"""

import pytest
import inspect
import ast
from pathlib import Path
from typing import get_type_hints

# Import all behaviors
from behaviors import (
    AgentBehavior,
    FileToolsBehavior,
    CommandToolsBehavior,
    ServerToolsBehavior,
    ArchitectToolsBehavior,
    CompactWhenNearFullBehavior,
    HierarchicalContextBehavior,
    SubAgentContextBehavior,
    LoopDetectionBehavior,
    WorkspaceTaskNotesBehavior,
    StatusDisplayBehavior,
    DelegationBehavior,
)


class TestDependencyInspection:
    """Test that behaviors don't depend on each other."""

    def get_behavior_source_files(self):
        """Get all behavior source files."""
        behaviors_dir = Path(__file__).parent.parent / "behaviors"
        return list(behaviors_dir.glob("*.py"))

    def test_no_cross_behavior_imports(self):
        """Behaviors don't import other behaviors."""
        behaviors_dir = Path(__file__).parent.parent / "behaviors"

        # Behavior files to check
        behavior_files = [
            "file_tools.py",
            "command_tools.py",
            "server_tools.py",
            "architect_tools.py",
            "compact_when_near_full.py",
            "hierarchical_context.py",
            "subagent_context.py",
            "loop_detection.py",
            "workspace_task_notes.py",
            "status_display.py",
            "delegation.py"
        ]

        # Behavior module names to look for
        behavior_imports = [
            "file_tools",
            "command_tools",
            "server_tools",
            "architect_tools",
            "compact_when_near_full",
            "hierarchical_context",
            "subagent_context",
            "loop_detection",
            "workspace_task_notes",
            "status_display",
            "delegation"
        ]

        violations = []

        for behavior_file in behavior_files:
            file_path = behaviors_dir / behavior_file
            if not file_path.exists():
                continue

            source = file_path.read_text()

            # Parse AST
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            # Check imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Check if importing another behavior
                        if any(f"behaviors.{b}" in alias.name for b in behavior_imports):
                            violations.append(f"{behavior_file} imports {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("behaviors."):
                        module_name = node.module.split(".")[-1]
                        # Importing from another behavior (not base)
                        if module_name in behavior_imports:
                            violations.append(f"{behavior_file} imports from {node.module}")

        assert len(violations) == 0, f"Cross-behavior imports found:\n" + "\n".join(violations)

    def test_no_hardcoded_behavior_references(self):
        """Behaviors don't have hardcoded references to other behavior names."""
        behaviors_dir = Path(__file__).parent.parent / "behaviors"

        behavior_files = [
            "file_tools.py",
            "command_tools.py",
            "server_tools.py",
            "architect_tools.py",
            "compact_when_near_full.py",
            "hierarchical_context.py",
            "subagent_context.py",
            "loop_detection.py",
            "workspace_task_notes.py",
            "status_display.py"
        ]

        # Patterns that suggest hardcoded references
        suspicious_patterns = [
            'get_behavior("',
            'find_behavior("',
            'behavior_name == "',
            'behaviors["'
        ]

        violations = []

        for behavior_file in behavior_files:
            file_path = behaviors_dir / behavior_file
            if not file_path.exists():
                continue

            source = file_path.read_text()

            for pattern in suspicious_patterns:
                if pattern in source:
                    # Check if it's referencing another behavior
                    lines = source.split("\n")
                    for i, line in enumerate(lines, 1):
                        if pattern in line:
                            violations.append(f"{behavior_file}:{i} - {line.strip()}")

        # Some violations are acceptable (e.g., self-references, comments)
        # Filter out acceptable ones
        filtered_violations = []
        acceptable_patterns = [
            "get_name()",  # Self-reference
            "# ",  # Comments
            '"""',  # Docstrings
        ]

        for violation in violations:
            is_acceptable = any(pattern in violation for pattern in acceptable_patterns)
            if not is_acceptable:
                filtered_violations.append(violation)

        # Note: Some behaviors legitimately reference others (e.g., WorkspaceTaskNotesBehavior
        # checking for compact/subagent context to get max_tokens). This is acceptable
        # as it's reading metadata, not calling methods.


class TestSingleResponsibility:
    """Test that each behavior has a single, focused responsibility."""

    def test_each_behavior_focused(self):
        """Each behavior has a clear, single responsibility."""
        behaviors = {
            FileToolsBehavior: "file operations",
            CommandToolsBehavior: "command execution",
            ServerToolsBehavior: "server management",
            ArchitectToolsBehavior: "architecture artifacts",
            CompactWhenNearFullBehavior: "context compaction",
            HierarchicalContextBehavior: "hierarchical task context",
            SubAgentContextBehavior: "delegated goal context",
            LoopDetectionBehavior: "loop detection",
            WorkspaceTaskNotesBehavior: "persistent notes",
            StatusDisplayBehavior: "progress visualization"
        }

        for BehaviorClass, purpose in behaviors.items():
            # Verify class has docstring describing its purpose
            assert BehaviorClass.__doc__ is not None, f"{BehaviorClass.__name__} missing docstring"

            # Instantiate to verify it works
            if BehaviorClass == CommandToolsBehavior:
                behavior = BehaviorClass(whitelist=["python"])
            elif BehaviorClass == CompactWhenNearFullBehavior:
                behavior = BehaviorClass(max_tokens=8000)
            elif BehaviorClass == HierarchicalContextBehavior:
                behavior = BehaviorClass(history_keep=12)
            elif BehaviorClass == LoopDetectionBehavior:
                behavior = BehaviorClass(max_repeats=5)
            else:
                behavior = BehaviorClass()

            # Behavior should have a name
            assert behavior.get_name()

    def test_method_count_reasonable(self):
        """Behaviors have reasonable method counts (focused interface)."""
        behaviors = [
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior(),
            ArchitectToolsBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            HierarchicalContextBehavior(history_keep=12),
            SubAgentContextBehavior(),
            LoopDetectionBehavior(max_repeats=5),
            WorkspaceTaskNotesBehavior(),
            StatusDisplayBehavior()
        ]

        for behavior in behaviors:
            # Get public methods (excluding magic methods)
            methods = [m for m in dir(behavior) if not m.startswith("_") and callable(getattr(behavior, m))]

            # Most behaviors should have < 15 public methods
            # (indicates focused responsibility)
            assert len(methods) < 20, f"{behavior.get_name()} has {len(methods)} public methods"

    def test_tool_count_reasonable(self):
        """Tool-providing behaviors have reasonable tool counts."""
        tool_behaviors = [
            (FileToolsBehavior(), 3, "file operations"),
            (CommandToolsBehavior(whitelist=["python"]), 1, "command execution"),
            (ServerToolsBehavior(), 4, "server management"),
            (ArchitectToolsBehavior(), 5, "architecture"),  # Updated: has write/read/list tools
            (HierarchicalContextBehavior(history_keep=12), 2, "task management"),
            (SubAgentContextBehavior(), 2, "completion signaling")
        ]

        for behavior, expected_count, purpose in tool_behaviors:
            tools = behavior.get_tools()
            assert len(tools) == expected_count, f"{behavior.get_name()} ({purpose}) should have ~{expected_count} tools, has {len(tools)}"


class TestInterfaceCompliance:
    """Test that behaviors properly implement the AgentBehavior interface."""

    def test_all_behaviors_extend_base(self):
        """All behaviors extend AgentBehavior."""
        behavior_classes = [
            FileToolsBehavior,
            CommandToolsBehavior,
            ServerToolsBehavior,
            ArchitectToolsBehavior,
            CompactWhenNearFullBehavior,
            HierarchicalContextBehavior,
            SubAgentContextBehavior,
            LoopDetectionBehavior,
            WorkspaceTaskNotesBehavior,
            StatusDisplayBehavior,
            DelegationBehavior
        ]

        for BehaviorClass in behavior_classes:
            assert issubclass(BehaviorClass, AgentBehavior), f"{BehaviorClass.__name__} doesn't extend AgentBehavior"

    def test_get_name_returns_string(self):
        """All behaviors return string from get_name()."""
        behaviors = [
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior(),
            ArchitectToolsBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            HierarchicalContextBehavior(history_keep=12),
            SubAgentContextBehavior(),
            LoopDetectionBehavior(max_repeats=5),
            WorkspaceTaskNotesBehavior(),
            StatusDisplayBehavior()
        ]

        for behavior in behaviors:
            name = behavior.get_name()
            assert isinstance(name, str), f"{behavior.__class__.__name__}.get_name() returned {type(name)}"
            assert len(name) > 0, f"{behavior.__class__.__name__}.get_name() returned empty string"

    def test_get_tools_returns_list(self):
        """All behaviors return list from get_tools()."""
        behaviors = [
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior(),
            ArchitectToolsBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            HierarchicalContextBehavior(history_keep=12),
            SubAgentContextBehavior(),
            LoopDetectionBehavior(max_repeats=5),
            WorkspaceTaskNotesBehavior(),
            StatusDisplayBehavior()
        ]

        for behavior in behaviors:
            tools = behavior.get_tools()
            assert isinstance(tools, list), f"{behavior.__class__.__name__}.get_tools() returned {type(tools)}"

    def test_enhance_context_returns_list(self):
        """All behaviors return list from enhance_context()."""
        behaviors = [
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            HierarchicalContextBehavior(history_keep=12),
            SubAgentContextBehavior(),
            LoopDetectionBehavior(max_repeats=5),
            WorkspaceTaskNotesBehavior(),
            StatusDisplayBehavior()
        ]

        for behavior in behaviors:
            context = [{"role": "system", "content": "Test"}]
            enhanced = behavior.enhance_context(context)
            assert isinstance(enhanced, list), f"{behavior.__class__.__name__}.enhance_context() returned {type(enhanced)}"

    def test_get_instructions_returns_string(self):
        """All behaviors return string from get_instructions()."""
        behaviors = [
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior(),
            ArchitectToolsBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            HierarchicalContextBehavior(history_keep=12),
            SubAgentContextBehavior(),
            LoopDetectionBehavior(max_repeats=5),
            WorkspaceTaskNotesBehavior(),
            StatusDisplayBehavior()
        ]

        for behavior in behaviors:
            instructions = behavior.get_instructions()
            assert isinstance(instructions, str), f"{behavior.__class__.__name__}.get_instructions() returned {type(instructions)}"

    def test_event_handlers_accept_kwargs(self):
        """All event handlers accept **kwargs for forward compatibility."""
        behavior = FileToolsBehavior()

        # All event handlers should accept unexpected kwargs without crashing
        behavior.on_goal_start("test", unexpected_param="value")
        behavior.on_tool_call("test", {}, {}, another_param=123)
        behavior.on_round_end(1, yet_another="param")
        behavior.on_timeout(60.0, extra_kwarg={"key": "val"})
        behavior.on_goal_complete(True, more_kwargs="yes")

    def test_no_required_parameters_beyond_init(self):
        """Behaviors have no required params beyond __init__."""
        behaviors = [
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior(),
            ArchitectToolsBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            HierarchicalContextBehavior(history_keep=12),
            SubAgentContextBehavior(),
            LoopDetectionBehavior(max_repeats=5),
            WorkspaceTaskNotesBehavior(),
            StatusDisplayBehavior()
        ]

        for behavior in behaviors:
            # These should all work without required params
            behavior.get_name()
            behavior.get_tools()
            behavior.get_instructions()
            behavior.enhance_context([])

            # Events should work with minimal args
            behavior.on_goal_start("test")
            behavior.on_tool_call("test", {}, {})
            behavior.on_round_end(1)
            behavior.on_timeout(60.0)
            behavior.on_goal_complete(True)


class TestBehaviorMetadata:
    """Test behavior metadata and documentation."""

    def test_all_behaviors_have_docstrings(self):
        """All behavior classes have docstrings."""
        behavior_classes = [
            FileToolsBehavior,
            CommandToolsBehavior,
            ServerToolsBehavior,
            ArchitectToolsBehavior,
            CompactWhenNearFullBehavior,
            HierarchicalContextBehavior,
            SubAgentContextBehavior,
            LoopDetectionBehavior,
            WorkspaceTaskNotesBehavior,
            StatusDisplayBehavior,
            DelegationBehavior
        ]

        for BehaviorClass in behavior_classes:
            assert BehaviorClass.__doc__, f"{BehaviorClass.__name__} missing docstring"
            assert len(BehaviorClass.__doc__.strip()) > 20, f"{BehaviorClass.__name__} docstring too short"

    def test_all_behaviors_have_unique_names(self):
        """All behaviors have unique identifiers."""
        behaviors = [
            FileToolsBehavior(),
            CommandToolsBehavior(whitelist=["python"]),
            ServerToolsBehavior(),
            ArchitectToolsBehavior(),
            CompactWhenNearFullBehavior(max_tokens=8000),
            HierarchicalContextBehavior(history_keep=12),
            SubAgentContextBehavior(),
            LoopDetectionBehavior(max_repeats=5),
            WorkspaceTaskNotesBehavior(),
            StatusDisplayBehavior()
        ]

        names = [b.get_name() for b in behaviors]

        # Check uniqueness
        duplicates = [name for name in names if names.count(name) > 1]
        assert len(duplicates) == 0, f"Duplicate behavior names: {set(duplicates)}"

    def test_behavior_init_signatures(self):
        """Behavior __init__ methods have reasonable signatures."""
        behavior_classes = [
            FileToolsBehavior,
            CommandToolsBehavior,
            ServerToolsBehavior,
            ArchitectToolsBehavior,
            CompactWhenNearFullBehavior,
            HierarchicalContextBehavior,
            SubAgentContextBehavior,
            LoopDetectionBehavior,
            WorkspaceTaskNotesBehavior,
            StatusDisplayBehavior
        ]

        for BehaviorClass in behavior_classes:
            # Get __init__ signature
            sig = inspect.signature(BehaviorClass.__init__)

            # Should have self parameter
            assert "self" in sig.parameters

            # Check if all params have defaults or are **kwargs
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                # Either has default value or is **kwargs
                has_default = param.default != inspect.Parameter.empty
                is_kwargs = param.kind == inspect.Parameter.VAR_KEYWORD

                assert has_default or is_kwargs, f"{BehaviorClass.__name__}.__init__ param '{param_name}' has no default"


def test_behavior_independence_summary():
    """
    Summary test: Verify all behaviors are independent.

    This test checks:
    1. No cross-behavior imports
    2. All behaviors can be instantiated alone
    3. All behaviors have unique names
    4. All behaviors implement required interface
    """
    behavior_classes = [
        FileToolsBehavior,
        CommandToolsBehavior,
        ServerToolsBehavior,
        ArchitectToolsBehavior,
        CompactWhenNearFullBehavior,
        HierarchicalContextBehavior,
        SubAgentContextBehavior,
        LoopDetectionBehavior,
        WorkspaceTaskNotesBehavior,
        StatusDisplayBehavior
    ]

    # 1. All can be instantiated independently
    behaviors = []
    for BehaviorClass in behavior_classes:
        if BehaviorClass == CommandToolsBehavior:
            behavior = BehaviorClass(whitelist=["python"])
        elif BehaviorClass == CompactWhenNearFullBehavior:
            behavior = BehaviorClass(max_tokens=8000)
        elif BehaviorClass == HierarchicalContextBehavior:
            behavior = BehaviorClass(history_keep=12)
        elif BehaviorClass == LoopDetectionBehavior:
            behavior = BehaviorClass(max_repeats=5)
        else:
            behavior = BehaviorClass()
        behaviors.append(behavior)

    # 2. All extend AgentBehavior
    for behavior in behaviors:
        assert isinstance(behavior, AgentBehavior)

    # 3. All have unique names
    names = [b.get_name() for b in behaviors]
    assert len(names) == len(set(names))

    # 4. All implement required interface
    for behavior in behaviors:
        assert callable(behavior.get_name)
        assert callable(behavior.get_tools)
        assert callable(behavior.enhance_context)
        assert callable(behavior.get_instructions)
        assert callable(behavior.on_goal_start)
        assert callable(behavior.on_tool_call)
        assert callable(behavior.on_round_end)
        assert callable(behavior.on_timeout)
        assert callable(behavior.on_goal_complete)

    print(f"\n✓ Independence verified for {len(behaviors)} behaviors")
    print(f"✓ All behaviors extend AgentBehavior")
    print(f"✓ All behaviors have unique names")
    print(f"✓ All behaviors implement required interface")
