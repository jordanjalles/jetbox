"""
Integration test for enhancement composition pattern.

Demonstrates that:
1. TaskExecutor uses AppendUntilFull + JetboxNotes by default
2. Multiple enhancements can be composed together
3. Strategy and enhancements work independently
"""
import tempfile
from pathlib import Path

from task_executor_agent import TaskExecutorAgent
from context_strategies import (
    AppendUntilFullStrategy,
    HierarchicalStrategy,
    JetboxNotesEnhancement,
)


def test_default_composition():
    """Test TaskExecutor default composition: AppendUntilFull + JetboxNotes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        agent = TaskExecutorAgent(workspace=Path(temp_dir))

        # Default strategy should be AppendUntilFull
        assert agent.context_strategy.get_name() == "append_until_full"

        # No enhancements initially
        assert len(agent.enhancements) == 0

        # Set goal adds JetboxNotesEnhancement
        agent.set_goal("test default composition")

        assert len(agent.enhancements) == 1
        assert isinstance(agent.enhancements[0], JetboxNotesEnhancement)

        print("✓ Default composition: AppendUntilFull + JetboxNotes")


def test_hierarchical_with_jetbox():
    """Test TaskExecutor with HierarchicalStrategy + JetboxNotes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        agent = TaskExecutorAgent(
            workspace=Path(temp_dir),
            context_strategy=HierarchicalStrategy()
        )

        # Strategy should be hierarchical
        assert agent.context_strategy.get_name() == "hierarchical"

        # Set goal adds JetboxNotesEnhancement
        agent.set_goal("test hierarchical composition")

        assert len(agent.enhancements) == 1
        assert isinstance(agent.enhancements[0], JetboxNotesEnhancement)

        # Should have hierarchical tools + no enhancement tools
        tools = agent.get_tools()
        tool_names = [t["function"]["name"] for t in tools]

        assert "mark_subtask_complete" in tool_names  # From hierarchical
        assert "decompose_task" in tool_names  # From hierarchical

        print("✓ Hierarchical + JetboxNotes composition works")


def test_system_prompt_composition():
    """Test that system prompt combines strategy + enhancement instructions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        agent = TaskExecutorAgent(
            workspace=Path(temp_dir),
            context_strategy=HierarchicalStrategy()
        )

        agent.set_goal("test prompt composition")

        prompt = agent.get_system_prompt()

        # Should include base prompt
        assert len(prompt) > 0

        # Should include hierarchical instructions
        assert "HIERARCHICAL WORKFLOW" in prompt
        assert "decompose_task" in prompt

        # Should include jetbox instructions
        assert "JETBOX NOTES" in prompt
        assert "automatically" in prompt.lower()

        print("✓ System prompt composes strategy + enhancement instructions")


def test_context_building_with_notes():
    """Test that context includes enhancement injections."""
    with tempfile.TemporaryDirectory() as temp_dir:
        agent = TaskExecutorAgent(workspace=Path(temp_dir))
        agent.set_goal("test context building")

        # Create some jetbox notes
        notes_file = agent.workspace_manager.workspace_dir / "jetboxnotes.md"
        notes_file.write_text(
            "# Jetbox Notes\n\n## Previous Work\n- Implemented feature A\n- Fixed bug B\n"
        )

        # Build context
        context = agent.build_context()

        # Find jetbox notes in context
        notes_found = False
        for msg in context:
            if msg.get("role") == "user" and "JETBOX NOTES" in msg.get("content", ""):
                notes_found = True
                assert "Implemented feature A" in msg["content"]
                break

        assert notes_found
        print("✓ Context building includes enhancement injections")


def test_enhancement_independence():
    """Test that enhancements work independently of strategy choice."""
    strategies = [
        AppendUntilFullStrategy(),
        HierarchicalStrategy(),
    ]

    for strategy in strategies:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = TaskExecutorAgent(
                workspace=Path(temp_dir),
                context_strategy=strategy
            )

            agent.set_goal(f"test {strategy.get_name()}")

            # Should always have JetboxNotesEnhancement
            assert len(agent.enhancements) == 1
            assert isinstance(agent.enhancements[0], JetboxNotesEnhancement)

            # System prompt should include jetbox instructions
            prompt = agent.get_system_prompt()
            assert "JETBOX NOTES" in prompt

    print("✓ Enhancements work independently of strategy choice")


if __name__ == "__main__":
    print("Testing enhancement composition pattern...\n")

    test_default_composition()
    test_hierarchical_with_jetbox()
    test_system_prompt_composition()
    test_context_building_with_notes()
    test_enhancement_independence()

    print("\n✓ All integration tests passed!")
