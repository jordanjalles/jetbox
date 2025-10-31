"""
Test JetboxNotesEnhancement integration with TaskExecutor.

Verifies that:
1. JetboxNotesEnhancement is added automatically on set_goal()
2. Enhancement composes with AppendUntilFull strategy
3. Notes are loaded if they exist
4. Context injection works correctly
"""
import pytest
from pathlib import Path
import tempfile
import shutil

from task_executor_agent import TaskExecutorAgent
from context_strategies import JetboxNotesEnhancement, AppendUntilFullStrategy
import jetbox_notes


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_jetbox_notes_enhancement_added_on_set_goal(temp_workspace):
    """Test that JetboxNotesEnhancement is added when goal is set."""
    agent = TaskExecutorAgent(
        workspace=temp_workspace,
        max_rounds=5,
        context_strategy=AppendUntilFullStrategy()
    )

    # Initially no enhancements
    assert len(agent.enhancements) == 0

    # Set goal should add JetboxNotesEnhancement
    agent.set_goal("test goal")

    # Check that enhancement was added
    assert len(agent.enhancements) == 1
    assert isinstance(agent.enhancements[0], JetboxNotesEnhancement)
    print("✓ JetboxNotesEnhancement added on set_goal()")


def test_enhancement_context_injection_with_no_notes(temp_workspace):
    """Test that enhancement returns None when no notes exist."""
    agent = TaskExecutorAgent(
        workspace=temp_workspace,
        max_rounds=5,
        context_strategy=AppendUntilFullStrategy()
    )

    agent.set_goal("test goal")

    # Get the enhancement
    enhancement = agent.enhancements[0]

    # Should return None when no notes exist
    context_injection = enhancement.get_context_injection(
        context_manager=agent.context_manager,
        workspace=agent.workspace_manager.workspace_dir
    )

    assert context_injection is None
    print("✓ Enhancement returns None when no notes exist")


def test_enhancement_context_injection_with_notes(temp_workspace):
    """Test that enhancement injects notes when they exist."""
    agent = TaskExecutorAgent(
        workspace=temp_workspace,
        max_rounds=5,
        context_strategy=AppendUntilFullStrategy()
    )

    agent.set_goal("test goal")

    # Create some notes manually
    notes_file = agent.workspace_manager.workspace_dir / "jetboxnotes.md"
    notes_file.write_text("# Jetbox Notes\n\n## Previous work\n- Created feature X\n- Fixed bug Y\n")

    # Get the enhancement
    enhancement = agent.enhancements[0]

    # Should return context injection with notes
    context_injection = enhancement.get_context_injection(
        context_manager=agent.context_manager,
        workspace=agent.workspace_manager.workspace_dir
    )

    assert context_injection is not None
    assert context_injection["role"] == "user"
    assert "JETBOX NOTES" in context_injection["content"]
    assert "Created feature X" in context_injection["content"]
    print("✓ Enhancement injects notes when they exist")


def test_enhancement_provides_no_tools():
    """Test that JetboxNotesEnhancement provides no tools."""
    enhancement = JetboxNotesEnhancement()
    tools = enhancement.get_enhancement_tools()

    assert tools == []
    print("✓ Enhancement provides no tools (notes are auto-generated)")


def test_enhancement_provides_instructions():
    """Test that JetboxNotesEnhancement provides instructions."""
    enhancement = JetboxNotesEnhancement()
    instructions = enhancement.get_enhancement_instructions()

    assert instructions
    assert "JETBOX NOTES" in instructions
    assert "automatically" in instructions.lower()
    print("✓ Enhancement provides instructions about automatic note-taking")


def test_build_context_includes_enhancement():
    """Test that build_context() includes enhancement context."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_workspace = Path(temp_dir)

        agent = TaskExecutorAgent(
            workspace=temp_workspace,
            max_rounds=5,
            context_strategy=AppendUntilFullStrategy()
        )

        agent.set_goal("test goal with notes")

        # Create some notes
        notes_file = agent.workspace_manager.workspace_dir / "jetboxnotes.md"
        notes_file.write_text("# Jetbox Notes\n\n## Task Complete\n- Implemented feature Z\n")

        # Build context
        context = agent.build_context()

        # Context should have: [system, goal, notes, ...messages]
        # Check that notes are injected
        notes_found = False
        for msg in context:
            if msg.get("role") == "user" and "JETBOX NOTES" in msg.get("content", ""):
                notes_found = True
                assert "Implemented feature Z" in msg["content"]
                break

        assert notes_found, "Notes should be injected into context"
        print("✓ build_context() includes enhancement context")


def test_system_prompt_includes_enhancement_instructions(temp_workspace):
    """Test that system prompt includes enhancement instructions."""
    agent = TaskExecutorAgent(
        workspace=temp_workspace,
        max_rounds=5,
        context_strategy=AppendUntilFullStrategy()
    )

    agent.set_goal("test goal")

    # Get system prompt
    system_prompt = agent.get_system_prompt()

    # Should include enhancement instructions
    assert "JETBOX NOTES" in system_prompt
    assert "automatically" in system_prompt.lower()
    print("✓ System prompt includes enhancement instructions")


if __name__ == "__main__":
    print("Testing JetboxNotesEnhancement integration...\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_workspace = Path(temp_dir)

        test_jetbox_notes_enhancement_added_on_set_goal(temp_workspace)
        test_enhancement_context_injection_with_no_notes(temp_workspace)
        test_enhancement_context_injection_with_notes(temp_workspace)
        test_enhancement_provides_no_tools()
        test_enhancement_provides_instructions()
        test_build_context_includes_enhancement()
        test_system_prompt_includes_enhancement_instructions(temp_workspace)

    print("\n✓ All tests passed!")
