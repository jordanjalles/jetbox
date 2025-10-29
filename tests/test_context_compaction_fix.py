"""
Test that context compaction actually updates state.messages.

This test verifies the fix for the bug where self.state.messages
would grow infinitely because compaction only affected the context
sent to the LLM, not the stored state.
"""
from pathlib import Path
import tempfile

from orchestrator_agent import OrchestratorAgent


def test_compaction_actually_updates_state():
    """
    Test that when compaction happens, self.state.messages is updated.

    Before fix:
    - self.state.messages grows infinitely
    - Compaction only affects context, not state
    - Eventually Ollama hangs/crashes

    After fix:
    - self.state.messages is replaced with compacted version
    - Message count stabilizes around 21 messages
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = OrchestratorAgent(workspace=Path(tmpdir))

        # Add enough messages to trigger compaction
        # Token threshold is 80% of 8000 = 6400 tokens
        # Estimate: 4 chars per token, so ~25,600 chars
        # Each message ~100 chars = need ~256 messages

        print("Adding messages to trigger compaction...")
        for i in range(300):
            # Alternate user and assistant messages
            if i % 2 == 0:
                orch.add_message({
                    "role": "user",
                    "content": f"Test message {i} with some additional content to increase token count " * 5
                })
            else:
                orch.add_message({
                    "role": "assistant",
                    "content": f"Response {i} with some additional content to increase token count " * 5
                })

        # Check state before building context
        messages_before = len(orch.state.messages)
        print(f"Messages before build_context: {messages_before}")

        # Build context (should trigger compaction)
        context = orch.build_context()

        # Check state after building context
        messages_after = len(orch.state.messages)
        print(f"Messages after build_context: {messages_after}")

        # Assertions
        assert messages_before == 300, f"Expected 300 messages before, got {messages_before}"
        assert messages_after <= 21, f"Expected ~21 messages after compaction, got {messages_after}"
        assert messages_after < messages_before, "Compaction should reduce message count"

        print(f"✓ Compaction reduced messages from {messages_before} to {messages_after}")

        # Build context again - should NOT trigger compaction (already compacted)
        messages_before_2 = len(orch.state.messages)
        context2 = orch.build_context()
        messages_after_2 = len(orch.state.messages)

        assert messages_after_2 == messages_before_2, "No compaction should occur on already-compacted state"
        print(f"✓ Second build_context did not re-compact (stayed at {messages_after_2})")

        # Add a few more messages
        for i in range(5):
            orch.add_message({
                "role": "user",
                "content": f"New message {i} " * 10
            })

        messages_before_3 = len(orch.state.messages)
        print(f"Messages after adding 5 more: {messages_before_3}")
        assert messages_before_3 == messages_after_2 + 5, "Should have added 5 messages"

        # Build context again - might or might not trigger compaction depending on size
        context3 = orch.build_context()
        messages_after_3 = len(orch.state.messages)
        print(f"Messages after third build_context: {messages_after_3}")

        # The key test: messages should be stable, not growing infinitely
        assert messages_after_3 <= 30, f"Messages should stabilize below 30, got {messages_after_3}"

        print("✓ Message count stabilized - compaction fix working!")


def test_compaction_preserves_recent_messages():
    """Test that compaction keeps the 20 most recent messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orch = OrchestratorAgent(workspace=Path(tmpdir))

        # Add messages with identifiable content
        for i in range(100):
            orch.add_message({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message number {i} with unique identifier {i*1000} " * 10
            })

        # Build context (trigger compaction)
        context = orch.build_context()

        # Check that recent messages are preserved
        last_message = orch.state.messages[-1]
        assert "Message number 99" in last_message["content"], "Last message should be preserved"

        # First message should be summary (not original message 0)
        first_message = orch.state.messages[0]
        assert "Earlier conversation summary" in first_message["content"], "First message should be summary"

        print("✓ Compaction preserves recent messages and adds summary")


if __name__ == "__main__":
    test_compaction_actually_updates_state()
    test_compaction_preserves_recent_messages()
    print("\n✅ All compaction tests passed!")
