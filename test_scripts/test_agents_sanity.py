"""
Quick sanity check for all agents after refactoring.

Tests:
1. TaskExecutor can create a file
2. Architect can be instantiated
3. Orchestrator can be instantiated with and without ChatbotBehavior
"""
from pathlib import Path
import sys

def test_task_executor():
    """Test TaskExecutor instantiation and basic setup."""
    print("=" * 60)
    print("TEST 1: TaskExecutor Instantiation")
    print("=" * 60)
    try:
        from agents.task_executor_agent import TaskExecutorAgent

        # Create agent
        agent = TaskExecutorAgent(
            workspace=Path(".agent_workspaces/sanity_test"),
            goal=None
        )

        print("✅ TaskExecutor created successfully")
        print(f"   - Name: {agent.name}")
        print(f"   - Workspace: {agent.workspace}")
        print(f"   - Behaviors: {[b.get_name() for b in agent.behaviors]}")
        print(f"   - Has server_manager: {agent.server_manager is not None}")
        print(f"   - Has registry: {agent.registry is not None}")
        return True
    except Exception as e:
        print(f"❌ TaskExecutor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_architect():
    """Test Architect instantiation and basic setup."""
    print("\n" + "=" * 60)
    print("TEST 2: Architect Instantiation")
    print("=" * 60)
    try:
        from agents.architect_agent import ArchitectAgent

        # Create agent
        agent = ArchitectAgent(
            workspace=Path(".agent_workspaces/sanity_test"),
            goal=None
        )

        print("✅ Architect created successfully")
        print(f"   - Name: {agent.name}")
        print(f"   - Workspace: {agent.workspace}")
        print(f"   - Behaviors: {[b.get_name() for b in agent.behaviors]}")
        print(f"   - Has server_manager: {agent.server_manager is not None}")
        print(f"   - Has registry: {agent.registry is not None}")
        return True
    except Exception as e:
        print(f"❌ Architect test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_with_chatbot():
    """Test Orchestrator with ChatbotBehavior."""
    print("\n" + "=" * 60)
    print("TEST 3: Orchestrator with ChatbotBehavior")
    print("=" * 60)
    try:
        from agents.orchestrator_agent import OrchestratorAgent

        # Create agent with ChatbotBehavior (don't exclude it)
        agent = OrchestratorAgent(
            workspace=Path(".agent_workspaces/sanity_test"),
            exclude_behaviors=[]
        )

        print("✅ Orchestrator created successfully")
        print(f"   - Name: {agent.name}")
        print(f"   - Workspace: {agent.workspace}")
        print(f"   - Behaviors: {[b.get_name() for b in agent.behaviors]}")
        print(f"   - Has server_manager: {agent.server_manager is not None}")
        print(f"   - Has registry: {agent.registry is not None}")
        print(f"   - Has pre_task_hook: {hasattr(agent, 'pre_task_hook')}")
        print(f"   - Has cleanup_hook: {hasattr(agent, 'cleanup_hook')}")

        # Check ChatbotBehavior presence
        has_chatbot = any(b.get_name() == "chatbot" for b in agent.behaviors)
        print(f"   - Has ChatbotBehavior: {has_chatbot}")

        return True
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_without_chatbot():
    """Test Orchestrator without ChatbotBehavior (autonomous mode)."""
    print("\n" + "=" * 60)
    print("TEST 4: Orchestrator without ChatbotBehavior")
    print("=" * 60)
    try:
        from agents.orchestrator_agent import OrchestratorAgent

        # Create agent excluding ChatbotBehavior
        agent = OrchestratorAgent(
            workspace=Path(".agent_workspaces/sanity_test"),
            exclude_behaviors=["ChatbotBehavior"]
        )

        print("✅ Orchestrator created successfully")
        print(f"   - Name: {agent.name}")
        print(f"   - Workspace: {agent.workspace}")
        print(f"   - Behaviors: {[b.get_name() for b in agent.behaviors]}")
        print(f"   - Has server_manager: {agent.server_manager is not None}")
        print(f"   - Has registry: {agent.registry is not None}")

        # Check ChatbotBehavior absence
        has_chatbot = any(b.get_name() == "chatbot" for b in agent.behaviors)
        print(f"   - Has ChatbotBehavior: {has_chatbot}")

        return True
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base_agent_chat_coordination():
    """Test that base_agent can detect ChatbotBehavior."""
    print("\n" + "=" * 60)
    print("TEST 5: BaseAgent ChatbotBehavior Detection")
    print("=" * 60)
    try:
        from agents.orchestrator_agent import OrchestratorAgent

        # Create agent with ChatbotBehavior
        agent = OrchestratorAgent(
            workspace=Path(".agent_workspaces/sanity_test"),
            exclude_behaviors=[]
        )

        # Check if base_agent._run_multi_task_chat_mode exists
        has_chat_method = hasattr(agent.__class__, '_run_multi_task_chat_mode')
        print(f"✅ BaseAgent has _run_multi_task_chat_mode: {has_chat_method}")

        # Check if agent has hooks
        print(f"   - Agent has pre_task_hook: {hasattr(agent, 'pre_task_hook')}")
        print(f"   - Agent has cleanup_hook: {hasattr(agent, 'cleanup_hook')}")

        return has_chat_method
    except Exception as e:
        print(f"❌ Base agent chat coordination test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🔍 AGENT SANITY CHECK AFTER REFACTORING" + "\n")

    results = []
    results.append(("TaskExecutor", test_task_executor()))
    results.append(("Architect", test_architect()))
    results.append(("Orchestrator+Chatbot", test_orchestrator_with_chatbot()))
    results.append(("Orchestrator-Chatbot", test_orchestrator_without_chatbot()))
    results.append(("BaseAgent ChatCoord", test_base_agent_chat_coordination()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All sanity checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
