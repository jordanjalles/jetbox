#!/usr/bin/env python3
"""
Quick verification tests for bug fixes.

Tests:
1. TaskExecutor _behaviors attribute fix
2. OrchestratorAgent optional workspace parameter
3. ArchitectAgent optional workspace parameter
"""
import sys
from pathlib import Path
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_task_executor_behaviors():
    """Test that TaskExecutor can access _behaviors attribute."""
    print("\n=== TEST 1: TaskExecutor _behaviors attribute ===")
    try:
        from task_executor_agent import TaskExecutorAgent

        # Create agent with use_behaviors=True
        agent = TaskExecutorAgent(
            workspace=Path("/tmp/test_executor"),
            goal="test goal",
            use_behaviors=True
        )

        # Verify _behaviors exists
        assert hasattr(agent, '_behaviors'), "Missing _behaviors attribute"
        assert isinstance(agent._behaviors, list), "_behaviors should be a list"

        # Verify behaviors exists (public alias)
        assert hasattr(agent, 'behaviors'), "Missing behaviors attribute"
        assert isinstance(agent.behaviors, list), "behaviors should be a list"

        # Verify they point to same object
        assert agent._behaviors is agent.behaviors, "_behaviors and behaviors should be same object"

        print(f"✓ TaskExecutor has _behaviors: {len(agent._behaviors)} behaviors loaded")
        print(f"✓ TaskExecutor has behaviors: {len(agent.behaviors)} behaviors loaded")
        print(f"✓ _behaviors and behaviors point to same list")

        # Verify behavior loading works
        behavior_names = [b.get_name() for b in agent._behaviors]
        print(f"✓ Loaded behaviors: {', '.join(behavior_names)}")

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False

def test_orchestrator_workspace():
    """Test that OrchestratorAgent can be instantiated without workspace."""
    print("\n=== TEST 2: OrchestratorAgent optional workspace ===")
    try:
        from orchestrator_agent import OrchestratorAgent

        # Test 1: No workspace provided
        agent1 = OrchestratorAgent(use_behaviors=True)
        assert hasattr(agent1, '_behaviors'), "Missing _behaviors attribute"
        assert hasattr(agent1, 'workspace'), "Missing workspace attribute"
        print(f"✓ Orchestrator instantiated without workspace (workspace={agent1.workspace})")
        print(f"✓ Orchestrator has {len(agent1._behaviors)} behaviors")

        # Test 2: With workspace provided
        agent2 = OrchestratorAgent(workspace=Path("/tmp/test_orch"), use_behaviors=True)
        assert agent2.workspace == Path("/tmp/test_orch")
        print(f"✓ Orchestrator instantiated with workspace={agent2.workspace}")

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False

def test_architect_workspace():
    """Test that ArchitectAgent can be instantiated without workspace."""
    print("\n=== TEST 3: ArchitectAgent optional workspace ===")
    try:
        from architect_agent import ArchitectAgent

        # Test 1: No workspace provided
        agent1 = ArchitectAgent(use_behaviors=True)
        assert hasattr(agent1, '_behaviors'), "Missing _behaviors attribute"
        assert hasattr(agent1, 'workspace'), "Missing workspace attribute"
        print(f"✓ Architect instantiated without workspace (workspace={agent1.workspace})")
        print(f"✓ Architect has {len(agent1._behaviors)} behaviors")

        # Test 2: With workspace provided
        agent2 = ArchitectAgent(workspace=Path("/tmp/test_arch"), use_behaviors=True)
        assert agent2.workspace == Path("/tmp/test_arch")
        print(f"✓ Architect instantiated with workspace={agent2.workspace}")

        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("=" * 70)
    print("QUICK FIX VERIFICATION TESTS")
    print("=" * 70)

    results = []

    # Test 1: TaskExecutor _behaviors
    results.append(("TaskExecutor _behaviors", test_task_executor_behaviors()))

    # Test 2: Orchestrator workspace
    results.append(("Orchestrator workspace", test_orchestrator_workspace()))

    # Test 3: Architect workspace
    results.append(("Architect workspace", test_architect_workspace()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL FIXES VERIFIED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) still failing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
