#!/usr/bin/env python3
"""
Quick test to verify the bug fixes work correctly.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("QUICK FIX VERIFICATION TEST")
print("="*80)

# Test 1: TaskExecutor with behaviors
print("\n[Test 1] TaskExecutor with use_behaviors=True")
try:
    from agents.task_executor_agent import TaskExecutorAgent

    agent = TaskExecutorAgent(
        workspace=Path("/tmp/test_executor"),
        goal="test goal",
        use_behaviors=True
    )

    # Check _behaviors attribute exists
    assert hasattr(agent, '_behaviors'), "Missing _behaviors attribute"
    assert isinstance(agent._behaviors, list), "_behaviors should be a list"

    # Check behaviors were loaded
    behavior_count = len(agent._behaviors)
    print("✓ TaskExecutor created successfully")
    print("  - _behaviors attribute: OK")
    print(f"  - Behaviors loaded: {behavior_count}")
    print(f"  - Behavior names: {[b.get_name() for b in agent._behaviors]}")

except Exception as e:
    print(f"✗ TaskExecutor FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Orchestrator without workspace parameter
print("\n[Test 2] Orchestrator without workspace parameter")
try:
    from agents.orchestrator_agent import OrchestratorAgent

    agent = OrchestratorAgent(use_behaviors=True)

    assert hasattr(agent, '_behaviors'), "Missing _behaviors attribute"
    assert isinstance(agent._behaviors, list), "_behaviors should be a list"

    behavior_count = len(agent._behaviors)
    print("✓ Orchestrator created successfully")
    print("  - _behaviors attribute: OK")
    print(f"  - Behaviors loaded: {behavior_count}")
    print(f"  - Workspace: {agent.workspace}")

except Exception as e:
    print(f"✗ Orchestrator FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Orchestrator with workspace parameter
print("\n[Test 3] Orchestrator with workspace parameter")
try:
    from agents.orchestrator_agent import OrchestratorAgent

    agent = OrchestratorAgent(
        workspace=Path("/tmp/test_orchestrator"),
        use_behaviors=True
    )

    assert hasattr(agent, '_behaviors'), "Missing _behaviors attribute"

    print("✓ Orchestrator (with workspace) created successfully")
    print(f"  - Workspace: {agent.workspace}")

except Exception as e:
    print(f"✗ Orchestrator (with workspace) FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Architect without workspace parameter
print("\n[Test 4] Architect without workspace parameter")
try:
    from agents.architect_agent import ArchitectAgent

    agent = ArchitectAgent(use_behaviors=True)

    assert hasattr(agent, '_behaviors'), "Missing _behaviors attribute"

    behavior_count = len(agent._behaviors)
    print("✓ Architect created successfully")
    print("  - _behaviors attribute: OK")
    print(f"  - Behaviors loaded: {behavior_count}")
    print(f"  - Workspace: {agent.workspace}")

except Exception as e:
    print(f"✗ Architect FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Architect with workspace parameter
print("\n[Test 5] Architect with workspace parameter")
try:
    from agents.architect_agent import ArchitectAgent

    agent = ArchitectAgent(
        workspace=Path("/tmp/test_architect"),
        use_behaviors=True
    )

    assert hasattr(agent, '_behaviors'), "Missing _behaviors attribute"

    print("✓ Architect (with workspace) created successfully")
    print(f"  - Workspace: {agent.workspace}")

except Exception as e:
    print(f"✗ Architect (with workspace) FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ALL TESTS COMPLETED")
print("="*80)
