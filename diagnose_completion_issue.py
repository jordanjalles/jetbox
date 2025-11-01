"""
Diagnostic test to understand why agent doesn't call mark_goal_complete.

This test runs a simple goal and captures all LLM responses and tool calls
to see what the agent is doing in each round.
"""
import sys
from pathlib import Path
from task_executor_agent import TaskExecutorAgent

# Monkey-patch to capture tool calls
original_dispatch = None
tool_calls_history = []

def capture_dispatch(self, tool_call):
    """Capture all tool calls for analysis."""
    tool_name = tool_call["function"]["name"]
    args = tool_call["function"]["arguments"]

    tool_calls_history.append({
        "round": len(tool_calls_history) + 1,
        "tool": tool_name,
        "args": args
    })

    print(f"\n[CAPTURE] Round {len(tool_calls_history)}: {tool_name}({list(args.keys())})")

    # Call original
    return original_dispatch(tool_call)

# Run test
print("="*70)
print("DIAGNOSTIC TEST: Why doesn't agent call mark_goal_complete?")
print("="*70)

workspace = Path(".agent_workspace/diagnostic_test")
workspace.mkdir(parents=True, exist_ok=True)

# Clean workspace
for f in workspace.glob("*"):
    if f.is_file():
        f.unlink()

agent = TaskExecutorAgent(
    workspace=workspace,
    goal="Create a file hello.py that prints 'Hello World'",
    use_behaviors=True,
    max_rounds=5
)

print(f"\nGoal: {agent.context_manager.state.goal.description}")
print(f"Max rounds: 5")
print(f"Tools available: {len(agent.get_tools())}")
print()

# Monkey-patch dispatch
original_dispatch = agent.dispatch_tool
agent.dispatch_tool = lambda tc: capture_dispatch(agent, tc)

# Run
print("Running agent...\n")
result = agent.run()

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Status: {result.get('status')}")
print(f"Reason: {result.get('reason')}")

# Check files
files = list(workspace.glob("*.py"))
print(f"\nFiles created: {len(files)}")
for f in files:
    print(f"  - {f.name}: {f.read_text()[:50]}...")

print(f"\n" + "="*70)
print("TOOL CALLS HISTORY")
print("="*70)
for call in tool_calls_history:
    print(f"Round {call['round']}: {call['tool']} - {call['args']}")

print(f"\n" + "="*70)
print("ANALYSIS")
print("="*70)

# Count tool calls
mark_complete_calls = [c for c in tool_calls_history if c['tool'] == 'mark_goal_complete']
write_file_calls = [c for c in tool_calls_history if c['tool'] == 'write_file']

print(f"Total rounds: {len(tool_calls_history)}")
print(f"write_file calls: {len(write_file_calls)}")
print(f"mark_goal_complete calls: {len(mark_complete_calls)}")

if len(write_file_calls) > 0 and len(mark_complete_calls) == 0:
    print("\n⚠️  ISSUE FOUND: Agent wrote file but never called mark_goal_complete!")
    print("   This explains why tests are failing despite files being created.")

    if len(write_file_calls) > 1:
        print(f"   Agent wrote file {len(write_file_calls)} times (redundant writes)")

    print("\n   Possible causes:")
    print("   1. LLM not recognizing task is complete")
    print("   2. Completion detector not triggering nudge")
    print("   3. Agent ignoring completion nudge")
    print("   4. Agent stuck in verification loop")
