"""Integration guide for hierarchical context manager with agent.py.

This shows how to replace the current flat context approach with
hierarchical, crash-resilient context management.
"""

from __future__ import annotations

from typing import Any

from context_manager import ContextManager, Subtask, Task


# ----------------------------
# Integration example
# ----------------------------
class EnhancedAgent:
    """Agent with hierarchical context management."""

    def __init__(self, goal: str) -> None:
        self.ctx = ContextManager()
        self.ctx.load_or_init(goal)
        self._initialize_tasks_from_goal(goal)

    def _initialize_tasks_from_goal(self, goal: str) -> None:
        """
        Parse goal and create initial task hierarchy.

        This can be done via LLM or heuristics.
        """
        # Example: if loading existing state, tasks already exist
        if self.ctx.state.goal and self.ctx.state.goal.tasks:
            return

        # For new goals, create initial tasks
        # This is a simple example - in practice, use LLM to decompose
        if "mathx" in goal.lower():
            tasks = [
                Task(
                    description="Create mathx package structure",
                    subtasks=[
                        Subtask(description="Create mathx/__init__.py"),
                        Subtask(description="Implement add(a, b) function"),
                    ],
                ),
                Task(
                    description="Add tests",
                    subtasks=[
                        Subtask(description="Create tests/test_mathx.py"),
                        Subtask(description="Write test cases for add()"),
                    ],
                ),
                Task(
                    description="Verify quality",
                    subtasks=[
                        Subtask(description="Run ruff check"),
                        Subtask(description="Run pytest"),
                    ],
                ),
            ]

            for task in tasks:
                task.parent_goal = goal
                self.ctx.state.goal.tasks.append(task)

            # Mark first task as active
            if tasks:
                tasks[0].status = "in_progress"
                tasks[0].subtasks[0].status = "in_progress"

            self.ctx._save_state()

    def get_llm_context(self) -> str:
        """
        Get compact context to send to LLM.

        This replaces the old approach of sending full message history.
        """
        # Get hierarchical summary
        context = self.ctx.get_compact_context(max_chars=1500)

        # Add system guidance based on context state
        guidance = self._get_contextual_guidance()

        return f"{context}\n\n{guidance}"

    def _get_contextual_guidance(self) -> str:
        """Generate guidance based on current context state."""
        lines = ["GUIDANCE:"]

        # Check for loops
        if self.ctx.state.blocked_actions:
            lines.append(
                "⚠ Some actions are blocked due to loops. Try different approaches."
            )
            lines.append(self.ctx.get_loop_summary())

        # Check for repeated failures
        task = self.ctx._get_current_task()
        if task:
            subtask = task.active_subtask()
            if subtask and subtask.attempt_count > 1:
                lines.append(
                    f"⚠ Current subtask attempted {subtask.attempt_count} times."
                )
                if subtask.failure_reason:
                    lines.append(f"  Last failure: {subtask.failure_reason}")
                lines.append("  Consider alternative approach.")

        return "\n".join(lines)

    def execute_tool_call(
        self, name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute tool call with loop detection.

        Returns result dict. If loop detected, returns error.
        """
        # Check if action is allowed (not in loop)
        if not self.ctx.record_action(name, args, result="pending"):
            return {
                "error": "Action blocked: loop detected",
                "suggestion": "This action has been tried multiple times. "
                "Try a different approach or escalate to higher-level task.",
            }

        # Execute the actual tool (import from agent.py)
        try:
            from agent import TOOLS

            tool_fn = TOOLS.get(name)
            if not tool_fn:
                result = {"error": f"Unknown tool: {name}"}
                self.ctx.action_history[-1].result = "error"
                self.ctx.action_history[-1].error_msg = result["error"]
            else:
                result = {"result": tool_fn(**args)}
                self.ctx.action_history[-1].result = "success"

        except Exception as e:
            result = {"error": str(e)}
            self.ctx.action_history[-1].result = "error"
            self.ctx.action_history[-1].error_msg = str(e)

        self.ctx._save_state()
        return result

    def probe_and_plan(self, probe_state: dict[str, Any]) -> list[str]:
        """
        Update probe state and get next actions.

        This replaces the old plan_next() approach.
        """
        # Update probe state
        self.ctx.update_probe_state(probe_state)

        # Check if goal is complete
        if probe_state.get("pytest_ok") and probe_state.get("ruff_ok"):
            return ["DONE"]

        # Get current subtask
        task = self.ctx._get_current_task()
        if not task:
            return ["ERROR: No active task"]

        subtask = task.active_subtask()
        if not subtask:
            # Try to advance to next subtask
            if self.ctx.advance_to_next_subtask():
                subtask = task.active_subtask()
            else:
                # Task complete, move to next task
                self.ctx.state.current_task_idx += 1
                self.ctx._save_state()
                return self.probe_and_plan(probe_state)

        # Generate concrete actions for current subtask
        actions = self._subtask_to_actions(subtask, probe_state)
        return actions

    def _subtask_to_actions(
        self, subtask: Subtask, probe_state: dict[str, Any]
    ) -> list[str]:
        """
        Convert subtask to concrete actions.

        This can be done via LLM or heuristics.
        """
        desc = subtask.description.lower()

        # Simple heuristic mapping
        if "create" in desc and "__init__.py" in desc:
            return ["write_file mathx/__init__.py with add function"]
        elif "create" in desc and "test_mathx.py" in desc:
            return ["write_file tests/test_mathx.py with test functions"]
        elif "ruff" in desc:
            return ["run_cmd ['ruff', 'check', '.']"]
        elif "pytest" in desc:
            return ["run_cmd ['pytest', '-q']"]
        else:
            return [f"Complete: {subtask.description}"]

    def mark_subtask_done(self, success: bool, reason: str = "") -> None:
        """Mark current subtask as complete or failed."""
        self.ctx.mark_subtask_complete(success, reason)

        if success:
            # Move to next subtask
            self.ctx.advance_to_next_subtask()


# ----------------------------
# Migration guide from old agent.py
# ----------------------------
def migration_example() -> None:
    """
    Show how to migrate from old agent.py approach.

    OLD APPROACH (agent.py):
    ------------------------
    - Flat message list with manual pruning
    - Simple deduplication counter (SEEN dict)
    - Status in status.txt, ledger in agent_ledger.log
    - Context = system prompt + last N messages

    NEW APPROACH (with ContextManager):
    -----------------------------------
    - Hierarchical: Goal → Task → Subtask → Action
    - Sophisticated loop detection (patterns, alternating, etc.)
    - Structured state in .agent_context/state.json
    - Context = compact hierarchical summary (need-to-know only)

    MIGRATION STEPS:
    ----------------
    1. Replace global SEEN dict with ctx.record_action()
    2. Replace _ledger_summary() with ctx.get_compact_context()
    3. Replace manual message pruning with hierarchical context
    4. Add task decomposition step at goal start
    5. Replace plan_next() with ctx.probe_and_plan()
    """

    # OLD WAY:
    # messages = [system, user, ...past N messages...]
    # summary = _ledger_summary()
    # messages.append({"role": "assistant", "content": summary})

    # NEW WAY:
    agent = EnhancedAgent(goal="Create mathx package")
    context = agent.get_llm_context()  # Hierarchical, compact

    print("=== OLD: Flat message history ===")
    print("messages = [")
    print('  {"role": "system", "content": "..."},')
    print('  {"role": "user", "content": "Create mathx..."},')
    print("  ... 12 most recent messages ...")
    print("]")

    print("\n=== NEW: Hierarchical context ===")
    print(context)

    print("\n=== Key differences ===")
    print("1. OLD: Linear message history")
    print("   NEW: Tree structure (Goal → Task → Subtask)")
    print("\n2. OLD: Shows all recent actions")
    print("   NEW: Shows only current branch of tree")
    print("\n3. OLD: Simple repeat counter (>3 = skip)")
    print("   NEW: Pattern detection (A-B-A-B = loop)")
    print("\n4. OLD: Crash = start from ledger summary")
    print("   NEW: Crash = restore full hierarchy from state.json")


# ----------------------------
# Example: Full agent loop with new context
# ----------------------------
def example_agent_loop() -> None:
    """Example of complete agent loop with hierarchical context."""
    goal = "Create mathx package with add function and tests"

    # Initialize
    agent = EnhancedAgent(goal)

    # Simulate agent rounds
    for round_no in range(1, 10):
        print(f"\n{'='*60}")
        print(f"ROUND {round_no}")
        print('='*60)

        # 1. Probe current state
        from agent import probe_state

        probe = probe_state()

        # 2. Get plan based on current hierarchy
        plan = agent.probe_and_plan(probe)

        if plan == ["DONE"]:
            print("Goal complete!")
            break

        print(f"\nPlan: {plan}")

        # 3. Get compact context for LLM
        llm_context = agent.get_llm_context()
        print(f"\nLLM Context:\n{llm_context}")

        # 4. (In real agent: call LLM with context + plan)
        # 5. (LLM returns tool calls)

        # 6. Execute tool calls with loop detection
        # Example: simulate a tool call
        if round_no == 1:
            result = agent.execute_tool_call(
                "write_file",
                {"path": "mathx/__init__.py", "content": "def add(a, b):\n    return a+b"},
            )
            print(f"\nTool result: {result}")

            if "error" not in result:
                agent.mark_subtask_done(success=True)

        # Context automatically saved at each step


if __name__ == "__main__":
    print("CONTEXT MANAGER INTEGRATION GUIDE")
    print("=" * 60)

    print("\n1. Migration Example")
    print("-" * 60)
    migration_example()

    print("\n\n2. Example Agent Loop")
    print("-" * 60)
    # Uncomment to run full example:
    # example_agent_loop()

    print("\n\n3. Key Benefits")
    print("-" * 60)
    print("✓ Hierarchical structure matches how humans think")
    print("✓ Automatic loop detection prevents infinite retries")
    print("✓ Compact context = less token usage")
    print("✓ Crash recovery = resume exactly where you left off")
    print("✓ Need-to-know = LLM only sees relevant info for current step")
