#!/usr/bin/env python3
"""
Debug script to investigate empty rounds where LLM doesn't call tools.

This script patches the base agent to log full LLM responses and context.
"""
from pathlib import Path
import json
from orchestrator_agent import OrchestratorAgent

# Patch base_agent to add debug logging
import base_agent

original_run = base_agent.BaseAgent.run

def debug_run(self, max_rounds: int | None = None) -> dict:
    """Patched run method with debug logging."""
    # Get max rounds from config or parameter
    if max_rounds is None:
        max_rounds = getattr(self.config.rounds, 'max_per_subtask', 128) if self.config else 128

    # Get model and temperature from config or defaults
    model = getattr(self, 'model', None) or getattr(self.config.llm, 'model', 'gpt-oss:20b') if self.config else 'gpt-oss:20b'
    temperature = getattr(self, 'temperature', None) or getattr(self.config.llm, 'temperature', 0.2) if self.config else 0.2

    # Trigger onGoalStart event
    if self.context_manager and self.context_manager.state.goal:
        goal = self.context_manager.state.goal.description
        self.trigger_behavior_event("onGoalStart", goal=goal)

    print(f"[{self.name}] Starting run loop (max_rounds={max_rounds}, model={model})")

    consecutive_empty_rounds = 0

    try:
        for round_no in range(1, max_rounds + 1):
            # Trigger onRoundStart
            self.trigger_behavior_event("onRoundStart", round_number=round_no)

            # Build context and call LLM
            print(f"\n[{self.name}] Round {round_no}/{max_rounds}")

            # DEBUG: Log context summary
            context = self.build_context()
            print(f"[DEBUG] Context has {len(context)} messages, ~{sum(len(str(m.get('content', ''))) for m in context)//4} tokens")

            response = self.call_llm(model=model, temperature=temperature)

            # Check for circuit breaker (consecutive timeouts)
            if response.get("_circuit_breaker"):
                print(f"[{self.name}] Circuit breaker triggered - saving partial progress")
                partial_result = self._save_partial_progress()
                return partial_result

            # Check for timeout (will retry)
            if response.get("_timeout"):
                print(f"[{self.name}] LLM timeout - retrying next round")
                continue

            # Add assistant message to history
            if "message" in response:
                msg = response["message"]

                # DEBUG: Log message content
                msg_content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                if not tool_calls:
                    consecutive_empty_rounds += 1
                    print(f"[DEBUG] ⚠️  Empty round #{consecutive_empty_rounds} - no tool calls!")
                    print(f"[DEBUG] Message content: {msg_content[:300]}...")

                    if consecutive_empty_rounds >= 3:
                        print("[DEBUG] 🚨 3 consecutive empty rounds! Dumping full context...")
                        with open(f"debug_empty_context_{self.name}_round{round_no}.json", "w") as f:
                            json.dump({
                                "round": round_no,
                                "context": context,
                                "response": msg,
                            }, f, indent=2)
                        print(f"[DEBUG] Context saved to debug_empty_context_{self.name}_round{round_no}.json")
                else:
                    consecutive_empty_rounds = 0

                self.add_message(msg)

                # Execute tool calls
                if tool_calls:
                    print(f"[{self.name}] Executing {len(tool_calls)} tool call(s)")

                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        print(f"[{self.name}] -> {tool_name}")

                        # Dispatch tool
                        result = self.dispatch_tool(tool_call)

                        # Add tool result to messages
                        tool_result_str = json.dumps(result)
                        tool_message = {
                            "role": "tool",
                            "content": tool_result_str,
                        }
                        self.add_message(tool_message)

                        # Check for goal completion
                        if isinstance(result, dict):
                            # Check for mark_complete/mark_failed
                            if result.get("success") is True and "summary" in result:
                                print(f"[{self.name}] Goal marked complete")
                                self.trigger_behavior_event("onGoalComplete", success=True, result=result)
                                return {
                                    "status": "success",
                                    "summary": result.get("summary"),
                                    "workspace": str(self.workspace) if self.workspace else None,
                                }
                            elif result.get("success") is False and "reason" in result:
                                print(f"[{self.name}] Goal marked failed")
                                self.trigger_behavior_event("onGoalComplete", success=False, result=result)
                                return {
                                    "status": "failure",
                                    "reason": result.get("reason"),
                                    "workspace": str(self.workspace) if self.workspace else None,
                                }

                            # Check for goal_complete status (legacy tools)
                            actual_result = result.get("result", result)
                            if isinstance(actual_result, dict) and actual_result.get("status") == "goal_complete":
                                print(f"[{self.name}] Goal completed (legacy signal)")
                                self.trigger_behavior_event("onGoalComplete", success=True, result=actual_result)
                                return {
                                    "status": "success",
                                    "message": actual_result.get("message", "Goal completed"),
                                    "workspace": str(self.workspace) if self.workspace else None,
                                }

            # Trigger onRoundEnd
            self.trigger_behavior_event("onRoundEnd", round_number=round_no)

            # Increment round counter
            self.increment_round()

        # Max rounds reached
        print(f"[{self.name}] Max rounds ({max_rounds}) reached without completion")
        return {
            "status": "failure",
            "message": f"Max rounds ({max_rounds}) reached without completion",
            "workspace": str(self.workspace) if self.workspace else None,
        }

    except Exception as e:
        import traceback
        print(f"[{self.name}] Error in run loop: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "workspace": str(self.workspace) if self.workspace else None,
        }

# Apply patch
base_agent.BaseAgent.run = debug_run

# Test with simple delegation
if __name__ == "__main__":
    print("=" * 80)
    print("DEBUG: Testing delegation with empty round logging")
    print("=" * 80)

    workspace = Path(".agent_workspaces/debug-empty-rounds")

    orchestrator = OrchestratorAgent(workspace=workspace)
    orchestrator.add_message({
        "role": "user",
        "content": "Create a Flask REST API with CRUD endpoints for a User model. Use in-memory storage. Include tests."
    })

    result = orchestrator.run(max_rounds=10)

    print("\n" + "=" * 80)
    print("RESULT:", result)
    print("=" * 80)
