#!/usr/bin/env python3
"""
Integration test for Orchestrator agent.

Tests orchestrator's ability to:
1. Clarify ambiguous requirements with user
2. Delegate to multiple task executors
3. Iterate on existing workspace
4. Use jetbox notes for context continuity
"""
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_registry import AgentRegistry
from agent_config import config


class SimulatedUser:
    """Simulates user responses to orchestrator questions."""

    def __init__(self, responses: dict[str, str]):
        """
        Args:
            responses: Map of question keywords to user responses
        """
        self.responses = responses
        self.questions_asked = []

    def respond_to(self, question: str) -> str:
        """
        Find matching response based on question keywords.

        Args:
            question: Question from orchestrator

        Returns:
            User response
        """
        self.questions_asked.append(question)

        # Match keywords in question to predefined responses
        question_lower = question.lower()
        for keyword, response in self.responses.items():
            if keyword in question_lower:
                return response

        # Default response
        return "Use your best judgment"


# Test Scenarios
SCENARIOS = {
    "clarification_and_iteration": {
        "name": "Clarification + Iteration",
        "description": "Test clarification, delegation, and workspace iteration",
        "initial_request": "Create a calculator app",
        "simulated_responses": {
            "language": "Python",
            "features": "Just basic addition and multiplication",
            "tests": "Yes, use pytest",
        },
        "follow_up_request": "Add subtraction and division to the calculator",
        "success_criteria": [
            "orchestrator_asks_clarifying_questions",
            "delegates_to_task_executor",
            "creates_calculator_files",
            "uses_existing_workspace_for_iteration",
            "subtraction_and_division_added",
        ],
    },

    "multi_agent_collaboration": {
        "name": "Multi-Agent Collaboration",
        "description": "Multiple task executors working on related tasks",
        "initial_request": "Build a simple blog: create data models, then create a REST API for it",
        "simulated_responses": {
            "database": "Just use in-memory storage for now",
            "api framework": "Use Flask",
        },
        "success_criteria": [
            "delegates_multiple_tasks",
            "models_created_first",
            "api_uses_models",
            "jetbox_notes_shared_between_tasks",
        ],
    },

    "workspace_continuity": {
        "name": "Workspace Continuity",
        "description": "Jetbox notes provide context across iterations",
        "initial_request": "Create a Todo class with add/remove methods",
        "follow_up_request": "Add a mark_complete method to the Todo class",
        "follow_up_request_2": "Add tests for all Todo methods",
        "simulated_responses": {},
        "success_criteria": [
            "todo_class_created",
            "mark_complete_added_to_existing_class",
            "tests_cover_all_methods",
            "jetbox_notes_loaded_on_continuation",
        ],
    },
}


def run_scenario(scenario_config: dict) -> dict:
    """
    Run a single orchestrator test scenario.

    Args:
        scenario_config: Scenario configuration

    Returns:
        Test results dict
    """
    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario_config['name']}")
    print("=" * 70)
    print(f"Description: {scenario_config['description']}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Initialize agent registry
        registry = AgentRegistry(config_path="agents.yaml", workspace=workspace)
        orchestrator = registry.get_agent("orchestrator")

        # Initialize simulated user
        user = SimulatedUser(scenario_config.get("simulated_responses", {}))

        results = {
            "scenario": scenario_config["name"],
            "start_time": datetime.now().isoformat(),
            "success": False,
            "details": {},
            "errors": [],
        }

        try:
            # Phase 1: Initial request
            print(f"User: {scenario_config['initial_request']}")
            orchestrator.add_message({
                "role": "user",
                "content": scenario_config["initial_request"]
            })

            # Let orchestrator process (limited rounds for testing)
            max_turns = 10
            for turn in range(max_turns):
                print(f"\n--- Turn {turn + 1}/{max_turns} ---")

                # Get orchestrator response
                context = orchestrator.build_context()
                response = orchestrator._call_llm(context)

                orchestrator.add_message(response.get("message"))

                # Check for tool calls
                message = response.get("message", {})
                tool_calls = message.get("tool_calls", [])

                if not tool_calls:
                    # Orchestrator just responded with text
                    print(f"Orchestrator: {message.get('content', '')}")
                    break

                # Process tool calls
                all_complete = True
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    args = tool_call["function"]["arguments"]

                    print(f"Tool: {tool_name}({args})")

                    if tool_name == "clarify_with_user":
                        # Simulate user response
                        question = args.get("question", "")
                        user_response = user.respond_to(question)
                        print(f"User: {user_response}")

                        orchestrator.add_message({
                            "role": "user",
                            "content": user_response
                        })
                        all_complete = False

                    elif tool_name == "delegate_to_executor":
                        # Track delegation
                        task_desc = args.get("task_description", "")
                        workspace_mode = args.get("workspace_mode", "new")

                        results["details"]["delegated_task"] = task_desc
                        results["details"]["workspace_mode"] = workspace_mode

                        print(f"  → Delegated: {task_desc}")
                        print(f"  → Workspace mode: {workspace_mode}")

                        # For testing, simulate executor success
                        orchestrator.add_message({
                            "role": "tool",
                            "content": json.dumps({
                                "status": "success",
                                "message": "Task completed by executor"
                            })
                        })

                    elif tool_name == "create_task_plan":
                        tasks = args.get("tasks", [])
                        results["details"]["planned_tasks"] = tasks
                        print(f"  → Planned {len(tasks)} tasks")

                        orchestrator.add_message({
                            "role": "tool",
                            "content": json.dumps({"status": "plan_created"})
                        })

                if all_complete:
                    break

            # Phase 2: Follow-up request (if specified)
            if "follow_up_request" in scenario_config:
                print(f"\n\nUser: {scenario_config['follow_up_request']}")
                orchestrator.add_message({
                    "role": "user",
                    "content": scenario_config["follow_up_request"]
                })

                # Process follow-up (simplified for now)
                context = orchestrator.build_context()
                response = orchestrator._call_llm(context)

                message = response.get("message", {})
                tool_calls = message.get("tool_calls", [])

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    args = tool_call["function"]["arguments"]

                    if tool_name == "delegate_to_executor":
                        workspace_mode = args.get("workspace_mode", "new")
                        results["details"]["follow_up_workspace_mode"] = workspace_mode

                        if workspace_mode == "existing":
                            print("  ✓ Correctly used 'existing' workspace mode")
                        else:
                            print("  ✗ Should have used 'existing' workspace mode")

            # Evaluate success criteria
            success_count = 0
            criteria = scenario_config.get("success_criteria", [])

            for criterion in criteria:
                passed = evaluate_criterion(criterion, results, user)
                if passed:
                    success_count += 1
                    print(f"✓ {criterion}")
                else:
                    print(f"✗ {criterion}")

            results["success"] = (success_count >= len(criteria) * 0.7)  # 70% threshold
            results["success_rate"] = f"{success_count}/{len(criteria)}"

        except Exception as e:
            results["errors"].append(str(e))
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

        results["end_time"] = datetime.now().isoformat()
        return results


def evaluate_criterion(criterion: str, results: dict, user: SimulatedUser) -> bool:
    """
    Evaluate if a success criterion was met.

    Args:
        criterion: Criterion to evaluate
        results: Test results so far
        user: Simulated user

    Returns:
        True if criterion met
    """
    details = results.get("details", {})

    if criterion == "orchestrator_asks_clarifying_questions":
        return len(user.questions_asked) > 0

    elif criterion == "delegates_to_task_executor":
        return "delegated_task" in details

    elif criterion == "uses_existing_workspace_for_iteration":
        follow_up_mode = details.get("follow_up_workspace_mode")
        return follow_up_mode == "existing"

    elif criterion == "delegates_multiple_tasks":
        # Would need to track all delegations
        return details.get("planned_tasks") and len(details["planned_tasks"]) > 1

    # Default: can't evaluate, assume false
    return False


def main():
    """Run all orchestrator integration tests."""
    print("=" * 70)
    print("ORCHESTRATOR INTEGRATION TESTS")
    print("=" * 70)

    all_results = []

    for scenario_key, scenario_config in SCENARIOS.items():
        results = run_scenario(scenario_config)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in all_results if r["success"])
    total = len(all_results)

    for result in all_results:
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"{status} - {result['scenario']} ({result.get('success_rate', 'N/A')})")

    print(f"\nTotal: {passed}/{total} scenarios passed")

    # Save results
    output_file = Path("orchestrator_integration_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
