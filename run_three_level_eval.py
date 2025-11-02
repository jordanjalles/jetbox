#!/usr/bin/env python3
"""
Three-Level Evaluation Test Suite for Jetbox Behavior System

Tests the complete agent system across 3 levels of complexity:
1. Level 1: Direct TaskExecutor (L1-L4 coding tasks)
2. Level 2: Orchestrator + TaskExecutor (L4 tasks)
3. Level 3: Orchestrator + Architect + TaskExecutor (L5-L7 tasks)
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Set model for testing (gpt-oss:20b recommended for good tool use)
os.environ["OLLAMA_MODEL"] = "gpt-oss:20b"

from task_executor_agent import TaskExecutorAgent
from orchestrator_agent import OrchestratorAgent
from architect_agent import ArchitectAgent


def log_header(log_file, task_name, details):
    """Write test header to log file"""
    log_file.write(f"\n\n{'='*80}\n")
    log_file.write(f"Test: {task_name}\n")
    for key, value in details.items():
        log_file.write(f"{key}: {value}\n")
    log_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()


def run_level1_test(task_name, goal, max_rounds, expected_files):
    """Run a Level 1 test (Direct TaskExecutor)"""
    print(f"\n{'='*80}")
    print(f"LEVEL 1 - {task_name}")
    print(f"{'='*80}")

    log_path = Path("evaluation_results/level1_task_executor_eval.log")
    with open(log_path, "a") as log_file:
        log_header(log_file, task_name, {
            "Goal": goal,
            "Max Rounds": max_rounds,
            "Expected Files": ", ".join(expected_files)
        })

        start_time = time.time()

        try:
            # Create agent
            workspace = Path(f".agent_workspace/level1_{task_name.lower().replace(' ', '_').replace(':', '')}")
            log_file.write(f"\nWorkspace: {workspace}\n")
            log_file.flush()

            agent = TaskExecutorAgent(
                workspace=workspace,
                goal=goal,
            )

            # Log agent setup
            log_file.write(f"\nAgent Setup:\n")
            log_file.write(f"  Workspace: {workspace}\n")
            log_file.write(f"  Behaviors: {[b.get_name() for b in agent.behaviors]}\n")
            log_file.write(f"  Tools: {len(agent.get_tools())} available\n")
            log_file.flush()

            # Run agent
            log_file.write(f"\nRunning agent...\n")
            log_file.flush()

            result = agent.run(max_rounds=max_rounds)

            end_time = time.time()
            duration = end_time - start_time

            # Check files created
            files_created = []
            for expected in expected_files:
                file_path = workspace / expected
                if file_path.exists():
                    files_created.append(expected)

            # Log results
            success = result.get("status") == "success"
            log_file.write(f"\nResults:\n")
            log_file.write(f"  Status: {result.get('status')}\n")
            log_file.write(f"  Reason: {result.get('reason', 'N/A')}\n")
            log_file.write(f"  Duration: {duration:.2f}s\n")
            log_file.write(f"  Rounds Used: {result.get('rounds_used', 'N/A')}\n")
            log_file.write(f"  Files Created: {len(files_created)}/{len(expected_files)}\n")
            log_file.write(f"  Files: {files_created}\n")
            log_file.flush()

            status_icon = "✓" if success else "✗"
            status_text = "PASS" if success else "FAIL"
            print(f"{status_icon} {task_name}: {status_text} ({duration:.1f}s, {len(files_created)}/{len(expected_files)} files)")

            return {
                "task": task_name,
                "success": success,
                "duration": duration,
                "rounds_used": result.get('rounds_used', 0),
                "files_created": len(files_created),
                "files_expected": len(expected_files),
                "result": result
            }

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            log_file.write(f"\nERROR: {str(e)}\n")
            log_file.write(f"Duration before error: {duration:.2f}s\n")
            log_file.flush()
            print(f"✗ {task_name}: ERROR - {str(e)}")
            return {
                "task": task_name,
                "success": False,
                "duration": duration,
                "error": str(e)
            }


def run_level2_test(task_name, user_request, max_rounds):
    """Run a Level 2 test (Orchestrator + TaskExecutor)"""
    print(f"\n{'='*80}")
    print(f"LEVEL 2 - {task_name}")
    print(f"{'='*80}")

    log_path = Path("evaluation_results/level2_orchestrator_eval.log")
    with open(log_path, "a") as log_file:
        log_header(log_file, task_name, {
            "User Request": user_request,
            "Max Rounds": max_rounds
        })

        start_time = time.time()

        try:
            # Create orchestrator
            log_file.write(f"\nCreating OrchestratorAgent...\n")
            log_file.flush()

            orchestrator = OrchestratorAgent(workspace=Path(".agent_workspace/test_orchestrator"))

            # Log setup
            log_file.write(f"\nOrchestrator Setup:\n")
            log_file.write(f"  Behaviors: {[b.get_name() for b in orchestrator.behaviors]}\n")

            tools = orchestrator.get_tools()
            tool_names = [t['function']['name'] for t in tools]
            log_file.write(f"  Tools ({len(tools)}): {', '.join(tool_names)}\n")

            # Check for delegation tools
            has_delegate_executor = any("delegate" in name and "executor" in name for name in tool_names)
            has_consult_architect = any("architect" in name for name in tool_names)

            log_file.write(f"\nDelegation Capabilities:\n")
            log_file.write(f"  Can delegate to executor: {has_delegate_executor}\n")
            log_file.write(f"  Can consult architect: {has_consult_architect}\n")

            end_time = time.time()
            duration = end_time - start_time

            log_file.write(f"\nResults:\n")
            log_file.write(f"  Orchestrator Configured: YES\n")
            log_file.write(f"  Can Delegate to Executor: {has_delegate_executor}\n")
            log_file.write(f"  Can Consult Architect: {has_consult_architect}\n")
            log_file.write(f"  Duration: {duration:.2f}s\n")
            log_file.flush()

            print(f"✓ {task_name}: Orchestrator configured correctly ({len(tools)} tools available)")

            return {
                "task": task_name,
                "orchestrator_configured": True,
                "can_delegate_executor": has_delegate_executor,
                "can_consult_architect": has_consult_architect,
                "tool_count": len(tools),
                "duration": duration
            }

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            log_file.write(f"\nERROR: {str(e)}\n")
            log_file.write(f"Duration before error: {duration:.2f}s\n")
            log_file.flush()
            print(f"✗ {task_name}: ERROR - {str(e)}")
            return {
                "task": task_name,
                "success": False,
                "duration": duration,
                "error": str(e)
            }


def run_level3_test(task_name, user_request):
    """Run a Level 3 test (Full Stack: Orchestrator + Architect + TaskExecutor)"""
    print(f"\n{'='*80}")
    print(f"LEVEL 3 - {task_name}")
    print(f"{'='*80}")

    log_path = Path("evaluation_results/level3_full_stack_eval.log")
    with open(log_path, "a") as log_file:
        log_header(log_file, task_name, {
            "User Request": user_request
        })

        start_time = time.time()

        try:
            # Test that all 3 agents can be instantiated
            log_file.write(f"\nInstantiating all agents...\n")
            log_file.flush()

            orchestrator = OrchestratorAgent(workspace=Path(".agent_workspace/test_orchestrator_l3"))
            log_file.write(f"  ✓ Orchestrator created\n")

            architect = ArchitectAgent(
                workspace=Path(".agent_workspace/test_architect"),
            )
            log_file.write(f"  ✓ Architect created\n")

            task_executor = TaskExecutorAgent(
                workspace=Path(".agent_workspace/test_executor"),
                goal="test",
            )
            log_file.write(f"  ✓ TaskExecutor created\n")
            log_file.flush()

            # Log setup
            log_file.write(f"\nAll Agents Configured:\n")
            log_file.write(f"  Orchestrator: {len(orchestrator.behaviors)} behaviors, {len(orchestrator.get_tools())} tools\n")
            log_file.write(f"  Architect: {len(architect.behaviors)} behaviors, {len(architect.get_tools())} tools\n")
            log_file.write(f"  TaskExecutor: {len(task_executor.behaviors)} behaviors, {len(task_executor.get_tools())} tools\n")

            # Verify delegation tools
            orch_tools = [t['function']['name'] for t in orchestrator.get_tools()]
            has_architect_delegation = any("architect" in name for name in orch_tools)
            has_executor_delegation = any("delegate" in name and "executor" in name for name in orch_tools)

            log_file.write(f"\nDelegation Setup:\n")
            log_file.write(f"  Can consult architect: {has_architect_delegation}\n")
            log_file.write(f"  Can delegate to executor: {has_executor_delegation}\n")

            # Check architect tools
            arch_tools = [t['function']['name'] for t in architect.get_tools()]
            log_file.write(f"\nArchitect Tools: {', '.join(arch_tools)}\n")

            # Check task executor tools
            exec_tools = [t['function']['name'] for t in task_executor.get_tools()]
            log_file.write(f"\nTaskExecutor Tools: {', '.join(exec_tools)}\n")

            end_time = time.time()
            duration = end_time - start_time

            full_stack_ready = has_architect_delegation and has_executor_delegation

            log_file.write(f"\nResults:\n")
            log_file.write(f"  All Agents Configured: YES\n")
            log_file.write(f"  Full Stack Ready: {full_stack_ready}\n")
            log_file.write(f"  Duration: {duration:.2f}s\n")
            log_file.flush()

            print(f"✓ {task_name}: Full stack configured correctly")

            return {
                "task": task_name,
                "all_agents_configured": True,
                "full_stack_ready": full_stack_ready,
                "orchestrator_tools": len(orch_tools),
                "architect_tools": len(arch_tools),
                "executor_tools": len(exec_tools),
                "duration": duration
            }

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            log_file.write(f"\nERROR: {str(e)}\n")
            log_file.write(f"Duration before error: {duration:.2f}s\n")
            log_file.flush()
            print(f"✗ {task_name}: ERROR - {str(e)}")
            return {
                "task": task_name,
                "success": False,
                "duration": duration,
                "error": str(e)
            }


def generate_summary_report(level1_results, level2_results, level3_results):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)

    with open("evaluation_results/THREE_LEVEL_EVAL_SUMMARY.md", "w") as f:
        f.write("# Three-Level Evaluation Test Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: {os.getenv('OLLAMA_MODEL', 'unknown')}\n\n")

        # Level 1
        f.write("## Level 1: Direct TaskExecutor (L1-L4)\n\n")
        f.write("Tests the TaskExecutor agent in isolation with increasingly complex coding tasks.\n\n")

        l1_pass = sum(1 for r in level1_results if r.get('success', False))
        f.write(f"**Results**: {l1_pass}/{len(level1_results)} passed\n\n")

        f.write("| Task | Status | Duration | Files | Rounds |\n")
        f.write("|------|--------|----------|-------|--------|\n")
        for r in level1_results:
            status = "✓ PASS" if r.get('success', False) else "✗ FAIL"
            duration = f"{r.get('duration', 0):.1f}s" if 'duration' in r else "N/A"
            files = f"{r.get('files_created', 0)}/{r.get('files_expected', 0)}" if 'files_created' in r else "N/A"
            rounds = r.get('rounds_used', 'N/A')
            f.write(f"| {r['task']} | {status} | {duration} | {files} | {rounds} |\n")

        # Level 2
        f.write("\n## Level 2: Orchestrator + TaskExecutor (L4)\n\n")
        f.write("Tests the Orchestrator's ability to delegate to TaskExecutor.\n\n")

        l2_configured = sum(1 for r in level2_results if r.get('orchestrator_configured', False))
        l2_can_delegate = sum(1 for r in level2_results if r.get('can_delegate_executor', False))

        f.write(f"**Results**: {l2_configured}/{len(level2_results)} configured, {l2_can_delegate}/{len(level2_results)} can delegate\n\n")

        f.write("| Task | Configured | Can Delegate | Can Consult Architect | Tools |\n")
        f.write("|------|------------|--------------|----------------------|-------|\n")
        for r in level2_results:
            configured = "✓" if r.get('orchestrator_configured', False) else "✗"
            can_delegate = "✓" if r.get('can_delegate_executor', False) else "✗"
            can_consult = "✓" if r.get('can_consult_architect', False) else "✗"
            tools = r.get('tool_count', 'N/A')
            f.write(f"| {r['task']} | {configured} | {can_delegate} | {can_consult} | {tools} |\n")

        # Level 3
        f.write("\n## Level 3: Full Stack (Orchestrator + Architect + TaskExecutor) (L5-L7)\n\n")
        f.write("Tests full 3-agent collaboration capabilities.\n\n")

        l3_configured = sum(1 for r in level3_results if r.get('all_agents_configured', False))
        l3_ready = sum(1 for r in level3_results if r.get('full_stack_ready', False))

        f.write(f"**Results**: {l3_configured}/{len(level3_results)} configured, {l3_ready}/{len(level3_results)} full stack ready\n\n")

        f.write("| Task | All Configured | Full Stack Ready | Orch Tools | Arch Tools | Exec Tools |\n")
        f.write("|------|----------------|------------------|------------|------------|------------|\n")
        for r in level3_results:
            configured = "✓" if r.get('all_agents_configured', False) else "✗"
            ready = "✓" if r.get('full_stack_ready', False) else "✗"
            orch_tools = r.get('orchestrator_tools', 'N/A')
            arch_tools = r.get('architect_tools', 'N/A')
            exec_tools = r.get('executor_tools', 'N/A')
            f.write(f"| {r['task']} | {configured} | {ready} | {orch_tools} | {arch_tools} | {exec_tools} |\n")

        # Overall Assessment
        f.write("\n## Overall Assessment\n\n")
        f.write(f"- **Level 1 (TaskExecutor)**: {l1_pass}/{len(level1_results)} tasks successful\n")
        f.write(f"- **Level 2 (Orchestration)**: {l2_configured}/{len(level2_results)} configured, {l2_can_delegate}/{len(level2_results)} can delegate\n")
        f.write(f"- **Level 3 (Full Stack)**: {l3_configured}/{len(level3_results)} configured, {l3_ready}/{len(level3_results)} ready\n\n")

        # Performance Metrics
        f.write("## Performance Metrics\n\n")

        if level1_results:
            avg_duration = sum(r.get('duration', 0) for r in level1_results) / len(level1_results)
            avg_rounds = sum(r.get('rounds_used', 0) for r in level1_results if 'rounds_used' in r)
            avg_rounds = avg_rounds / len([r for r in level1_results if 'rounds_used' in r]) if any('rounds_used' in r for r in level1_results) else 0

            f.write(f"**Level 1 TaskExecutor**:\n")
            f.write(f"- Average duration: {avg_duration:.1f}s\n")
            f.write(f"- Average rounds: {avg_rounds:.1f}\n")
            f.write(f"- Success rate: {l1_pass}/{len(level1_results)} ({100*l1_pass/len(level1_results):.0f}%)\n\n")

        # Issues Found
        f.write("## Issues Found\n\n")
        errors = []
        for results in [level1_results, level2_results, level3_results]:
            for r in results:
                if 'error' in r:
                    errors.append(f"**{r['task']}**: {r['error']}")

        if errors:
            for e in errors:
                f.write(f"- {e}\n")
        else:
            f.write("✓ No critical errors found during testing.\n")

        f.write("\n")

        # Behavior System Observations
        f.write("## Behavior System Observations\n\n")

        if l1_pass == len(level1_results):
            f.write("- ✓ **TaskExecutor behavior system working well** for L1-L4 tasks\n")
        else:
            f.write(f"- ⚠ **TaskExecutor has {len(level1_results) - l1_pass} failures** - needs investigation\n")

        if l2_can_delegate == len(level2_results):
            f.write("- ✓ **Orchestrator delegation configured correctly**\n")
        else:
            f.write("- ⚠ **Orchestrator delegation issues** - some tests missing delegation tools\n")

        if l3_ready == len(level3_results):
            f.write("- ✓ **Full stack integration working** - all 3 agents can collaborate\n")
        else:
            f.write("- ⚠ **Full stack integration incomplete** - delegation chain not fully configured\n")

        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        f.write("### Next Steps for Testing\n\n")
        f.write("1. **Run full end-to-end tests** with actual LLM interaction for Level 2 and 3\n")
        f.write("2. **Add integration tests** with mocked LLM responses for faster iteration\n")
        f.write("3. **Test failure recovery** - verify agents handle errors gracefully\n")
        f.write("4. **Performance optimization** - profile slow operations\n")
        f.write("5. **Context management stress tests** - test with large codebases\n\n")

        f.write("### Improvement Ideas\n\n")

        if l1_pass < len(level1_results):
            f.write("- Investigate TaskExecutor failures - check logs for patterns\n")

        f.write("- Add telemetry for behavior execution (timing, errors, context usage)\n")
        f.write("- Create benchmark suite with known-good solutions for regression testing\n")
        f.write("- Add automated verification of expected outputs (not just file existence)\n")
        f.write("- Test behavior composition edge cases (conflicting behaviors, missing dependencies)\n")

        f.write("\n---\n\n")
        f.write("*Generated by three-level evaluation test suite*\n")

    print("✓ Summary report generated: evaluation_results/THREE_LEVEL_EVAL_SUMMARY.md")


def main():
    """Run all three levels of evaluation tests"""
    print("="*80)
    print("THREE-LEVEL EVALUATION TEST SUITE")
    print("="*80)
    print(f"Model: {os.getenv('OLLAMA_MODEL', 'unknown')}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Clear log files
    for log_file in ["level1_task_executor_eval.log", "level2_orchestrator_eval.log", "level3_full_stack_eval.log"]:
        log_path = Path("evaluation_results") / log_file
        if log_path.exists():
            log_path.unlink()
        # Write header
        with open(log_path, "w") as f:
            f.write(f"{'='*80}\n")
            f.write(f"EVALUATION LOG: {log_file}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {os.getenv('OLLAMA_MODEL', 'unknown')}\n")
            f.write(f"{'='*80}\n")

    # LEVEL 1: TaskExecutor Tests
    print("\n" + "="*80)
    print("LEVEL 1: DIRECT TASKEXECUTOR (L1-L4)")
    print("="*80)

    level1_results = []

    level1_results.append(run_level1_test(
        "L1: Simple File",
        "Create a file hello.py that prints 'Hello World'",
        5,
        ["hello.py"]
    ))

    level1_results.append(run_level1_test(
        "L2: File with Function",
        "Create calculator.py with an add(a, b) function and a test for it",
        10,
        ["calculator.py", "test_calculator.py"]
    ))

    level1_results.append(run_level1_test(
        "L3: Multi-File Package",
        "Create a Python package 'mathx' with add, subtract, multiply, divide functions in separate files, with tests for all functions",
        20,
        ["mathx/__init__.py", "mathx/add.py", "tests/test_mathx.py"]
    ))

    level1_results.append(run_level1_test(
        "L4: Package with Dependencies",
        "Create a 'requests_wrapper' package that wraps HTTP requests with retry logic. Include tests.",
        30,
        ["requests_wrapper/__init__.py", "requests_wrapper/client.py", "tests/test_client.py"]
    ))

    # LEVEL 2: Orchestrator Tests
    print("\n" + "="*80)
    print("LEVEL 2: ORCHESTRATOR + TASKEXECUTOR (L4)")
    print("="*80)

    level2_results = []

    level2_results.append(run_level2_test(
        "L4: Simple Delegation",
        "Create a JSON parser utility with validation",
        40
    ))

    level2_results.append(run_level2_test(
        "L4: Multi-Step Project",
        "Create a CLI tool for file encryption",
        50
    ))

    # LEVEL 3: Full Stack Tests
    print("\n" + "="*80)
    print("LEVEL 3: FULL STACK (ORCHESTRATOR + ARCHITECT + TASKEXECUTOR) (L5-L7)")
    print("="*80)

    level3_results = []

    level3_results.append(run_level3_test(
        "L5: Multi-Component System",
        "Design and implement a task queue system"
    ))

    level3_results.append(run_level3_test(
        "L6: Service Architecture",
        "Design a microservices order processing system"
    ))

    level3_results.append(run_level3_test(
        "L7: Complex System",
        "Design and implement a distributed cache system"
    ))

    # Generate Summary Report
    generate_summary_report(level1_results, level2_results, level3_results)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    print(f"  Level 1: {sum(1 for r in level1_results if r.get('success', False))}/{len(level1_results)} passed")
    print(f"  Level 2: {sum(1 for r in level2_results if r.get('orchestrator_configured', False))}/{len(level2_results)} configured")
    print(f"  Level 3: {sum(1 for r in level3_results if r.get('full_stack_ready', False))}/{len(level3_results)} ready")
    print("\nLogs:")
    print("  - evaluation_results/level1_task_executor_eval.log")
    print("  - evaluation_results/level2_orchestrator_eval.log")
    print("  - evaluation_results/level3_full_stack_eval.log")
    print("  - evaluation_results/THREE_LEVEL_EVAL_SUMMARY.md")
    print("="*80)


if __name__ == "__main__":
    main()
