#!/usr/bin/env python3
"""
Debug completion detection flow control.

Traces exactly what the LLM sees in its context when nudges are sent,
and how it responds.
"""
import sys
import tempfile
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from task_executor_agent import TaskExecutorAgent
from completion_detector import analyze_llm_response


def trace_messages(agent):
    """Pretty print the current message stack."""
    print("\n" + "="*70)
    print("CURRENT MESSAGE STACK")
    print("="*70)

    messages = agent.state.messages
    for i, msg in enumerate(messages[-10:]):  # Last 10 messages
        role = msg.get('role', 'unknown')

        if role == 'user':
            content = msg.get('content', '')[:200]
            print(f"\n[{i}] USER:")
            print(f"    {content}")

        elif role == 'assistant':
            content = msg.get('content', '')
            tool_calls = msg.get('tool_calls', [])

            print(f"\n[{i}] ASSISTANT:")
            if content:
                print(f"    Content: {content[:200]}")
            if tool_calls:
                print(f"    Tool calls: {[tc['function']['name'] for tc in tool_calls]}")

        elif role == 'tool':
            content = msg.get('content', '')
            try:
                parsed = json.loads(content)

                # Check for nudge
                if '_nudge' in parsed:
                    print(f"\n[{i}] TOOL (WITH NUDGE ⚠️):")
                    print(f"    Result: {str(parsed.get('result', {}))[:100]}")
                    print(f"    NUDGE: {parsed['_nudge'][:150]}...")
                else:
                    print(f"\n[{i}] TOOL:")
                    print(f"    {str(parsed)[:150]}")
            except:
                print(f"\n[{i}] TOOL:")
                print(f"    {content[:150]}")


def run_debug_task():
    """Run a simple task and trace the completion flow."""
    print("="*70)
    print("DEBUGGING COMPLETION DETECTION FLOW")
    print("="*70)
    print("\nTask: Create a simple Python file with one function")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        agent = TaskExecutorAgent(
            workspace=workspace,
            goal="Create test.py with a function add(a, b) that returns a+b",
            max_rounds=8,
            model="gpt-oss:20b",
            temperature=0.2
        )

        # Manually run rounds with tracing
        for round_no in range(1, 9):
            print(f"\n{'='*70}")
            print(f"ROUND {round_no}")
            print(f"{'='*70}")

            # Build context
            context = agent.build_context()

            # Show what LLM will see
            print("\nCONTEXT BEING SENT TO LLM:")
            print(f"  Messages: {len(context)}")
            for i, msg in enumerate(context):
                role = msg.get('role')
                if role == 'system':
                    print(f"  [{i}] SYSTEM: {msg.get('content', '')[:100]}...")
                elif role == 'user':
                    content = msg.get('content', '')
                    if len(content) > 200:
                        print(f"  [{i}] USER: {content[:200]}...")
                    else:
                        print(f"  [{i}] USER: {content}")
                elif role == 'assistant':
                    tool_calls = msg.get('tool_calls', [])
                    if tool_calls:
                        print(f"  [{i}] ASSISTANT: tool_calls={[tc['function']['name'] for tc in tool_calls]}")
                    else:
                        print(f"  [{i}] ASSISTANT: {msg.get('content', '')[:100]}")
                elif role == 'tool':
                    content = msg.get('content', '')
                    try:
                        parsed = json.loads(content)
                        if '_nudge' in parsed:
                            print(f"  [{i}] TOOL: (HAS NUDGE) {parsed.get('_nudge', '')[:80]}...")
                        else:
                            print(f"  [{i}] TOOL: {str(parsed)[:80]}...")
                    except:
                        print(f"  [{i}] TOOL: {content[:80]}...")

            # Call LLM
            from llm_utils import chat_with_inactivity_timeout
            import time

            start = time.time()
            response = chat_with_inactivity_timeout(
                model="gpt-oss:20b",
                messages=context,
                tools=agent.get_tools(),
                options={"temperature": 0.2}
            )
            duration = time.time() - start

            print(f"\nLLM RESPONSE (took {duration:.2f}s):")

            if "message" not in response:
                print("  No message in response!")
                break

            msg = response["message"]

            # Show response content
            if msg.get('content'):
                print(f"  Content: {msg['content'][:200]}")

            # Show tool calls
            tool_calls = msg.get('tool_calls', [])
            if tool_calls:
                print(f"  Tool calls:")
                for tc in tool_calls:
                    tool_name = tc['function']['name']
                    args = tc['function'].get('arguments', {})
                    print(f"    - {tool_name}({list(args.keys())})")

                    # CRITICAL: Check if mark_subtask_complete was called
                    if tool_name == 'mark_subtask_complete':
                        print(f"      🎯 COMPLETION DETECTED! Args: {args}")

                # Analyze for completion signals
                current_task = agent.context_manager._get_current_task()
                current_subtask = current_task.active_subtask() if current_task else None
                subtask_desc = current_subtask.description if current_subtask else None

                analysis = analyze_llm_response(msg.get("content", ""), tool_calls, subtask_desc)

                print(f"\nCOMPLETION ANALYSIS:")
                print(f"  Has completion signal: {analysis['has_completion_signal']}")
                print(f"  Should nudge: {analysis['should_nudge']}")
                print(f"  Reason: {analysis['reason']}")
                if analysis['matched_phrases']:
                    print(f"  Matched phrases: {analysis['matched_phrases']}")
                if analysis['should_nudge']:
                    print(f"  NUDGE MESSAGE: {analysis['nudge_message'][:150]}...")

                # Add assistant message
                agent.state.messages.append(msg)
                agent.add_message(msg)

                # Execute tools
                for idx, tool_call in enumerate(tool_calls):
                    is_last_call = (idx == len(tool_calls) - 1)

                    result = agent.dispatch_tool(tool_call)

                    # Add nudge if needed
                    if is_last_call and analysis["should_nudge"]:
                        result_with_nudge = result.copy() if isinstance(result, dict) else {"result": result}
                        result_with_nudge["_nudge"] = analysis["nudge_message"]
                        result = result_with_nudge
                        print(f"\n  ⚠️ NUDGE ADDED TO TOOL RESULT")

                    # Add tool result to messages
                    tool_result_str = json.dumps(result)
                    agent.state.messages.append({
                        "role": "tool",
                        "content": tool_result_str,
                    })

                    print(f"\n  Tool result added to messages (length: {len(tool_result_str)} chars)")

                # Check if completed
                actual_result = result.get("result") if isinstance(result, dict) and "result" in result else result
                if isinstance(actual_result, dict) and actual_result.get("status") == "goal_complete":
                    print("\n🎉 GOAL COMPLETE!")
                    break

            else:
                print("  No tool calls - LLM is just responding with text")
                agent.state.messages.append(msg)
                agent.add_message(msg)

            # Increment round
            agent.state.total_rounds += 1

            # Check if test.py was created
            test_file = agent.workspace_manager.workspace_dir / "test.py"
            if test_file.exists():
                print(f"\n✓ test.py exists")
                with open(test_file) as f:
                    content = f.read()
                    if 'def add' in content:
                        print(f"✓ add() function found")

        # Final trace
        trace_messages(agent)

        print("\n" + "="*70)
        print("DEBUG COMPLETE")
        print("="*70)


if __name__ == "__main__":
    run_debug_task()
