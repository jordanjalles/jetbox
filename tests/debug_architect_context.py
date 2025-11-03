#!/usr/bin/env python3
"""
Debug script to inspect architect context and tools.

Shows:
1. What tools the architect has available
2. What the system prompt contains
3. What context is built for a simple task
"""

import json
from pathlib import Path
from architect_agent import ArchitectAgent

def main():
    """Debug architect context construction."""

    print("="*80)
    print("ARCHITECT CONTEXT DEBUG")
    print("="*80)

    # Create architect agent
    architect = ArchitectAgent(workspace=Path(".agent_workspaces/debug_architect"))

    # Add a simple task
    architect.add_message({
        "role": "user",
        "content": "Create a full-stack Flask app with user auth."
    })

    print("\n" + "="*80)
    print("1. TOOLS AVAILABLE")
    print("="*80)

    tools = architect.get_tools()
    print(f"\nTotal tools: {len(tools)}")

    for i, tool in enumerate(tools, 1):
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        print(f"\n{i}. {name}")
        print(f"   {desc[:100]}{'...' if len(desc) > 100 else ''}")

    print("\n" + "="*80)
    print("2. SYSTEM PROMPT")
    print("="*80)

    system_prompt = architect.get_system_prompt()
    print(f"\nLength: {len(system_prompt)} chars")
    print("\n--- PROMPT START ---")
    print(system_prompt)
    print("--- PROMPT END ---")

    print("\n" + "="*80)
    print("3. FULL CONTEXT")
    print("="*80)

    context = architect.build_context()
    print(f"\nMessages: {len(context)}")

    for i, msg in enumerate(context, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"\n{i}. {role.upper()}")
        if len(content) > 200:
            print(f"   {content[:200]}...")
            print(f"   ... ({len(content) - 200} more chars)")
        else:
            print(f"   {content}")

    print("\n" + "="*80)
    print("4. BEHAVIORS LOADED")
    print("="*80)

    print(f"\nTotal behaviors: {len(architect._behaviors)}")

    for i, behavior in enumerate(architect._behaviors, 1):
        name = behavior.get_name()
        tools_count = len(behavior.get_tools())
        print(f"{i}. {name} ({tools_count} tools)")

    print("\n" + "="*80)
    print("5. TOOL NAMES LIST")
    print("="*80)

    tool_names = [t.get("function", {}).get("name") for t in tools]
    print(f"\n{json.dumps(tool_names, indent=2)}")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Check for expected tools
    expected_tools = ["write_architecture_doc", "write_module_spec", "write_task_list", "mark_complete"]
    unexpected_tools = ["write_file", "read_file", "list_dir", "run_bash"]

    print("\nExpected architect tools:")
    for tool in expected_tools:
        if tool in tool_names:
            print(f"  ✓ {tool}")
        else:
            print(f"  ✗ MISSING: {tool}")

    print("\nUnexpected file/command tools:")
    for tool in unexpected_tools:
        if tool in tool_names:
            print(f"  ⚠️  PRESENT (should not be): {tool}")
        else:
            print(f"  ✓ Not present (correct): {tool}")

    # Check if tool documentation is in system prompt
    print("\nTool documentation in system prompt:")
    if "write_architecture_doc" in system_prompt:
        print("  ✓ Tool names found in system prompt")
    else:
        print("  ✗ Tool names NOT found in system prompt")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
