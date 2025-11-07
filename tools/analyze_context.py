#!/usr/bin/env python3
"""
Context Inspection Analysis Engine - Phase 3

Analyzes captured context snapshots to find inefficiencies, duplication,
and optimization opportunities.

Usage:
    python tools/analyze_context.py .context_inspection
    python tools/analyze_context.py .context_inspection --output report.md
"""

import json
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
from collections import defaultdict
import sys


def load_snapshots(snapshot_dir: Path) -> list[dict[str, Any]]:
    """
    Load all JSON snapshots from directory.

    Args:
        snapshot_dir: Path to directory containing snapshot JSON files

    Returns:
        List of snapshot dicts, sorted by agent_name then round number

    Raises:
        FileNotFoundError: If snapshot directory doesn't exist
        ValueError: If no valid snapshots found
    """
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

    snapshot_files = list(snapshot_dir.glob("*.json"))

    if not snapshot_files:
        raise ValueError(f"No snapshot files found in {snapshot_dir}")

    snapshots = []
    errors = []

    for snapshot_file in snapshot_files:
        try:
            with open(snapshot_file) as f:
                snapshot = json.load(f)
                snapshots.append(snapshot)
        except json.JSONDecodeError as e:
            errors.append(f"Failed to parse {snapshot_file.name}: {e}")
        except Exception as e:
            errors.append(f"Failed to load {snapshot_file.name}: {e}")

    if errors:
        print("Warning: Some snapshot files had errors:")
        for error in errors:
            print(f"  - {error}")

    if not snapshots:
        raise ValueError(f"No valid snapshots loaded from {snapshot_dir}")

    # Sort by agent_name, then round number
    snapshots.sort(key=lambda s: (s.get("agent_name", ""), s.get("round", 0)))

    print(f"Loaded {len(snapshots)} snapshots from {snapshot_dir}")
    return snapshots


def _calculate_tokens(text: str) -> int:
    """Approximate token count (char count / 4)."""
    return len(text) // 4


def _generate_latency_chart_horizontal(
    latency_data: list[dict],
    max_width: int = 60,
    agent_name: str = ""
) -> str:
    """
    Generate horizontal bar chart showing latency per round.

    Args:
        latency_data: List of dicts with 'round', 'latency_seconds'
        max_width: Maximum width for bars
        agent_name: Name of agent for chart title

    Returns:
        ASCII art horizontal bar chart
    """
    if not latency_data:
        return "No latency data available"

    # Find max latency for scaling
    max_latency = max(l.get('latency_seconds', 0) for l in latency_data)

    if max_latency == 0:
        return "No latency data available"

    output_lines = []

    # Title with agent name
    title = f"Latency per round - {agent_name} (seconds)" if agent_name else "Latency per round (seconds)"
    output_lines.append(title)
    output_lines.append("")

    # Header with scale
    scale_markers = [0, max_latency / 4, max_latency / 2, (3 * max_latency) / 4]
    header = "ROUND  │ "
    spacing = max_width // 4
    for i, marker in enumerate(scale_markers):
        header += f"{marker:.1f}s{' ' * (spacing - 4)}"
    output_lines.append(header)
    output_lines.append("───────┼" + "─" * max_width)

    # Bar for each round
    for data in latency_data:
        round_num = data.get('round', 0)
        latency = data.get('latency_seconds', 0)
        is_delegation = data.get('is_delegation', False)
        delegated_to_agent = data.get('delegated_to_agent')

        # Calculate bar width (proportional to max)
        if latency > 0 and max_latency > 0:
            bar_width = int((latency / max_latency) * max_width)
            bar = "█" * bar_width
        else:
            bar = ""

        # Add delegation tag if applicable
        time_str = f"{latency:.2f}s"
        if is_delegation:
            if delegated_to_agent:
                time_str += f" [delegation to {delegated_to_agent}]"
            else:
                time_str += " [delegation]"

        line = f"  {round_num:>2}   │ {bar} {time_str}"
        output_lines.append(line)

    return "\n".join(output_lines)


def _generate_ascii_chart_horizontal(
    snapshots_data: list[dict],
    max_width: int = 60,
    agent_name: str = ""
) -> str:
    """
    Generate horizontal bar chart with rounds on Y-axis, color-coded by content type.

    Args:
        snapshots_data: List of dicts with granular token breakdown
        max_width: Maximum width for bars
        agent_name: Name of agent for chart title

    Returns:
        ASCII art horizontal bar chart with colors
    """
    if not snapshots_data:
        return "No data available"

    # Use distinct ASCII characters for each category (markdown-compatible)
    # No ANSI colors since markdown files don't support them
    SYSTEM_CHAR = "█"       # System prompt
    TOOLDEF_CHAR = "▓"      # Tool definitions
    TOOLCALL_CHAR = "▒"     # Tool calls
    TOOLRES_CHAR = "░"      # Tool responses
    THINKING_CHAR = "●"     # Thinking
    MESSAGE_CHAR = "○"      # Messages

    # Find max total for scaling
    max_total = 0
    for s in snapshots_data:
        total = (s.get('system_tokens', 0) + s.get('tool_def_tokens', 0) +
                s.get('tool_call_tokens', 0) + s.get('tool_response_tokens', 0) +
                s.get('thinking_tokens', 0) + s.get('user_message_tokens', 0) +
                s.get('assistant_content_tokens', 0))
        if total > max_total:
            max_total = total

    if max_total == 0:
        return "No token data available"

    output_lines = []

    # Title with legend (using distinct ASCII characters)
    title = f"Context composition by round - {agent_name}" if agent_name else "Context composition by round"
    output_lines.append(title)
    legend = (f"{SYSTEM_CHAR} System  "
             f"{TOOLDEF_CHAR} ToolDefs  "
             f"{TOOLCALL_CHAR} ToolCalls  "
             f"{TOOLRES_CHAR} ToolResults  "
             f"{THINKING_CHAR} Thinking  "
             f"{MESSAGE_CHAR} Messages")
    output_lines.append(legend)
    output_lines.append("")

    # Header with scale
    scale_markers = [0, max_total // 4, max_total // 2, (3 * max_total) // 4, max_total]
    header = "ROUND  │ "
    for marker in scale_markers:
        if marker < max_total:
            header += f"{marker:<{max_width // 5}}"
    output_lines.append(header)
    output_lines.append("───────┼" + "─" * max_width)

    # Bar for each round
    for snap in snapshots_data:
        round_num = snap.get('round', 0)
        system_tokens = snap.get('system_tokens', 0)
        tool_def_tokens = snap.get('tool_def_tokens', 0)
        tool_call_tokens = snap.get('tool_call_tokens', 0)
        tool_response_tokens = snap.get('tool_response_tokens', 0)
        thinking_tokens = snap.get('thinking_tokens', 0)
        user_message_tokens = snap.get('user_message_tokens', 0)
        assistant_content_tokens = snap.get('assistant_content_tokens', 0)

        total_tokens = (system_tokens + tool_def_tokens + tool_call_tokens +
                       tool_response_tokens + thinking_tokens + user_message_tokens +
                       assistant_content_tokens)

        # Calculate bar widths (each component scaled to max_total, not to round's total)
        # This keeps fixed components like system prompt at constant visual width
        if total_tokens > 0:
            # Calculate each segment width directly as proportion of max_total
            system_width = int((system_tokens / max_total) * max_width)
            tool_def_width = int((tool_def_tokens / max_total) * max_width)
            tool_call_width = int((tool_call_tokens / max_total) * max_width)
            tool_response_width = int((tool_response_tokens / max_total) * max_width)
            thinking_width = int((thinking_tokens / max_total) * max_width)
            message_width = int(((user_message_tokens + assistant_content_tokens) / max_total) * max_width)

            # Calculate actual total bar width from segments
            total_bar_width = (system_width + tool_def_width + tool_call_width +
                             tool_response_width + thinking_width + message_width)

            if total_bar_width > 0:
                # Build bar with distinct ASCII characters
                bar = (SYSTEM_CHAR * system_width +
                      TOOLDEF_CHAR * tool_def_width +
                      TOOLCALL_CHAR * tool_call_width +
                      TOOLRES_CHAR * tool_response_width +
                      THINKING_CHAR * thinking_width +
                      MESSAGE_CHAR * message_width)
            else:
                bar = ""
        else:
            bar = ""

        # Add total label at end
        line = f"  {round_num:>2}   │ {bar} {total_tokens:,}"
        output_lines.append(line)

    output_lines.append("")

    return "\n".join(output_lines)


def _find_exact_duplicates(texts: list[tuple[str, str]]) -> dict[str, list[str]]:
    """
    Find exact duplicate text strings.

    Args:
        texts: List of (identifier, text) tuples

    Returns:
        Dict mapping text to list of identifiers where it appears
    """
    text_to_locations = defaultdict(list)

    for identifier, text in texts:
        if text:  # Skip empty strings
            text_to_locations[text].append(identifier)

    # Filter to only duplicates
    duplicates = {
        text: locations
        for text, locations in text_to_locations.items()
        if len(locations) > 1
    }

    return duplicates


def _find_fuzzy_duplicates(
    texts: list[tuple[str, str]], similarity_threshold: float = 0.8
) -> list[dict[str, Any]]:
    """
    Find fuzzy duplicate text strings (>threshold similarity).

    Args:
        texts: List of (identifier, text) tuples
        similarity_threshold: Minimum similarity ratio (0-1)

    Returns:
        List of duplicate groups with similarity scores
    """
    fuzzy_duplicates = []

    # Only check texts above minimum length (avoid false positives)
    min_length = 100
    filtered_texts = [(id_, text) for id_, text in texts if len(text) >= min_length]

    # Compare each pair of texts
    for i, (id1, text1) in enumerate(filtered_texts):
        for id2, text2 in filtered_texts[i + 1 :]:
            similarity = SequenceMatcher(None, text1, text2).ratio()

            if similarity >= similarity_threshold:
                fuzzy_duplicates.append(
                    {
                        "locations": [id1, id2],
                        "similarity": similarity,
                        "length": len(text1),
                        "tokens": _calculate_tokens(text1),
                        "sample": text1[:200],
                    }
                )

    return fuzzy_duplicates


def analyze_duplication(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Find exact and fuzzy duplicates in context.

    Analyzes:
    - WITHIN-ROUND duplication (same content appearing multiple times in a single snapshot)
    - Cross-round repetition (content persisting across rounds - normal for accumulated context)

    Args:
        snapshots: List of snapshot dicts

    Returns:
        Duplication report with token waste calculation
    """
    print("\nAnalyzing duplication...")

    # Analyze each snapshot independently for WITHIN-ROUND duplicates
    within_round_duplicates = []
    cross_round_tracking = defaultdict(set)  # Track which rounds contain each unique text

    for snapshot in snapshots:
        agent_name = snapshot.get("agent_name", "unknown")
        round_num = snapshot.get("round", 0)
        identifier = f"{agent_name}_r{round_num}"

        # Collect texts within THIS snapshot only
        system_prompts = []
        messages = []
        tool_defs = []

        context = snapshot.get("context", [])
        for i, msg in enumerate(context):
            if msg.get("role") == "system":
                system_prompts.append((f"{identifier}_system_{i}", msg.get("content", "")))
                cross_round_tracking[("system", msg.get("content", ""))].add(round_num)
            else:
                messages.append(
                    (f"{identifier}_{msg.get('role')}_{i}", msg.get("content", ""))
                )
                cross_round_tracking[("message", msg.get("content", ""))].add(round_num)

        tools = snapshot.get("tools", [])
        for i, tool in enumerate(tools):
            tool_name = tool.get("function", {}).get("name", f"tool_{i}")
            tool_str = json.dumps(tool, sort_keys=True)
            tool_defs.append((f"{identifier}_{tool_name}", tool_str))
            cross_round_tracking[("tool", tool_str)].add(round_num)

        # Find duplicates WITHIN this single round
        within_exact_system = _find_exact_duplicates(system_prompts)
        within_exact_messages = _find_exact_duplicates(messages)
        within_exact_tools = _find_exact_duplicates(tool_defs)

        if within_exact_system or within_exact_messages or within_exact_tools:
            within_round_duplicates.append({
                "snapshot": identifier,
                "round": round_num,
                "system_prompts": within_exact_system,
                "messages": within_exact_messages,
                "tools": within_exact_tools,
            })

    # Calculate within-round duplication waste
    within_round_waste = 0
    total_within_duplicates = 0

    for dup_info in within_round_duplicates:
        for text, locations in dup_info["system_prompts"].items():
            tokens = _calculate_tokens(text)
            within_round_waste += tokens * (len(locations) - 1)
            total_within_duplicates += len(locations) - 1

        for text, locations in dup_info["messages"].items():
            tokens = _calculate_tokens(text)
            within_round_waste += tokens * (len(locations) - 1)
            total_within_duplicates += len(locations) - 1

        for text, locations in dup_info["tools"].items():
            tokens = _calculate_tokens(text)
            within_round_waste += tokens * (len(locations) - 1)
            total_within_duplicates += len(locations) - 1

    # Analyze cross-round repetition (normal context accumulation)
    cross_round_stats = {
        "system_prompts": 0,
        "messages": 0,
        "tools": 0,
    }

    for (content_type, _), rounds in cross_round_tracking.items():
        if len(rounds) > 1:
            if content_type == "system":
                cross_round_stats["system_prompts"] += 1
            elif content_type == "message":
                cross_round_stats["messages"] += 1
            elif content_type == "tool":
                cross_round_stats["tools"] += 1

    report = {
        "within_round_duplicates": {
            "count": len(within_round_duplicates),
            "total_duplicate_instances": total_within_duplicates,
            "snapshots_with_duplicates": [
                {
                    "snapshot": dup["snapshot"],
                    "round": dup["round"],
                    "system_prompt_dups": len(dup["system_prompts"]),
                    "message_dups": len(dup["messages"]),
                    "tool_dups": len(dup["tools"]),
                }
                for dup in within_round_duplicates
            ],
        },
        "cross_round_repetition": {
            "description": "Content appearing in multiple rounds (normal for accumulated context)",
            "stats": cross_round_stats,
            "total_unique_content": len(cross_round_tracking),
        },
        "token_waste": {
            "within_round": within_round_waste,
            "cross_round_note": "Cross-round repetition is normal context accumulation, not waste",
        },
    }

    print(f"  - Within-round duplicates: {total_within_duplicates} instances across {len(within_round_duplicates)} snapshots")
    print(f"  - Cross-round repetition: {cross_round_stats['messages']} messages, {cross_round_stats['system_prompts']} system prompts, {cross_round_stats['tools']} tools")
    print(f"  - Token waste (within-round only): {within_round_waste:,}")

    return report


def analyze_growth(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Track context size trends and growth patterns.

    Analyzes:
    - Context size over rounds
    - Growth rate (linear/exponential)
    - Growth causes (messages, tools, system prompt changes)
    - Projection to token limit

    Args:
        snapshots: List of snapshot dicts

    Returns:
        Growth analysis report
    """
    print("\nAnalyzing growth patterns...")

    # Group by agent and track first timestamp for ordering
    agent_snapshots = defaultdict(list)
    agent_first_timestamp = {}
    for snapshot in snapshots:
        agent_name = snapshot.get("agent_name", "unknown")
        agent_snapshots[agent_name].append(snapshot)

        # Track first timestamp for this agent
        if agent_name not in agent_first_timestamp:
            agent_first_timestamp[agent_name] = snapshot.get("timestamp", 0)

    # Build global timeline for delegation detection
    # Map timestamps to child agents that started (excluding orchestrator)
    global_timeline = []
    for agent_name, snaps in agent_snapshots.items():
        if agent_name == "orchestrator":
            continue
        sorted_snaps = sorted(snaps, key=lambda s: s.get("timestamp", 0))
        if sorted_snaps:
            first_timestamp = sorted_snaps[0].get("timestamp", 0)
            global_timeline.append((first_timestamp, agent_name))
    global_timeline.sort()  # Sort by timestamp

    agent_analyses = {}

    # Sort agents by first timestamp (execution order)
    sorted_agents = sorted(agent_snapshots.keys(), key=lambda a: agent_first_timestamp.get(a, 0))

    for agent_name in sorted_agents:
        agent_snaps = agent_snapshots[agent_name]
        # Sort by round
        agent_snaps.sort(key=lambda s: s.get("round", 0))

        # Track metrics over rounds
        rounds = []
        context_lengths_tokens = []  # Total tokens including tools
        message_counts = []
        tool_counts = []
        system_prompt_lengths = []
        token_breakdowns = []  # NEW: For chart visualization
        timestamps = []  # Track timestamps for latency calculation

        for snapshot in agent_snaps:
            metrics = snapshot.get("metrics", {})
            rounds.append(snapshot.get("round", 0))
            message_counts.append(metrics.get("total_messages", 0))
            tool_counts.append(metrics.get("tool_count", 0))
            timestamps.append(snapshot.get("timestamp", 0))

            system_len = metrics.get("system_prompt_length", 0)
            system_prompt_lengths.append(system_len)

            # Calculate granular token breakdown for this snapshot
            tool_def_len = metrics.get("tool_definition_length", 0)

            # Break down messages by type
            context_messages = snapshot.get("context", [])
            tool_call_chars = 0
            tool_response_chars = 0
            thinking_chars = 0
            user_message_chars = 0
            assistant_content_chars = 0

            for msg in context_messages:
                role = msg.get("role", "")

                if role == "system":
                    continue  # Already counted in system_len
                elif role == "tool":
                    # Tool responses (file contents, command outputs, etc.)
                    tool_response_chars += len(json.dumps(msg))
                elif role == "assistant":
                    # Assistant messages may have: content, thinking, tool_calls
                    if msg.get("tool_calls"):
                        tool_call_chars += len(json.dumps(msg.get("tool_calls", [])))
                    if msg.get("thinking"):
                        thinking_chars += len(msg.get("thinking", ""))
                    if msg.get("content"):
                        assistant_content_chars += len(msg.get("content", ""))
                elif role == "user":
                    user_message_chars += len(msg.get("content", ""))

            # Convert to tokens (chars / 4)
            system_tokens = system_len // 4
            tool_def_tokens = tool_def_len // 4
            tool_call_tokens = tool_call_chars // 4
            tool_response_tokens = tool_response_chars // 4
            thinking_tokens = thinking_chars // 4
            user_message_tokens = user_message_chars // 4
            assistant_content_tokens = assistant_content_chars // 4

            total_tokens = (system_tokens + tool_def_tokens + tool_call_tokens +
                          tool_response_tokens + thinking_tokens + user_message_tokens +
                          assistant_content_tokens)

            context_lengths_tokens.append(total_tokens)

            token_breakdowns.append({
                "round": snapshot.get("round", 0),
                "system_tokens": system_tokens,
                "tool_def_tokens": tool_def_tokens,
                "tool_call_tokens": tool_call_tokens,
                "tool_response_tokens": tool_response_tokens,
                "thinking_tokens": thinking_tokens,
                "user_message_tokens": user_message_tokens,
                "assistant_content_tokens": assistant_content_tokens,
            })

        # Calculate latency per round (time between consecutive snapshots)
        latency_data = []
        for i in range(1, len(timestamps)):
            if timestamps[i] > 0 and timestamps[i-1] > 0:
                latency_seconds = timestamps[i] - timestamps[i-1]

                # Detect delegation rounds (high latency for orchestrator)
                is_delegation = False
                delegated_to_agent = None
                if agent_name == "orchestrator" and latency_seconds > 10:
                    is_delegation = True

                    # Find which agent started during this time window
                    start_time = timestamps[i-1]
                    end_time = timestamps[i]
                    for child_start_time, child_agent_name in global_timeline:
                        if start_time <= child_start_time <= end_time:
                            delegated_to_agent = child_agent_name
                            break

                latency_data.append({
                    "round": rounds[i],
                    "latency_seconds": latency_seconds,
                    "is_delegation": is_delegation,
                    "delegated_to_agent": delegated_to_agent,
                })
            else:
                latency_data.append({
                    "round": rounds[i],
                    "latency_seconds": 0,
                    "is_delegation": False,
                    "delegated_to_agent": None,
                })

        if len(rounds) < 2:
            continue

        # Calculate growth rate (using tokens including tools)
        initial_length = context_lengths_tokens[0]
        final_length = context_lengths_tokens[-1]
        growth_absolute = final_length - initial_length
        growth_rate = (
            (final_length / initial_length - 1) * 100 if initial_length > 0 else 0
        )

        # Determine growth pattern
        # Check if exponential (each step grows proportionally)
        if len(context_lengths_tokens) >= 3:
            mid_idx = len(context_lengths_tokens) // 2
            first_half_growth = context_lengths_tokens[mid_idx] - context_lengths_tokens[0]
            second_half_growth = context_lengths_tokens[-1] - context_lengths_tokens[mid_idx]

            if second_half_growth > first_half_growth * 1.5:
                pattern = "exponential"
            elif second_half_growth > first_half_growth * 0.5:
                pattern = "linear"
            else:
                pattern = "stable"
        else:
            pattern = "insufficient_data"

        # Project to token limit (assume 128K limit)
        token_limit = 128000
        if growth_absolute > 0 and pattern in ("linear", "exponential"):
            rounds_to_limit = None
            if pattern == "linear":
                avg_growth_per_round = growth_absolute / len(rounds)
                remaining_tokens = token_limit - final_length
                rounds_to_limit = int(remaining_tokens / avg_growth_per_round)
            elif pattern == "exponential" and len(context_lengths_tokens) >= 3:
                # Estimate exponential growth factor
                growth_factor = (final_length / initial_length) ** (1 / len(rounds))
                if growth_factor > 1:
                    # How many rounds until we exceed limit?
                    import math

                    rounds_to_limit = int(
                        math.log(token_limit / final_length) / math.log(growth_factor)
                    )

            projection = {
                "pattern": pattern,
                "rounds_to_limit": rounds_to_limit,
                "projected_at_100_rounds": int(
                    final_length * (growth_factor ** (100 - len(rounds)))
                )
                if pattern == "exponential" and "growth_factor" in locals()
                else None,
            }
        else:
            projection = {"pattern": pattern, "rounds_to_limit": None}

        # Identify growth causes
        message_growth = message_counts[-1] - message_counts[0]
        tool_growth = tool_counts[-1] - tool_counts[0]
        system_prompt_growth = system_prompt_lengths[-1] - system_prompt_lengths[0]

        causes = []
        if message_growth > 5:
            causes.append(
                f"Message history growing ({message_counts[0]} -> {message_counts[-1]})"
            )
        if tool_growth > 0:
            causes.append(f"Tool count increased ({tool_counts[0]} -> {tool_counts[-1]})")
        if system_prompt_growth > 1000:
            causes.append(
                f"System prompt grew by {system_prompt_growth:,} chars"
            )

        agent_analyses[agent_name] = {
            "rounds": len(rounds),
            "initial_context_length": initial_length,
            "final_context_length": final_length,
            "growth_absolute": growth_absolute,
            "growth_rate_percent": growth_rate,
            "growth_pattern": pattern,
            "projection": projection,
            "causes": causes,
            "history": {
                "rounds": rounds,
                "context_lengths_tokens": context_lengths_tokens,
                "message_counts": message_counts,
                "tool_counts": tool_counts,
            },
            "token_breakdowns": token_breakdowns,  # For context composition chart
            "latency_data": latency_data,  # For latency chart
        }

        print(f"  - {agent_name}: {pattern} growth, {growth_rate:.1f}% increase")

    return {"agents": agent_analyses}


def attribute_tokens_to_behaviors(
    snapshots: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Calculate token contribution per behavior.

    Analyzes:
    - System prompt instructions per behavior
    - Tool definitions per behavior (mapped by tool name patterns)
    - Context injections per behavior

    Args:
        snapshots: List of snapshot dicts

    Returns:
        Behavior contribution matrix
    """
    print("\nAttributing tokens to behaviors...")

    # Use the most recent snapshot for each agent (most complete)
    agent_latest = {}
    for snapshot in snapshots:
        agent_name = snapshot.get("agent_name", "unknown")
        round_num = snapshot.get("round", 0)
        if (
            agent_name not in agent_latest
            or round_num > agent_latest[agent_name].get("round", -1)
        ):
            agent_latest[agent_name] = snapshot

    behavior_contributions = defaultdict(
        lambda: {"system_prompt_tokens": 0, "tool_tokens": 0, "context_tokens": 0}
    )

    # Map tool names to behaviors (based on naming conventions)
    # Behaviors typically provide tools named after themselves
    TOOL_TO_BEHAVIOR_MAPPING = {
        # File and directory tools
        "write_file": "write_file_tools",
        "read_file": "read_file_tools",
        "list_dir": "directory_tools",
        "run_bash": "command_tools",

        # Server tools
        "start_server": "server_tools",
        "stop_server": "server_tools",
        "check_server": "server_tools",
        "list_servers": "server_tools",

        # Task executor tools
        "mark_complete": "task_executor",
        "mark_failed": "task_executor",

        # Delegation tools
        "consult_architect": "delegation",
        "delegate_to_executor": "delegation",

        # Workspace management tools
        "list_workspaces": "workspace_management",
        "search_workspaces": "workspace_management",
        "get_workspace_info": "workspace_management",
        "find_workspace": "workspace_management",

        # Architect tools
        "write_architecture_doc": "architect_tools",
        "read_architecture_doc": "architect_tools",
        "list_architecture_docs": "architect_tools",
        "write_module_spec": "architect_tools",
        "write_task_list": "architect_tools",
    }

    for agent_name, snapshot in agent_latest.items():
        behaviors_loaded = snapshot.get("behaviors_loaded", [])

        # Get all tools and attribute them to specific behaviors
        tools = snapshot.get("tools", [])
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "")
            tool_tokens = _calculate_tokens(json.dumps(tool))

            # Find which behavior provides this tool
            providing_behavior = TOOL_TO_BEHAVIOR_MAPPING.get(tool_name)

            if providing_behavior and providing_behavior in behaviors_loaded:
                behavior_contributions[providing_behavior]["tool_tokens"] += tool_tokens
            elif providing_behavior:
                # Tool provider not in loaded behaviors - might be core agent
                behavior_contributions[providing_behavior]["tool_tokens"] += tool_tokens
            else:
                # Unknown tool - attribute to "unknown" for debugging
                behavior_contributions["unknown_tool_provider"]["tool_tokens"] += tool_tokens

        # Get system prompt
        context = snapshot.get("context", [])
        system_messages = [msg for msg in context if msg.get("role") == "system"]

        if system_messages and behaviors_loaded:
            system_prompt = system_messages[0].get("content", "")
            system_tokens = _calculate_tokens(system_prompt)

            # Try to attribute system prompt sections to behaviors
            # Extract specific sections that behaviors contribute
            behavior_section_markers = {
                "loop_detection": ("LOOP DETECTION:", "WORKSPACE TASK NOTES:"),
                "workspace_task_notes": ("WORKSPACE TASK NOTES:", None),  # Goes to end
            }

            for behavior in behaviors_loaded:
                if behavior in behavior_section_markers:
                    start_marker, end_marker = behavior_section_markers[behavior]
                    start_idx = system_prompt.find(start_marker)

                    if start_idx != -1:
                        # Find where this section ends
                        if end_marker:
                            end_idx = system_prompt.find(end_marker, start_idx)
                            if end_idx == -1:
                                end_idx = len(system_prompt)
                        else:
                            end_idx = len(system_prompt)

                        # Extract just this behavior's section
                        section = system_prompt[start_idx:end_idx]
                        section_tokens = _calculate_tokens(section)
                        behavior_contributions[behavior]["system_prompt_tokens"] += section_tokens

    # Convert to regular dict and add totals
    result = {}
    for behavior, tokens in behavior_contributions.items():
        total = sum(tokens.values())
        result[behavior] = {**tokens, "total_tokens": total}

    # Sort by total tokens
    result = dict(sorted(result.items(), key=lambda x: x[1]["total_tokens"], reverse=True))

    print(f"  - Analyzed {len(result)} behaviors")
    for behavior, tokens in list(result.items())[:5]:
        print(f"    - {behavior}: {tokens['total_tokens']:,} tokens ({tokens['tool_tokens']:,} tools, {tokens['system_prompt_tokens']:,} system)")

    return result


def generate_recommendations(
    analysis_results: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Generate prioritized, actionable recommendations.

    Analyzes all findings and produces:
    - HIGH/MEDIUM/LOW priority issues
    - Specific fixes with file paths
    - Potential token savings
    - Implementation difficulty

    Args:
        analysis_results: Combined analysis from all functions

    Returns:
        Prioritized recommendation list
    """
    print("\nGenerating recommendations...")

    recommendations = []

    # Check duplication findings (within-round only)
    duplication = analysis_results.get("duplication", {})
    token_waste = duplication.get("token_waste", {})
    within_round_waste = token_waste.get("within_round", 0)

    if within_round_waste > 10000:
        priority = "HIGH"
    elif within_round_waste > 5000:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    if within_round_waste > 1000:
        within_round_dups = duplication.get("within_round_duplicates", {})
        dup_count = within_round_dups.get("total_duplicate_instances", 0)
        recommendations.append(
            {
                "priority": priority,
                "category": "duplication",
                "title": f"Reduce within-round duplication ({within_round_waste:,} wasted tokens)",
                "description": f"Found {dup_count} duplicate instances within single context snapshots",
                "fixes": [
                    "Review context building logic for duplicate message insertion",
                    "Ensure tool definitions aren't added multiple times per round",
                    "Check behavior composition for redundant content injection",
                ],
                "potential_savings": within_round_waste,
                "difficulty": "medium",
            }
        )

    # Check growth patterns
    growth = analysis_results.get("growth", {})
    for agent_name, agent_growth in growth.get("agents", {}).items():
        pattern = agent_growth.get("growth_pattern")
        growth_rate = agent_growth.get("growth_rate_percent", 0)
        projection = agent_growth.get("projection", {})
        rounds_to_limit = projection.get("rounds_to_limit")

        if pattern == "exponential":
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "growth",
                    "title": f"Agent '{agent_name}' has exponential context growth",
                    "description": f"Context growing at {growth_rate:.1f}% rate, will hit limits soon",
                    "fixes": [
                        "Implement aggressive context compaction",
                        "Clear message history more frequently",
                        "Reduce tool definition sizes",
                    ],
                    "potential_savings": agent_growth.get("growth_absolute", 0) // 2,
                    "difficulty": "medium",
                    "rounds_to_limit": rounds_to_limit,
                }
            )
        elif pattern == "linear" and growth_rate > 50:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "growth",
                    "title": f"Agent '{agent_name}' has steady context growth",
                    "description": f"Context growing linearly at {growth_rate:.1f}%",
                    "fixes": [
                        "Review context strategy configuration",
                        "Consider message history limits",
                    ],
                    "potential_savings": agent_growth.get("growth_absolute", 0) // 3,
                    "difficulty": "low",
                }
            )

    # Check behavior contributions
    behaviors = analysis_results.get("behaviors", {})
    for behavior, tokens in behaviors.items():
        total = tokens.get("total_tokens", 0)

        if total > 20000:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "behavior_overhead",
                    "title": f"Behavior '{behavior}' uses {total:,} tokens",
                    "description": "Large token contribution from single behavior",
                    "fixes": [
                        f"Review {behavior} for verbose instructions",
                        "Optimize tool definitions",
                        "Consider splitting into smaller behaviors",
                    ],
                    "potential_savings": total // 4,  # Assume 25% reduction possible
                    "difficulty": "medium",
                }
            )
        elif total > 10000:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "behavior_overhead",
                    "title": f"Behavior '{behavior}' uses {total:,} tokens",
                    "description": "Moderate token contribution",
                    "fixes": [f"Review {behavior} for optimization opportunities"],
                    "potential_savings": total // 5,
                    "difficulty": "low",
                }
            )

    # Sort by priority and potential savings
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(
        key=lambda r: (priority_order[r["priority"]], -r["potential_savings"])
    )

    print(f"  - Generated {len(recommendations)} recommendations")
    for rec in recommendations[:3]:
        print(f"    - [{rec['priority']}] {rec['title']}")

    return recommendations


def generate_report(
    analysis_results: dict[str, Any], output_file: Path
) -> None:
    """
    Generate comprehensive markdown report.

    Args:
        analysis_results: Combined analysis from all functions
        output_file: Path to save markdown report
    """
    print(f"\nGenerating report: {output_file}")

    lines = []

    # Header
    lines.append("# Context Inspection Analysis Report")
    lines.append("")

    # Timestamp and metadata
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Generated**: {timestamp}")
    lines.append("")

    # Snapshot directory info
    snapshot_dir = Path(analysis_results.get('snapshot_dir', 'unknown'))
    lines.append(f"**Snapshot Directory**: `{snapshot_dir}`")

    # List snapshot files analyzed
    if snapshot_dir.exists():
        snapshot_files = sorted(snapshot_dir.glob("*.json"))
        lines.append(f"**Snapshots Analyzed**: {len(snapshot_files)}")
        lines.append(f"**Snapshot Files**: See `{snapshot_dir}/`")
        lines.append("")
    else:
        lines.append("**Snapshots Analyzed**: 0")
        lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    duplication = analysis_results.get("duplication", {})
    token_waste = duplication.get("token_waste", {})
    within_round_waste = token_waste.get("within_round", 0)

    growth = analysis_results.get("growth", {})
    agent_count = len(growth.get("agents", {}))

    behavior_count = len(analysis_results.get("behaviors", {}))
    rec_count = len(analysis_results.get("recommendations", []))

    lines.append(f"- **Agents analyzed**: {agent_count}")
    lines.append(f"- **Behaviors detected**: {behavior_count}")
    lines.append(f"- **Token waste from within-round duplication**: {within_round_waste:,}")
    lines.append(f"- **Recommendations**: {rec_count}")
    lines.append("")

    # Top Recommendations
    recommendations = analysis_results.get("recommendations", [])
    if recommendations:
        lines.append("### Top 3 Recommendations")
        lines.append("")
        for i, rec in enumerate(recommendations[:3], 1):
            lines.append(
                f"{i}. **[{rec['priority']}]** {rec['title']} (Save ~{rec['potential_savings']:,} tokens)"
            )
        lines.append("")

    # Growth Analysis
    lines.append("## Growth Analysis")
    lines.append("")

    # Add legend and notes once at the top
    lines.append("**Chart Legend**:")
    lines.append("- **System** (█): System prompt and behavior instructions")
    lines.append("- **ToolDefs** (▓): Tool definitions sent to LLM API")
    lines.append("- **ToolCalls** (▒): Structured function calls from assistant")
    lines.append("- **ToolResults** (░): Tool execution results (file contents, command outputs, etc.)")
    lines.append("- **Thinking** (●): Reasoning traces from thinking-capable models")
    lines.append("- **Messages** (○): User messages and assistant content")
    lines.append("")
    lines.append("**Notes**:")
    lines.append("- **All bars scaled to max context**: Each bar is proportional to the largest context size. Fixed components (System, ToolDefs) remain visually constant, while growing components expand.")
    lines.append("- **Context includes tools**: Token counts include system prompt, tools, and messages (everything sent to LLM)")
    lines.append("- Round 0 includes tool documentation injected by behaviors as user messages")
    lines.append("- Context may shrink when temporary warnings/nudges are removed")
    lines.append("")

    for agent_name, agent_growth in growth.get("agents", {}).items():
        lines.append(f"### Agent: {agent_name}")
        lines.append("")
        lines.append(f"- **Rounds**: {agent_growth.get('rounds', 0)}")
        lines.append(
            f"- **Initial context**: {agent_growth.get('initial_context_length', 0):,} tokens"
        )
        lines.append(
            f"- **Final context**: {agent_growth.get('final_context_length', 0):,} tokens"
        )
        lines.append(
            f"- **Growth rate**: {agent_growth.get('growth_rate_percent', 0):.1f}%"
        )
        lines.append(f"- **Pattern**: {agent_growth.get('growth_pattern', 'unknown')}")
        lines.append("")

        # Add ASCII chart visualization (horizontal with color coding)
        token_breakdowns = agent_growth.get('token_breakdowns', [])
        if token_breakdowns:
            lines.append("**Context Composition by Round**:")
            lines.append("")
            lines.append("```")
            chart = _generate_ascii_chart_horizontal(token_breakdowns, agent_name=agent_name)
            lines.append(chart)
            lines.append("```")
            lines.append("")

        # Add latency chart
        latency_data = agent_growth.get('latency_data', [])
        if latency_data:
            lines.append("**Latency per Round**:")
            lines.append("")
            lines.append("```")
            latency_chart = _generate_latency_chart_horizontal(latency_data, agent_name=agent_name)
            lines.append(latency_chart)
            lines.append("```")
            lines.append("")

        projection = agent_growth.get("projection", {})
        rounds_to_limit = projection.get("rounds_to_limit")
        if rounds_to_limit is not None:
            lines.append(f"- **Rounds to token limit**: {rounds_to_limit}")

        causes = agent_growth.get("causes", [])
        if causes:
            lines.append("- **Growth causes**:")
            for cause in causes:
                lines.append(f"  - {cause}")

        lines.append("")

    # Behavior Contributions
    lines.append("## Behavior Token Contributions")
    lines.append("")

    behaviors = analysis_results.get("behaviors", {})
    if behaviors:
        # Calculate column widths for proper alignment
        max_behavior_len = max(len(b) for b in behaviors.keys()) if behaviors else 20
        max_behavior_len = max(max_behavior_len, len("Behavior"))

        # Create aligned table
        lines.append(f"| {'Behavior':<{max_behavior_len}} | System Prompt | Tools | Total |")
        lines.append(f"|{'-' * (max_behavior_len + 1)}|{'-' * 15}|{'-' * 7}|{'-' * 7}|")

        for behavior, tokens in behaviors.items():
            sys_tokens = tokens.get("system_prompt_tokens", 0)
            tool_tokens = tokens.get("tool_tokens", 0)
            total = tokens.get("total_tokens", 0)
            lines.append(
                f"| {behavior:<{max_behavior_len}} | {sys_tokens:>13,} | {tool_tokens:>5,} | {total:>5,} |"
            )

        lines.append("")

    # All Recommendations
    lines.append("## Detailed Recommendations")
    lines.append("")

    for i, rec in enumerate(recommendations, 1):
        lines.append(f"### {i}. [{rec['priority']}] {rec['title']}")
        lines.append("")
        lines.append(f"**Category**: {rec['category']}")
        lines.append("")
        lines.append(rec["description"])
        lines.append("")
        lines.append("**Suggested fixes**:")
        for fix in rec["fixes"]:
            lines.append(f"- {fix}")
        lines.append("")
        lines.append(f"**Potential savings**: ~{rec['potential_savings']:,} tokens")
        lines.append(f"**Difficulty**: {rec['difficulty']}")
        lines.append("")

    # Duplication Analysis (moved to end)
    lines.append("## Duplication Analysis")
    lines.append("")

    within_round = duplication.get("within_round_duplicates", {})
    cross_round = duplication.get("cross_round_repetition", {})

    lines.append("### Within-Round Duplication")
    lines.append("")
    lines.append(f"- **Snapshots with duplicates**: {within_round.get('count', 0)}")
    lines.append(f"- **Total duplicate instances**: {within_round.get('total_duplicate_instances', 0)}")

    snapshots_with_dups = within_round.get("snapshots_with_duplicates", [])
    if snapshots_with_dups:
        lines.append("")
        lines.append("**Snapshots with within-round duplicates**:")
        for snap in snapshots_with_dups:
            lines.append(f"- {snap['snapshot']}: {snap['system_prompt_dups']} system, {snap['message_dups']} messages, {snap['tool_dups']} tools")

    lines.append("")
    lines.append("### Cross-Round Repetition")
    lines.append("")
    lines.append(cross_round.get("description", ""))
    lines.append("")
    stats = cross_round.get("stats", {})
    lines.append(f"- **Messages appearing in multiple rounds**: {stats.get('messages', 0)}")
    lines.append(f"- **System prompts appearing in multiple rounds**: {stats.get('system_prompts', 0)}")
    lines.append(f"- **Tools appearing in multiple rounds**: {stats.get('tools', 0)}")
    lines.append(f"- **Total unique content tracked**: {cross_round.get('total_unique_content', 0)}")
    lines.append("")

    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {output_file}")


def main(snapshot_dir: Path, output_file: Path | None = None) -> dict[str, Any]:
    """
    Run all analyses and generate comprehensive report.

    Args:
        snapshot_dir: Path to directory containing snapshot JSON files
        output_file: Optional path for markdown report (default: context_report.md)

    Returns:
        Combined analysis results dict
    """
    if output_file is None:
        output_file = Path("context_inspection_report.md")

    print(f"\n{'='*60}")
    print("Context Inspection Analysis")
    print(f"{'='*60}")

    # Load snapshots
    try:
        snapshots = load_snapshots(snapshot_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Run analyses
    duplication = analyze_duplication(snapshots)
    growth = analyze_growth(snapshots)
    behaviors = attribute_tokens_to_behaviors(snapshots)

    # Combine results
    analysis_results = {
        "snapshot_dir": str(snapshot_dir),
        "snapshot_count": len(snapshots),
        "duplication": duplication,
        "growth": growth,
        "behaviors": behaviors,
    }

    # Generate recommendations
    recommendations = generate_recommendations(analysis_results)
    analysis_results["recommendations"] = recommendations

    # Generate report
    generate_report(analysis_results, output_file)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")

    return analysis_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze context inspection snapshots"
    )
    parser.add_argument(
        "snapshot_dir",
        type=Path,
        help="Directory containing snapshot JSON files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("docs/context_inspection/context_inspection_report.md"),
        help="Output markdown report path (default: docs/context_inspection/context_inspection_report.md)",
    )

    args = parser.parse_args()

    main(args.snapshot_dir, args.output)
