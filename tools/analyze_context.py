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
    - System prompt duplication across rounds
    - Message content duplication
    - Tool definition duplication

    Args:
        snapshots: List of snapshot dicts

    Returns:
        Duplication report with token waste calculation
    """
    print("\nAnalyzing duplication...")

    # Collect all text strings with identifiers
    system_prompts = []
    messages = []
    tool_defs = []

    for snapshot in snapshots:
        agent_name = snapshot.get("agent_name", "unknown")
        round_num = snapshot.get("round", 0)
        identifier = f"{agent_name}_r{round_num}"

        # Extract system prompts
        context = snapshot.get("context", [])
        for msg in context:
            if msg.get("role") == "system":
                system_prompts.append((f"{identifier}_system", msg.get("content", "")))
            else:
                messages.append(
                    (f"{identifier}_{msg.get('role')}", msg.get("content", ""))
                )

        # Extract tool definitions
        tools = snapshot.get("tools", [])
        for i, tool in enumerate(tools):
            tool_name = tool.get("function", {}).get("name", f"tool_{i}")
            tool_str = json.dumps(tool, sort_keys=True)
            tool_defs.append((f"{identifier}_{tool_name}", tool_str))

    # Find exact duplicates
    exact_system = _find_exact_duplicates(system_prompts)
    exact_messages = _find_exact_duplicates(messages)
    exact_tools = _find_exact_duplicates(tool_defs)

    # Find fuzzy duplicates (only for system prompts and longer messages)
    fuzzy_system = _find_fuzzy_duplicates(system_prompts, 0.8)
    long_messages = [(id_, text) for id_, text in messages if len(text) >= 200]
    fuzzy_messages = _find_fuzzy_duplicates(long_messages, 0.85)

    # Calculate token waste
    exact_waste = 0
    for text, locations in exact_system.items():
        tokens = _calculate_tokens(text)
        # Each duplicate beyond the first is waste
        exact_waste += tokens * (len(locations) - 1)

    for text, locations in exact_messages.items():
        tokens = _calculate_tokens(text)
        exact_waste += tokens * (len(locations) - 1)

    for text, locations in exact_tools.items():
        tokens = _calculate_tokens(text)
        exact_waste += tokens * (len(locations) - 1)

    fuzzy_waste = sum(dup["tokens"] for dup in fuzzy_system + fuzzy_messages)

    report = {
        "exact_duplicates": {
            "system_prompts": {
                "count": len(exact_system),
                "examples": [
                    {
                        "text": text[:200],
                        "locations": locations,
                        "tokens": _calculate_tokens(text),
                    }
                    for text, locations in list(exact_system.items())[:5]
                ],
            },
            "messages": {
                "count": len(exact_messages),
                "examples": [
                    {
                        "text": text[:200],
                        "locations": locations,
                        "tokens": _calculate_tokens(text),
                    }
                    for text, locations in list(exact_messages.items())[:5]
                ],
            },
            "tools": {
                "count": len(exact_tools),
                "examples": [
                    {
                        "text": text[:200],
                        "locations": locations,
                        "tokens": _calculate_tokens(text),
                    }
                    for text, locations in list(exact_tools.items())[:5]
                ],
            },
        },
        "fuzzy_duplicates": {
            "system_prompts": {"count": len(fuzzy_system), "examples": fuzzy_system[:5]},
            "messages": {"count": len(fuzzy_messages), "examples": fuzzy_messages[:5]},
        },
        "token_waste": {
            "exact": exact_waste,
            "fuzzy_estimated": fuzzy_waste,
            "total_estimated": exact_waste + fuzzy_waste,
        },
    }

    print(f"  - Exact duplicates: {len(exact_system) + len(exact_messages) + len(exact_tools)}")
    print(f"  - Fuzzy duplicates: {len(fuzzy_system) + len(fuzzy_messages)}")
    print(f"  - Estimated token waste: {exact_waste + fuzzy_waste:,}")

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

    # Group by agent
    agent_snapshots = defaultdict(list)
    for snapshot in snapshots:
        agent_name = snapshot.get("agent_name", "unknown")
        agent_snapshots[agent_name].append(snapshot)

    agent_analyses = {}

    for agent_name, agent_snaps in agent_snapshots.items():
        # Sort by round
        agent_snaps.sort(key=lambda s: s.get("round", 0))

        # Track metrics over rounds
        rounds = []
        context_lengths = []
        message_counts = []
        tool_counts = []
        system_prompt_lengths = []

        for snapshot in agent_snaps:
            metrics = snapshot.get("metrics", {})
            rounds.append(snapshot.get("round", 0))
            context_lengths.append(metrics.get("total_context_length", 0))
            message_counts.append(metrics.get("total_messages", 0))
            tool_counts.append(metrics.get("tool_count", 0))
            system_prompt_lengths.append(metrics.get("system_prompt_length", 0))

        if len(rounds) < 2:
            continue

        # Calculate growth rate
        initial_length = context_lengths[0]
        final_length = context_lengths[-1]
        growth_absolute = final_length - initial_length
        growth_rate = (
            (final_length / initial_length - 1) * 100 if initial_length > 0 else 0
        )

        # Determine growth pattern
        # Check if exponential (each step grows proportionally)
        if len(context_lengths) >= 3:
            mid_idx = len(context_lengths) // 2
            first_half_growth = context_lengths[mid_idx] - context_lengths[0]
            second_half_growth = context_lengths[-1] - context_lengths[mid_idx]

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
            elif pattern == "exponential" and len(context_lengths) >= 3:
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
                "context_lengths": context_lengths,
                "message_counts": message_counts,
                "tool_counts": tool_counts,
            },
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
    - Tool definitions per behavior
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

    for agent_name, snapshot in agent_latest.items():
        behaviors_loaded = snapshot.get("behaviors_loaded", [])

        # Get all tools
        tools = snapshot.get("tools", [])
        total_tool_tokens = sum(
            _calculate_tokens(json.dumps(tool)) for tool in tools
        )

        # Estimate tokens per behavior (divide equally for now)
        # In a real implementation, we'd need behavior-specific markers
        if behaviors_loaded:
            tool_tokens_per_behavior = total_tool_tokens // len(behaviors_loaded)
            for behavior in behaviors_loaded:
                behavior_contributions[behavior]["tool_tokens"] += (
                    tool_tokens_per_behavior
                )

        # Get system prompt
        context = snapshot.get("context", [])
        system_messages = [msg for msg in context if msg.get("role") == "system"]

        if system_messages and behaviors_loaded:
            system_prompt = system_messages[0].get("content", "")
            system_tokens = _calculate_tokens(system_prompt)

            # Try to attribute system prompt sections to behaviors
            # Look for behavior markers in system prompt
            for behavior in behaviors_loaded:
                # Simple heuristic: if behavior name appears in system prompt
                if behavior in system_prompt:
                    # Count approximate tokens for that section
                    # This is a rough estimate
                    behavior_contributions[behavior]["system_prompt_tokens"] += (
                        system_tokens // len(behaviors_loaded)
                    )

    # Convert to regular dict and add totals
    result = {}
    for behavior, tokens in behavior_contributions.items():
        total = sum(tokens.values())
        result[behavior] = {**tokens, "total_tokens": total}

    # Sort by total tokens
    result = dict(sorted(result.items(), key=lambda x: x[1]["total_tokens"], reverse=True))

    print(f"  - Analyzed {len(result)} behaviors")
    for behavior, tokens in list(result.items())[:5]:
        print(f"    - {behavior}: {tokens['total_tokens']:,} tokens")

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

    # Check duplication findings
    duplication = analysis_results.get("duplication", {})
    token_waste = duplication.get("token_waste", {})
    total_waste = token_waste.get("total_estimated", 0)

    if total_waste > 10000:
        priority = "HIGH"
    elif total_waste > 5000:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    if total_waste > 1000:
        recommendations.append(
            {
                "priority": priority,
                "category": "duplication",
                "title": f"Reduce context duplication ({total_waste:,} wasted tokens)",
                "description": "Significant duplication detected in system prompts and messages",
                "fixes": [
                    "Implement context deduplication in context strategies",
                    "Cache system prompts instead of rebuilding each round",
                    "Use message compression for repeated content",
                ],
                "potential_savings": total_waste,
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
    lines.append(f"Generated from: {analysis_results.get('snapshot_dir', 'unknown')}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    duplication = analysis_results.get("duplication", {})
    token_waste = duplication.get("token_waste", {})
    total_waste = token_waste.get("total_estimated", 0)

    growth = analysis_results.get("growth", {})
    agent_count = len(growth.get("agents", {}))

    behavior_count = len(analysis_results.get("behaviors", {}))
    rec_count = len(analysis_results.get("recommendations", []))

    lines.append(f"- **Agents analyzed**: {agent_count}")
    lines.append(f"- **Behaviors detected**: {behavior_count}")
    lines.append(f"- **Token waste from duplication**: {total_waste:,}")
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

    # Duplication Analysis
    lines.append("## Duplication Analysis")
    lines.append("")

    exact = duplication.get("exact_duplicates", {})
    fuzzy = duplication.get("fuzzy_duplicates", {})

    lines.append(
        f"- **Exact duplicate system prompts**: {exact.get('system_prompts', {}).get('count', 0)}"
    )
    lines.append(
        f"- **Exact duplicate messages**: {exact.get('messages', {}).get('count', 0)}"
    )
    lines.append(
        f"- **Exact duplicate tools**: {exact.get('tools', {}).get('count', 0)}"
    )
    lines.append(
        f"- **Fuzzy duplicate system prompts**: {fuzzy.get('system_prompts', {}).get('count', 0)}"
    )
    lines.append(
        f"- **Fuzzy duplicate messages**: {fuzzy.get('messages', {}).get('count', 0)}"
    )
    lines.append("")

    # Growth Analysis
    lines.append("## Growth Analysis")
    lines.append("")

    for agent_name, agent_growth in growth.get("agents", {}).items():
        lines.append(f"### Agent: {agent_name}")
        lines.append("")
        lines.append(f"- **Rounds**: {agent_growth.get('rounds', 0)}")
        lines.append(
            f"- **Initial context**: {agent_growth.get('initial_context_length', 0):,} chars"
        )
        lines.append(
            f"- **Final context**: {agent_growth.get('final_context_length', 0):,} chars"
        )
        lines.append(
            f"- **Growth rate**: {agent_growth.get('growth_rate_percent', 0):.1f}%"
        )
        lines.append(f"- **Pattern**: {agent_growth.get('growth_pattern', 'unknown')}")

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
        lines.append("| Behavior | System Prompt | Tools | Total |")
        lines.append("|----------|---------------|-------|-------|")

        for behavior, tokens in behaviors.items():
            sys_tokens = tokens.get("system_prompt_tokens", 0)
            tool_tokens = tokens.get("tool_tokens", 0)
            total = tokens.get("total_tokens", 0)
            lines.append(
                f"| {behavior} | {sys_tokens:,} | {tool_tokens:,} | {total:,} |"
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
        default=Path("context_inspection_report.md"),
        help="Output markdown report path (default: context_inspection_report.md)",
    )

    args = parser.parse_args()

    main(args.snapshot_dir, args.output)
