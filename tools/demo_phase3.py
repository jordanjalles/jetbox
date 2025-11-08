#!/usr/bin/env python3
"""
Demo script for Phase 3: Analysis Engine

Shows the analysis capabilities with sample data.
"""

from pathlib import Path
from analyze_context import (
    load_snapshots,
    analyze_duplication,
    analyze_growth,
    attribute_tokens_to_behaviors,
    generate_recommendations,
    generate_report,
)


def main():
    """Run analysis demo on test data."""
    print("=" * 70)
    print("Phase 3 Analysis Engine Demo")
    print("=" * 70)
    print()

    # Load test snapshots
    snapshot_dir = Path(".context_inspection/test_data")
    print(f"Loading snapshots from: {snapshot_dir}")
    print()

    try:
        snapshots = load_snapshots(snapshot_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("\nPlease run from workspace root with test data in place.")
        return

    print(f"✓ Loaded {len(snapshots)} snapshots\n")

    # Show snapshot overview
    print("Snapshot Overview:")
    print("-" * 70)
    for i, snap in enumerate(snapshots, 1):
        agent = snap.get("agent_name", "unknown")
        round_num = snap.get("round", 0)
        context_len = snap.get("metrics", {}).get("total_context_length", 0)
        msg_count = snap.get("metrics", {}).get("total_messages", 0)
        print(
            f"{i}. {agent:20s} Round {round_num:2d}  |  "
            f"{context_len:5d} chars  |  {msg_count} messages"
        )
    print()

    # Duplication analysis
    print("\n" + "=" * 70)
    print("1. DUPLICATION ANALYSIS")
    print("=" * 70)
    duplication = analyze_duplication(snapshots)

    exact = duplication["exact_duplicates"]
    fuzzy = duplication["fuzzy_duplicates"]
    waste = duplication["token_waste"]

    print(f"\nExact Duplicates:")
    print(f"  - System prompts: {exact['system_prompts']['count']}")
    print(f"  - Messages:       {exact['messages']['count']}")
    print(f"  - Tools:          {exact['tools']['count']}")

    print(f"\nFuzzy Duplicates (>80% similarity):")
    print(f"  - System prompts: {fuzzy['system_prompts']['count']}")
    print(f"  - Messages:       {fuzzy['messages']['count']}")

    print(f"\nToken Waste:")
    print(f"  - Exact:          {waste['exact']:,} tokens")
    print(f"  - Fuzzy (est):    {waste['fuzzy_estimated']:,} tokens")
    print(f"  - Total (est):    {waste['total_estimated']:,} tokens")

    # Example duplicates
    if exact["system_prompts"]["examples"]:
        print(f"\nExample: First exact duplicate system prompt:")
        example = exact["system_prompts"]["examples"][0]
        print(f"  Text: {example['text'][:80]}...")
        print(f"  Locations: {', '.join(example['locations'])}")
        print(f"  Tokens: {example['tokens']}")

    # Growth analysis
    print("\n" + "=" * 70)
    print("2. GROWTH ANALYSIS")
    print("=" * 70)
    growth = analyze_growth(snapshots)

    for agent_name, agent_data in growth["agents"].items():
        print(f"\nAgent: {agent_name}")
        print(f"  Rounds analyzed:  {agent_data['rounds']}")
        print(
            f"  Initial context:  {agent_data['initial_context_length']:,} chars"
        )
        print(f"  Final context:    {agent_data['final_context_length']:,} chars")
        print(f"  Growth:           {agent_data['growth_absolute']:,} chars")
        print(f"  Growth rate:      {agent_data['growth_rate_percent']:.1f}%")
        print(f"  Pattern:          {agent_data['growth_pattern'].upper()}")

        projection = agent_data.get("projection", {})
        rounds_to_limit = projection.get("rounds_to_limit")
        if rounds_to_limit:
            print(f"  → Token limit in: ~{rounds_to_limit} rounds")

        if agent_data.get("causes"):
            print(f"  Causes:")
            for cause in agent_data["causes"]:
                print(f"    • {cause}")

    # Behavior attribution
    print("\n" + "=" * 70)
    print("3. BEHAVIOR TOKEN ATTRIBUTION")
    print("=" * 70)
    behaviors = attribute_tokens_to_behaviors(snapshots)

    print(
        f"\n{'Behavior':<30s} {'System':<10s} {'Tools':<10s} {'Total':<10s}"
    )
    print("-" * 70)
    for behavior, tokens in behaviors.items():
        sys_tokens = tokens["system_prompt_tokens"]
        tool_tokens = tokens["tool_tokens"]
        total = tokens["total_tokens"]
        print(f"{behavior:<30s} {sys_tokens:<10,d} {tool_tokens:<10,d} {total:<10,d}")

    # Recommendations
    print("\n" + "=" * 70)
    print("4. RECOMMENDATIONS")
    print("=" * 70)

    analysis_results = {
        "duplication": duplication,
        "growth": growth,
        "behaviors": behaviors,
    }

    recommendations = generate_recommendations(analysis_results)

    for i, rec in enumerate(recommendations, 1):
        priority = rec["priority"]
        icon = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"

        print(f"\n{i}. {icon} [{priority}] {rec['title']}")
        print(f"   Category: {rec['category']}")
        print(f"   Savings:  ~{rec['potential_savings']:,} tokens")
        print(f"   Effort:   {rec['difficulty']}")
        print(f"   Fixes:")
        for fix in rec["fixes"]:
            print(f"     • {fix}")

    # Generate report
    print("\n" + "=" * 70)
    print("5. REPORT GENERATION")
    print("=" * 70)

    output_file = Path(".context_inspection/demo_report.md")
    analysis_results["snapshot_dir"] = str(snapshot_dir)
    analysis_results["recommendations"] = recommendations

    generate_report(analysis_results, output_file)
    print(f"\n✓ Report saved to: {output_file}")

    # Show report preview
    print(f"\nReport preview (first 15 lines):")
    print("-" * 70)
    with open(output_file) as f:
        for i, line in enumerate(f, 1):
            if i > 15:
                break
            print(line.rstrip())

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print(f"\nFull report: {output_file}")
    print(f"Test command: python tools/analyze_context.py {snapshot_dir}")
    print()


if __name__ == "__main__":
    main()
