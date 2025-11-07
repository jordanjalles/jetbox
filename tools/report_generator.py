#!/usr/bin/env python3
"""
Context Inspection Report Generator (Phase 5)

Generates comprehensive, actionable markdown reports from context inspection data.
Reports include visualizations, metrics, and prioritized recommendations.
"""

import json
from pathlib import Path
from typing import Any
from datetime import datetime
from collections import defaultdict


def generate_ascii_chart(values: list[float], width: int = 50, height: int = 10) -> str:
    """Generate ASCII art line chart for values."""
    if not values:
        return "No data"

    max_val = max(values)
    min_val = min(values)

    if max_val == min_val:
        # All values are the same
        lines = []
        for i in range(height):
            if i == height // 2:
                lines.append("─" * width)
            else:
                lines.append(" " * width)
        return "\n".join(lines)

    # Normalize values to chart height
    normalized = [(v - min_val) / (max_val - min_val) * (height - 1) for v in values]

    # Create chart grid
    chart = [[" " for _ in range(width)] for _ in range(height)]

    # Plot points
    step = len(values) / width
    for x in range(width):
        idx = int(x * step)
        if idx >= len(normalized):
            idx = len(normalized) - 1
        y = height - 1 - int(normalized[idx])
        chart[y][x] = "█"

    # Convert to string
    lines = []
    for row in chart:
        lines.append("".join(row))

    return "\n".join(lines)


def generate_bar_chart(labels: list[str], values: list[float], width: int = 40) -> str:
    """Generate ASCII art horizontal bar chart."""
    if not labels or not values:
        return "No data"

    max_val = max(values)
    if max_val == 0:
        max_val = 1

    lines = []
    max_label_len = max(len(label) for label in labels)

    for label, value in zip(labels, values):
        bar_len = int((value / max_val) * width)
        bar = "█" * bar_len
        padded_label = label.ljust(max_label_len)
        lines.append(f"{padded_label} │ {bar} {value:,.0f}")

    return "\n".join(lines)


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def format_tokens(tokens: int) -> str:
    """Format token count with K suffix."""
    if tokens < 1000:
        return str(tokens)
    return f"{tokens/1000:.1f}K"


def priority_emoji(priority: str) -> str:
    """Get emoji for priority level."""
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority.upper(), "⚪")


class ContextInspectionReport:
    """Generate comprehensive context inspection reports."""

    def __init__(self, analysis_data: dict[str, Any]):
        """
        Initialize report generator.

        Args:
            analysis_data: Analysis results from analyze_context.py
        """
        self.data = analysis_data
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate(self) -> str:
        """Generate complete markdown report."""
        sections = [
            self._generate_header(),
            self._generate_executive_summary(),
            self._generate_scenario_analysis(),
            self._generate_duplication_deep_dive(),
            self._generate_behavior_contribution(),
            self._generate_recommendations(),
            self._generate_comparative_analysis(),
            self._generate_footer(),
        ]

        return "\n\n".join(sections)

    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# Context Inspection Report

**Generated**: {self.timestamp}
**Tool**: Jetbox Context Inspector
**Purpose**: Identify context inefficiencies and optimization opportunities"""

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        scenarios = self.data.get("scenarios", [])

        if not scenarios:
            return "## Executive Summary\n\n*No scenarios analyzed*"

        # Calculate aggregate metrics
        total_rounds = sum(s.get("total_rounds", 0) for s in scenarios)
        avg_context = sum(s.get("avg_context_size", 0) for s in scenarios) / len(scenarios)
        max_context = max(s.get("max_context_size", 0) for s in scenarios)
        total_duplication = sum(s.get("duplication", {}).get("total_duplicated_tokens", 0) for s in scenarios)

        # Get top recommendations
        recommendations = self.data.get("recommendations", [])
        top_recs = recommendations[:3] if len(recommendations) >= 3 else recommendations

        summary = f"""## Executive Summary

### Key Metrics

| Metric | Value |
|--------|-------|
| Scenarios Analyzed | {len(scenarios)} |
| Total Rounds | {total_rounds} |
| Avg Context Size | {format_tokens(int(avg_context))} tokens |
| Max Context Size | {format_tokens(int(max_context))} tokens |
| Total Duplication Found | {format_tokens(int(total_duplication))} tokens |

### Top Recommendations
"""

        if top_recs:
            for i, rec in enumerate(top_recs, 1):
                priority = rec.get("priority", "MEDIUM")
                title = rec.get("title", "Unknown")
                savings = rec.get("token_savings", 0)
                summary += f"{i}. {priority_emoji(priority)} **{title}** - Save {format_tokens(savings)} tokens\n"
        else:
            summary += "\n*No recommendations generated*"

        return summary

    def _generate_scenario_analysis(self) -> str:
        """Generate per-scenario analysis section."""
        scenarios = self.data.get("scenarios", [])

        if not scenarios:
            return "## Per-Scenario Analysis\n\n*No scenarios analyzed*"

        section = "## Per-Scenario Analysis\n"

        for scenario in scenarios:
            name = scenario.get("name", "Unknown")
            section += f"\n### Scenario: {name}\n\n"

            # Metrics table
            section += "**Metrics**:\n\n"
            section += "| Metric | Value |\n"
            section += "|--------|-------|\n"
            section += f"| Total Rounds | {scenario.get('total_rounds', 0)} |\n"
            section += f"| Avg Context Size | {format_tokens(scenario.get('avg_context_size', 0))} |\n"
            section += f"| Max Context Size | {format_tokens(scenario.get('max_context_size', 0))} |\n"
            section += f"| Avg System Prompt | {format_tokens(scenario.get('avg_system_prompt_size', 0))} |\n"
            section += f"| Avg Tool Definitions | {format_tokens(scenario.get('avg_tool_size', 0))} |\n"
            section += f"| Duplication Found | {format_tokens(scenario.get('duplication', {}).get('total_duplicated_tokens', 0))} |\n"
            section += "\n"

            # Growth chart
            growth = scenario.get("growth", {})
            context_sizes = growth.get("context_sizes", [])
            if context_sizes:
                section += "**Context Growth**:\n\n"
                section += "```\n"
                section += generate_ascii_chart([float(s) for s in context_sizes], width=60, height=8)
                section += "\n```\n\n"

                growth_rate = growth.get("growth_rate", "unknown")
                section += f"Growth Rate: **{growth_rate}**\n\n"

            # Behavior contribution
            behaviors = scenario.get("behaviors", {})
            if behaviors:
                section += "**Top Behaviors by Token Usage**:\n\n"
                # Sort by tokens
                sorted_behaviors = sorted(behaviors.items(), key=lambda x: x[1].get("total_tokens", 0), reverse=True)[:5]
                labels = [name for name, _ in sorted_behaviors]
                values = [float(data.get("total_tokens", 0)) for _, data in sorted_behaviors]
                section += "```\n"
                section += generate_bar_chart(labels, values, width=30)
                section += "\n```\n"

        return section

    def _generate_duplication_deep_dive(self) -> str:
        """Generate duplication analysis section."""
        section = "## Duplication Deep Dive\n\n"

        duplicates = self.data.get("duplication", {})
        if not duplicates:
            return section + "*No duplication data available*"

        exact = duplicates.get("exact", [])
        fuzzy = duplicates.get("fuzzy", [])

        if exact:
            section += "### Exact Duplicates\n\n"
            section += "| Content (truncated) | Count | Locations | Token Waste |\n"
            section += "|---------------------|-------|-----------|-------------|\n"

            for dup in exact[:10]:  # Top 10
                content = dup.get("content", "")[:50].replace("\n", " ")
                count = dup.get("count", 0)
                locations = dup.get("locations", [])
                tokens = dup.get("token_waste", 0)
                loc_str = ", ".join([f"r{loc.get('round', '?')}" for loc in locations[:3]])
                if len(locations) > 3:
                    loc_str += f" +{len(locations)-3} more"
                section += f"| {content}... | {count} | {loc_str} | {format_tokens(tokens)} |\n"
            section += "\n"

        if fuzzy:
            section += "### Fuzzy Duplicates (>80% similarity)\n\n"
            section += "| Content (truncated) | Similarity | Locations | Token Waste |\n"
            section += "|---------------------|------------|-----------|-------------|\n"

            for dup in fuzzy[:10]:  # Top 10
                content = dup.get("content", "")[:50].replace("\n", " ")
                similarity = dup.get("similarity", 0) * 100
                locations = dup.get("locations", [])
                tokens = dup.get("token_waste", 0)
                loc_str = ", ".join([f"r{loc.get('round', '?')}" for loc in locations[:3]])
                if len(locations) > 3:
                    loc_str += f" +{len(locations)-3} more"
                section += f"| {content}... | {similarity:.0f}% | {loc_str} | {format_tokens(tokens)} |\n"
            section += "\n"

        if not exact and not fuzzy:
            section += "*No duplicates found*\n"

        return section

    def _generate_behavior_contribution(self) -> str:
        """Generate behavior contribution matrix."""
        section = "## Behavior Contribution Matrix\n\n"

        behaviors = self.data.get("behaviors", {})
        if not behaviors:
            return section + "*No behavior data available*"

        section += "| Behavior | System Prompt | Tool Defs | Context Inject | Total | ROI Score |\n"
        section += "|----------|---------------|-----------|----------------|-------|----------|\n"

        # Sort by total tokens descending
        sorted_behaviors = sorted(
            behaviors.items(),
            key=lambda x: x[1].get("total_tokens", 0),
            reverse=True
        )

        for name, data in sorted_behaviors:
            system = format_tokens(data.get("system_prompt_tokens", 0))
            tools = format_tokens(data.get("tool_definition_tokens", 0))
            context = format_tokens(data.get("context_injection_tokens", 0))
            total = format_tokens(data.get("total_tokens", 0))
            roi = data.get("roi_score", 0.0)
            roi_str = f"{roi:.2f}" if roi > 0 else "N/A"

            section += f"| {name} | {system} | {tools} | {context} | {total} | {roi_str} |\n"

        section += "\n**ROI Score**: Value provided / tokens consumed (higher is better)\n"

        return section

    def _generate_recommendations(self) -> str:
        """Generate prioritized recommendations section."""
        section = "## Prioritized Recommendations\n\n"

        recommendations = self.data.get("recommendations", [])
        if not recommendations:
            return section + "*No recommendations available*"

        # Group by priority
        by_priority = defaultdict(list)
        for rec in recommendations:
            priority = rec.get("priority", "MEDIUM")
            by_priority[priority].append(rec)

        for priority in ["HIGH", "MEDIUM", "LOW"]:
            recs = by_priority.get(priority, [])
            if not recs:
                continue

            section += f"### {priority_emoji(priority)} {priority} Priority\n\n"

            for i, rec in enumerate(recs, 1):
                title = rec.get("title", "Unknown")
                description = rec.get("description", "")
                savings = rec.get("token_savings", 0)
                difficulty = rec.get("difficulty", "Unknown")
                file_path = rec.get("file_path", "")
                line_number = rec.get("line_number", "")

                section += f"#### {i}. {title}\n\n"
                section += f"{description}\n\n"
                section += f"**Impact**: Save {format_tokens(savings)} tokens per run\n\n"
                section += f"**Difficulty**: {difficulty}\n\n"

                if file_path:
                    location = file_path
                    if line_number:
                        location += f":{line_number}"
                    section += f"**Location**: `{location}`\n\n"

                implementation = rec.get("implementation", "")
                if implementation:
                    section += f"**Implementation**:\n```python\n{implementation}\n```\n\n"

        return section

    def _generate_comparative_analysis(self) -> str:
        """Generate comparative analysis across scenarios."""
        section = "## Comparative Analysis\n\n"

        scenarios = self.data.get("scenarios", [])
        if len(scenarios) < 2:
            return section + "*Need at least 2 scenarios for comparison*"

        section += "### Context Size by Scenario\n\n"

        # Compare max context sizes
        labels = [s.get("name", "Unknown") for s in scenarios]
        values = [float(s.get("max_context_size", 0)) for s in scenarios]

        section += "```\n"
        section += generate_bar_chart(labels, values, width=40)
        section += "\n```\n\n"

        # Growth rate comparison
        section += "### Growth Rate Comparison\n\n"
        section += "| Scenario | Growth Rate | Final Context | Duplication % |\n"
        section += "|----------|-------------|---------------|---------------|\n"

        for scenario in scenarios:
            name = scenario.get("name", "Unknown")
            growth_rate = scenario.get("growth", {}).get("growth_rate", "unknown")
            final_context = scenario.get("max_context_size", 0)
            dup_tokens = scenario.get("duplication", {}).get("total_duplicated_tokens", 0)
            dup_pct = (dup_tokens / final_context * 100) if final_context > 0 else 0

            section += f"| {name} | {growth_rate} | {format_tokens(final_context)} | {dup_pct:.1f}% |\n"

        section += "\n"

        # Key insights
        section += "### Key Insights\n\n"

        # Check if complexity leads to exponential growth
        max_growth = max(s.get("max_context_size", 0) for s in scenarios)
        min_growth = min(s.get("max_context_size", 0) for s in scenarios)
        growth_ratio = max_growth / min_growth if min_growth > 0 else 0

        if growth_ratio > 3:
            section += f"- ⚠️ **High complexity multiplication**: {growth_ratio:.1f}x context growth from simple to complex scenarios\n"
        else:
            section += f"- ✓ **Linear scaling**: {growth_ratio:.1f}x context growth from simple to complex scenarios\n"

        # Check duplication trend
        avg_dup = sum(s.get("duplication", {}).get("total_duplicated_tokens", 0) for s in scenarios) / len(scenarios)
        if avg_dup > 5000:
            section += f"- ⚠️ **High duplication**: Average {format_tokens(int(avg_dup))} duplicated tokens per scenario\n"
        else:
            section += f"- ✓ **Low duplication**: Average {format_tokens(int(avg_dup))} duplicated tokens per scenario\n"

        return section

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return """---

## How to Use This Report

1. **Start with HIGH priority recommendations** - Highest impact, quickest wins
2. **Review behavior contribution matrix** - Identify low-ROI behaviors to optimize
3. **Investigate exact duplicates** - Often easy to fix with caching or deduplication
4. **Monitor growth rate** - Exponential growth needs immediate attention
5. **Compare scenarios** - Understand how complexity affects context usage

## Next Steps

1. Implement HIGH priority recommendations
2. Re-run context inspection to validate improvements
3. Set up monitoring for context size trends
4. Document optimization guidelines for future behaviors

---

*Generated by Jetbox Context Inspector*"""


def generate_sample_report() -> str:
    """Generate a sample report with mock data for testing."""
    sample_data = {
        "scenarios": [
            {
                "name": "simple",
                "total_rounds": 8,
                "avg_context_size": 15000,
                "max_context_size": 20000,
                "avg_system_prompt_size": 8000,
                "avg_tool_size": 3000,
                "duplication": {
                    "total_duplicated_tokens": 1200
                },
                "growth": {
                    "context_sizes": [10000, 12000, 14000, 16000, 17000, 18000, 19000, 20000],
                    "growth_rate": "linear"
                },
                "behaviors": {
                    "FileToolsBehavior": {"total_tokens": 5000, "roi_score": 0.85},
                    "LoopDetectionBehavior": {"total_tokens": 2000, "roi_score": 0.92},
                    "HierarchicalContextBehavior": {"total_tokens": 3500, "roi_score": 0.78}
                }
            },
            {
                "name": "medium",
                "total_rounds": 18,
                "avg_context_size": 35000,
                "max_context_size": 50000,
                "avg_system_prompt_size": 12000,
                "avg_tool_size": 5000,
                "duplication": {
                    "total_duplicated_tokens": 4500
                },
                "growth": {
                    "context_sizes": [15000, 20000, 25000, 30000, 35000, 38000, 42000, 45000, 47000, 48000, 49000, 50000],
                    "growth_rate": "linear"
                },
                "behaviors": {
                    "FileToolsBehavior": {"total_tokens": 8000, "roi_score": 0.85},
                    "CommandToolsBehavior": {"total_tokens": 4000, "roi_score": 0.80},
                    "LoopDetectionBehavior": {"total_tokens": 2500, "roi_score": 0.92},
                    "HierarchicalContextBehavior": {"total_tokens": 6000, "roi_score": 0.78},
                    "WorkspaceTaskNotesBehavior": {"total_tokens": 3000, "roi_score": 0.88}
                }
            },
            {
                "name": "complex",
                "total_rounds": 35,
                "avg_context_size": 75000,
                "max_context_size": 110000,
                "avg_system_prompt_size": 18000,
                "avg_tool_size": 8000,
                "duplication": {
                    "total_duplicated_tokens": 12000
                },
                "growth": {
                    "context_sizes": [20000, 30000, 40000, 50000, 60000, 70000, 78000, 85000, 90000, 95000, 100000, 105000, 110000],
                    "growth_rate": "slightly exponential"
                },
                "behaviors": {
                    "FileToolsBehavior": {"total_tokens": 12000, "roi_score": 0.85},
                    "CommandToolsBehavior": {"total_tokens": 6000, "roi_score": 0.80},
                    "DelegationBehavior": {"total_tokens": 15000, "roi_score": 0.65},
                    "LoopDetectionBehavior": {"total_tokens": 3000, "roi_score": 0.92},
                    "HierarchicalContextBehavior": {"total_tokens": 10000, "roi_score": 0.78},
                    "WorkspaceTaskNotesBehavior": {"total_tokens": 5000, "roi_score": 0.88},
                    "StatusDisplayBehavior": {"total_tokens": 7000, "roi_score": 0.72}
                }
            }
        ],
        "duplication": {
            "exact": [
                {
                    "content": "def write_file(path: str, content: str) -> dict:",
                    "count": 8,
                    "locations": [{"round": 2}, {"round": 4}, {"round": 6}],
                    "token_waste": 150
                },
                {
                    "content": "Tool definitions are loaded from behaviors",
                    "count": 12,
                    "locations": [{"round": 1}, {"round": 3}, {"round": 5}],
                    "token_waste": 240
                }
            ],
            "fuzzy": [
                {
                    "content": "The agent should complete the current task",
                    "similarity": 0.87,
                    "locations": [{"round": 5}, {"round": 8}],
                    "token_waste": 95
                }
            ]
        },
        "behaviors": {
            "FileToolsBehavior": {
                "system_prompt_tokens": 800,
                "tool_definition_tokens": 3200,
                "context_injection_tokens": 1000,
                "total_tokens": 5000,
                "roi_score": 0.85
            },
            "CommandToolsBehavior": {
                "system_prompt_tokens": 500,
                "tool_definition_tokens": 2500,
                "context_injection_tokens": 1000,
                "total_tokens": 4000,
                "roi_score": 0.80
            },
            "DelegationBehavior": {
                "system_prompt_tokens": 2000,
                "tool_definition_tokens": 8000,
                "context_injection_tokens": 5000,
                "total_tokens": 15000,
                "roi_score": 0.65
            },
            "LoopDetectionBehavior": {
                "system_prompt_tokens": 400,
                "tool_definition_tokens": 800,
                "context_injection_tokens": 800,
                "total_tokens": 2000,
                "roi_score": 0.92
            },
            "HierarchicalContextBehavior": {
                "system_prompt_tokens": 1500,
                "tool_definition_tokens": 0,
                "context_injection_tokens": 2000,
                "total_tokens": 3500,
                "roi_score": 0.78
            }
        },
        "recommendations": [
            {
                "priority": "HIGH",
                "title": "Cache tool definitions across rounds",
                "description": "Tool definitions are being serialized in every context snapshot. Implement caching to reference definitions by hash instead of including full text.",
                "token_savings": 8000,
                "difficulty": "Medium",
                "file_path": "base_agent.py",
                "line_number": "425",
                "implementation": "# Add tool definition cache\nself._tool_def_cache = {}\n\ndef _get_tool_defs(self):\n    cache_key = hash(tuple(sorted(self.tools)))\n    if cache_key not in self._tool_def_cache:\n        self._tool_def_cache[cache_key] = self._serialize_tools()\n    return self._tool_def_cache[cache_key]"
            },
            {
                "priority": "HIGH",
                "title": "Deduplicate system prompt instructions",
                "description": "Multiple behaviors inject similar instructions into the system prompt. Create a shared instruction registry to avoid duplication.",
                "token_savings": 3500,
                "difficulty": "High",
                "file_path": "base_agent.py",
                "line_number": "280"
            },
            {
                "priority": "MEDIUM",
                "title": "Compress historical context",
                "description": "Message history grows linearly. Implement rolling summarization for messages older than 5 rounds.",
                "token_savings": 5000,
                "difficulty": "Medium",
                "file_path": "behaviors/hierarchical_context.py",
                "line_number": "156"
            },
            {
                "priority": "MEDIUM",
                "title": "Optimize DelegationBehavior tool definitions",
                "description": "DelegationBehavior has the lowest ROI (0.65). Tool definitions are verbose. Simplify descriptions and parameter schemas.",
                "token_savings": 4000,
                "difficulty": "Low",
                "file_path": "behaviors/delegation.py",
                "line_number": "78"
            },
            {
                "priority": "LOW",
                "title": "Implement context pruning strategy",
                "description": "Automatically remove low-value context elements when approaching token limits (e.g., old error messages, redundant status updates).",
                "token_savings": 2000,
                "difficulty": "High",
                "file_path": "base_agent.py",
                "line_number": "520"
            }
        ]
    }

    report = ContextInspectionReport(sample_data)
    return report.generate()


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) > 1:
        # Load from file
        input_file = Path(sys.argv[1])
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            sys.exit(1)

        with open(input_file) as f:
            data = json.load(f)

        output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("context_inspection_report.md")
    else:
        # Generate sample report
        print("No input file provided. Generating sample report...")
        data = None
        output_file = Path("sample_context_inspection_report.md")

    # Generate report
    if data:
        report = ContextInspectionReport(data)
        content = report.generate()
    else:
        content = generate_sample_report()

    # Save report
    output_file.write_text(content)
    print(f"Report generated: {output_file}")
    print(f"Size: {len(content)} bytes")


if __name__ == "__main__":
    main()
