#!/usr/bin/env python3
"""
Integration test for Phase 3 (analyze_context.py) + Phase 5 (report_generator.py)

Tests the complete pipeline:
1. Create sample snapshot data
2. Run analysis engine
3. Generate report
4. Validate report content
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime


def create_sample_snapshots(snapshot_dir: Path) -> None:
    """Create sample snapshot files for testing."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Sample snapshots with increasing context size
    snapshots = [
        {
            "agent_name": "task_executor",
            "round": 0,
            "phase": "initial",
            "timestamp": datetime.now().timestamp(),
            "context": [
                {"role": "system", "content": "You are a helpful agent. " * 100},
                {"role": "user", "content": "Create a calculator function"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write content to a file. " * 10,
                        "parameters": {"type": "object", "properties": {}}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read content from a file. " * 10,
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ],
            "behaviors_loaded": ["FileToolsBehavior", "LoopDetectionBehavior"],
            "metrics": {
                "system_prompt_length": 3000,
                "total_messages": 2,
                "total_context_length": 8000,
                "tool_count": 2,
                "tool_definition_length": 2000
            }
        },
        {
            "agent_name": "task_executor",
            "round": 1,
            "phase": "pre_llm",
            "timestamp": datetime.now().timestamp(),
            "context": [
                {"role": "system", "content": "You are a helpful agent. " * 100},
                {"role": "user", "content": "Create a calculator function"},
                {"role": "assistant", "content": "I'll create the calculator. " * 50}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write content to a file. " * 10,  # Duplicate
                        "parameters": {"type": "object", "properties": {}}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read content from a file. " * 10,  # Duplicate
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ],
            "behaviors_loaded": ["FileToolsBehavior", "LoopDetectionBehavior"],
            "metrics": {
                "system_prompt_length": 3000,
                "total_messages": 3,
                "total_context_length": 12000,
                "tool_count": 2,
                "tool_definition_length": 2000
            }
        },
        {
            "agent_name": "task_executor",
            "round": 2,
            "phase": "pre_llm",
            "timestamp": datetime.now().timestamp(),
            "context": [
                {"role": "system", "content": "You are a helpful agent. " * 100},
                {"role": "user", "content": "Create a calculator function"},
                {"role": "assistant", "content": "I'll create the calculator. " * 50},
                {"role": "user", "content": "Continue with the implementation"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write content to a file. " * 10,  # Duplicate
                        "parameters": {"type": "object", "properties": {}}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read content from a file. " * 10,  # Duplicate
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ],
            "behaviors_loaded": ["FileToolsBehavior", "LoopDetectionBehavior"],
            "metrics": {
                "system_prompt_length": 3000,
                "total_messages": 4,
                "total_context_length": 15000,
                "tool_count": 2,
                "tool_definition_length": 2000
            }
        }
    ]

    for snapshot in snapshots:
        filename = f"{snapshot['agent_name']}_round_{snapshot['round']:03d}_{snapshot['phase']}.json"
        filepath = snapshot_dir / filename
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)


def test_full_pipeline():
    """Test the complete Phase 3 → Phase 5 pipeline."""
    print("\n=== Testing Full Integration Pipeline ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_dir = Path(tmpdir) / "snapshots"

        # Step 1: Create sample snapshots
        print("Step 1: Creating sample snapshot data...")
        create_sample_snapshots(snapshot_dir)
        snapshot_files = list(snapshot_dir.glob("*.json"))
        print(f"  Created {len(snapshot_files)} snapshot files")
        assert len(snapshot_files) == 3, "Should create 3 snapshots"

        # Step 2: Run analysis engine
        print("\nStep 2: Running analysis engine...")
        from analyze_context import (
            load_snapshots,
            analyze_duplication,
            analyze_growth,
            attribute_tokens_to_behaviors,
            generate_recommendations
        )

        snapshots = load_snapshots(snapshot_dir)
        print(f"  Loaded {len(snapshots)} snapshots")
        assert len(snapshots) == 3, "Should load 3 snapshots"

        duplication_data = analyze_duplication(snapshots)
        print(f"  Found {len(duplication_data.get('exact', []))} exact duplicates")

        growth_data = analyze_growth(snapshots)
        print(f"  Growth rate: {growth_data.get('growth_rate', 'unknown')}")

        behavior_data = attribute_tokens_to_behaviors(snapshots)
        print(f"  Analyzed {len(behavior_data)} behaviors")

        recommendations = generate_recommendations({
            "duplication": duplication_data,
            "growth": growth_data,
            "behaviors": behavior_data
        })
        print(f"  Generated {len(recommendations)} recommendations")

        # Step 3: Generate report
        print("\nStep 3: Generating report...")
        from report_generator import ContextInspectionReport

        analysis_data = {
            "scenarios": [
                {
                    "name": "test_scenario",
                    "total_rounds": len(snapshots),
                    "avg_context_size": sum(s["metrics"]["total_context_length"] for s in snapshots) / len(snapshots),
                    "max_context_size": max(s["metrics"]["total_context_length"] for s in snapshots),
                    "avg_system_prompt_size": sum(s["metrics"]["system_prompt_length"] for s in snapshots) / len(snapshots),
                    "avg_tool_size": sum(s["metrics"]["tool_definition_length"] for s in snapshots) / len(snapshots),
                    "duplication": duplication_data,
                    "growth": growth_data,
                    "behaviors": behavior_data
                }
            ],
            "duplication": duplication_data,
            "behaviors": behavior_data,
            "recommendations": recommendations
        }

        report = ContextInspectionReport(analysis_data)
        report_content = report.generate()

        print(f"  Generated report: {len(report_content)} bytes")

        # Step 4: Validate report content
        print("\nStep 4: Validating report content...")

        required_sections = [
            "# Context Inspection Report",
            "## Executive Summary",
            "## Per-Scenario Analysis",
            "## Duplication Deep Dive",
            "## Behavior Contribution Matrix",
            "## Prioritized Recommendations",
            "## Comparative Analysis"
        ]

        for section in required_sections:
            assert section in report_content, f"Missing section: {section}"
            print(f"  ✓ Found section: {section}")

        # Validate data appeared in report
        assert "test_scenario" in report_content, "Scenario name should appear"
        assert "FileToolsBehavior" in report_content, "Behavior name should appear"

        # Validate visualizations
        assert '█' in report_content, "Should contain charts"
        assert '|' in report_content, "Should contain tables"

        # Save report for inspection
        output_file = Path(tmpdir) / "integration_test_report.md"
        output_file.write_text(report_content)
        print(f"\n  Report saved to: {output_file}")

        print("\n✓ Full pipeline integration test passed!")
        print(f"  Pipeline: Snapshots → Analysis → Report ({len(report_content)} bytes)")

        return True


def main():
    """Run integration test."""
    try:
        test_full_pipeline()
        print("\n=== Integration Test PASSED ===\n")
        return 0
    except Exception as e:
        print(f"\n✗ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
