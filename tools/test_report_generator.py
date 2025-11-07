#!/usr/bin/env python3
"""
Test suite for report_generator.py
"""

import json
from pathlib import Path
from report_generator import (
    ContextInspectionReport,
    generate_ascii_chart,
    generate_bar_chart,
    format_tokens,
    priority_emoji,
    generate_sample_report
)


def test_ascii_chart():
    """Test ASCII chart generation."""
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    chart = generate_ascii_chart(values, width=40, height=8)
    assert len(chart.split('\n')) == 8, "Chart should have 8 rows"
    assert '█' in chart, "Chart should contain block characters"
    print("✓ ASCII chart generation works")


def test_bar_chart():
    """Test bar chart generation."""
    labels = ["Behavior A", "Behavior B", "Behavior C"]
    values = [100, 200, 150]
    chart = generate_bar_chart(labels, values, width=30)
    assert "Behavior A" in chart, "Chart should contain labels"
    assert '█' in chart, "Chart should contain bars"
    print("✓ Bar chart generation works")


def test_token_formatting():
    """Test token number formatting."""
    assert format_tokens(500) == "500", "Small numbers should not use K"
    assert format_tokens(5000) == "5.0K", "5000 should be 5.0K"
    assert format_tokens(15234) == "15.2K", "15234 should be 15.2K"
    print("✓ Token formatting works")


def test_priority_emojis():
    """Test priority emoji selection."""
    assert priority_emoji("HIGH") == "🔴"
    assert priority_emoji("MEDIUM") == "🟡"
    assert priority_emoji("LOW") == "🟢"
    print("✓ Priority emoji selection works")


def test_sample_report_generation():
    """Test full sample report generation."""
    report_content = generate_sample_report()

    # Check all required sections exist
    required_sections = [
        "# Context Inspection Report",
        "## Executive Summary",
        "## Per-Scenario Analysis",
        "## Duplication Deep Dive",
        "## Behavior Contribution Matrix",
        "## Prioritized Recommendations",
        "## Comparative Analysis",
        "## How to Use This Report"
    ]

    for section in required_sections:
        assert section in report_content, f"Report should contain '{section}'"

    # Check visualizations
    assert '█' in report_content, "Report should contain charts"
    assert '🔴' in report_content, "Report should contain priority emojis"

    # Check tables
    assert '|' in report_content, "Report should contain markdown tables"
    assert '---' in report_content, "Report should contain table separators"

    print("✓ Sample report generation works")
    print(f"  Report size: {len(report_content)} bytes")


def test_empty_data_handling():
    """Test handling of missing/empty data."""
    empty_data = {
        "scenarios": [],
        "duplication": {},
        "behaviors": {},
        "recommendations": []
    }

    report = ContextInspectionReport(empty_data)
    content = report.generate()

    assert "No scenarios analyzed" in content or "No data" in content
    print("✓ Empty data handling works")


def test_report_with_real_structure():
    """Test report with realistic data structure."""
    data = {
        "scenarios": [
            {
                "name": "test",
                "total_rounds": 5,
                "avg_context_size": 10000,
                "max_context_size": 15000,
                "avg_system_prompt_size": 5000,
                "avg_tool_size": 2000,
                "duplication": {"total_duplicated_tokens": 500},
                "growth": {
                    "context_sizes": [8000, 10000, 12000, 14000, 15000],
                    "growth_rate": "linear"
                },
                "behaviors": {
                    "TestBehavior": {"total_tokens": 3000, "roi_score": 0.8}
                }
            }
        ],
        "recommendations": [
            {
                "priority": "HIGH",
                "title": "Test recommendation",
                "description": "This is a test",
                "token_savings": 1000,
                "difficulty": "Easy"
            }
        ]
    }

    report = ContextInspectionReport(data)
    content = report.generate()

    assert "test" in content.lower()
    assert "HIGH" in content or "🔴" in content
    assert "Test recommendation" in content
    print("✓ Real structure report generation works")


def main():
    """Run all tests."""
    print("\n=== Testing Report Generator ===\n")

    tests = [
        test_ascii_chart,
        test_bar_chart,
        test_token_formatting,
        test_priority_emojis,
        test_empty_data_handling,
        test_report_with_real_structure,
        test_sample_report_generation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1

    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
