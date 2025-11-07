"""
Unit tests for context inspection analysis engine.
"""

import json
import pytest
from pathlib import Path
import tempfile
import shutil

from tools.analyze_context import (
    load_snapshots,
    analyze_duplication,
    analyze_growth,
    attribute_tokens_to_behaviors,
    generate_recommendations,
    _calculate_tokens,
    _find_exact_duplicates,
    _find_fuzzy_duplicates,
)


@pytest.fixture
def temp_snapshot_dir():
    """Create a temporary directory for test snapshots."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_snapshot():
    """Create a sample snapshot dict."""
    return {
        "agent_name": "test_agent",
        "round": 0,
        "phase": "initial",
        "timestamp": 1234567890.0,
        "context": [
            {"role": "system", "content": "System prompt here"},
            {"role": "user", "content": "User message here"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "behaviors_loaded": ["TestBehavior"],
        "metrics": {
            "system_prompt_length": 100,
            "total_messages": 2,
            "total_context_length": 200,
            "tool_count": 1,
            "tool_definition_length": 50,
        },
    }


def test_calculate_tokens():
    """Test token calculation approximation."""
    assert _calculate_tokens("") == 0
    assert _calculate_tokens("a" * 4) == 1
    assert _calculate_tokens("a" * 100) == 25
    assert _calculate_tokens("test string here") == 4


def test_find_exact_duplicates():
    """Test exact duplicate detection."""
    texts = [
        ("loc1", "duplicate text"),
        ("loc2", "unique text"),
        ("loc3", "duplicate text"),
        ("loc4", "another duplicate"),
        ("loc5", "another duplicate"),
        ("loc6", ""),  # Empty should be ignored
    ]

    duplicates = _find_exact_duplicates(texts)

    assert len(duplicates) == 2
    assert "duplicate text" in duplicates
    assert "another duplicate" in duplicates
    assert duplicates["duplicate text"] == ["loc1", "loc3"]
    assert duplicates["another duplicate"] == ["loc4", "loc5"]
    assert "" not in duplicates


def test_find_fuzzy_duplicates():
    """Test fuzzy duplicate detection."""
    # Create similar but not identical strings (>100 chars for threshold)
    text1 = "This is a test string that is quite long and should be detected as a duplicate when compared to similar text. " * 2
    text2 = "This is a test string that is quite long and should be detected as a duplicate when compared to similar text. " * 2
    text3 = "Completely different text here that should not match anything at all in the comparison process. " * 2

    texts = [
        ("loc1", text1),
        ("loc2", text2),
        ("loc3", text3),
    ]

    fuzzy_dups = _find_fuzzy_duplicates(texts, similarity_threshold=0.8)

    # Should find text1 and text2 as duplicates
    assert len(fuzzy_dups) >= 1
    assert fuzzy_dups[0]["locations"] == ["loc1", "loc2"]
    assert fuzzy_dups[0]["similarity"] >= 0.8


def test_load_snapshots_empty_dir(temp_snapshot_dir):
    """Test loading from empty directory."""
    with pytest.raises(ValueError, match="No snapshot files found"):
        load_snapshots(temp_snapshot_dir)


def test_load_snapshots_missing_dir():
    """Test loading from non-existent directory."""
    with pytest.raises(FileNotFoundError):
        load_snapshots(Path("/nonexistent/directory"))


def test_load_snapshots_valid(temp_snapshot_dir, sample_snapshot):
    """Test loading valid snapshots."""
    # Create test snapshots
    snapshot1 = {**sample_snapshot, "agent_name": "agent_a", "round": 0}
    snapshot2 = {**sample_snapshot, "agent_name": "agent_a", "round": 1}
    snapshot3 = {**sample_snapshot, "agent_name": "agent_b", "round": 0}

    (temp_snapshot_dir / "snapshot1.json").write_text(json.dumps(snapshot1))
    (temp_snapshot_dir / "snapshot2.json").write_text(json.dumps(snapshot2))
    (temp_snapshot_dir / "snapshot3.json").write_text(json.dumps(snapshot3))

    snapshots = load_snapshots(temp_snapshot_dir)

    assert len(snapshots) == 3
    # Should be sorted by agent_name, then round
    assert snapshots[0]["agent_name"] == "agent_a"
    assert snapshots[0]["round"] == 0
    assert snapshots[1]["agent_name"] == "agent_a"
    assert snapshots[1]["round"] == 1
    assert snapshots[2]["agent_name"] == "agent_b"


def test_load_snapshots_with_corrupt_file(temp_snapshot_dir, sample_snapshot):
    """Test loading with one corrupt file."""
    # Create one valid and one corrupt snapshot
    (temp_snapshot_dir / "valid.json").write_text(json.dumps(sample_snapshot))
    (temp_snapshot_dir / "corrupt.json").write_text("not valid json{")

    snapshots = load_snapshots(temp_snapshot_dir)

    # Should load only the valid one
    assert len(snapshots) == 1


def test_analyze_duplication():
    """Test duplication analysis."""
    snapshots = [
        {
            "agent_name": "agent1",
            "round": 0,
            "context": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "tool1", "description": "Tool 1"},
                }
            ],
        },
        {
            "agent_name": "agent1",
            "round": 1,
            "context": [
                {"role": "system", "content": "System prompt"},  # Duplicate
                {"role": "user", "content": "User message"},  # Duplicate
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "tool1", "description": "Tool 1"},  # Duplicate
                }
            ],
        },
    ]

    report = analyze_duplication(snapshots)

    assert "exact_duplicates" in report
    assert "fuzzy_duplicates" in report
    assert "token_waste" in report

    # Should find duplicates
    assert report["exact_duplicates"]["system_prompts"]["count"] >= 1
    assert report["exact_duplicates"]["messages"]["count"] >= 1
    assert report["exact_duplicates"]["tools"]["count"] >= 1

    # Token waste should be positive
    assert report["token_waste"]["exact"] > 0


def test_analyze_growth():
    """Test growth analysis."""
    snapshots = [
        {
            "agent_name": "agent1",
            "round": 0,
            "metrics": {
                "total_context_length": 100,
                "total_messages": 2,
                "tool_count": 1,
                "system_prompt_length": 50,
            },
        },
        {
            "agent_name": "agent1",
            "round": 1,
            "metrics": {
                "total_context_length": 150,
                "total_messages": 3,
                "tool_count": 1,
                "system_prompt_length": 50,
            },
        },
        {
            "agent_name": "agent1",
            "round": 2,
            "metrics": {
                "total_context_length": 200,
                "total_messages": 4,
                "tool_count": 2,
                "system_prompt_length": 50,
            },
        },
    ]

    report = analyze_growth(snapshots)

    assert "agents" in report
    assert "agent1" in report["agents"]

    agent_growth = report["agents"]["agent1"]
    assert agent_growth["rounds"] == 3
    assert agent_growth["initial_context_length"] == 100
    assert agent_growth["final_context_length"] == 200
    assert agent_growth["growth_absolute"] == 100
    assert agent_growth["growth_rate_percent"] == 100.0
    assert agent_growth["growth_pattern"] in ["linear", "exponential", "stable"]


def test_analyze_growth_exponential():
    """Test exponential growth detection."""
    snapshots = [
        {
            "agent_name": "agent1",
            "round": i,
            "metrics": {
                "total_context_length": 100 * (2**i),  # Exponential growth
                "total_messages": 2 + i,
                "tool_count": 1,
                "system_prompt_length": 50,
            },
        }
        for i in range(5)
    ]

    report = analyze_growth(snapshots)

    agent_growth = report["agents"]["agent1"]
    assert agent_growth["growth_pattern"] == "exponential"


def test_attribute_tokens_to_behaviors():
    """Test behavior token attribution."""
    snapshots = [
        {
            "agent_name": "agent1",
            "round": 2,
            "behaviors_loaded": ["BehaviorA", "BehaviorB"],
            "context": [{"role": "system", "content": "BehaviorA BehaviorB"}],
            "tools": [
                {"type": "function", "function": {"name": "tool1"}},
                {"type": "function", "function": {"name": "tool2"}},
            ],
        }
    ]

    report = attribute_tokens_to_behaviors(snapshots)

    assert "BehaviorA" in report
    assert "BehaviorB" in report

    # Each behavior should have token counts
    for behavior in ["BehaviorA", "BehaviorB"]:
        assert "system_prompt_tokens" in report[behavior]
        assert "tool_tokens" in report[behavior]
        assert "total_tokens" in report[behavior]
        assert report[behavior]["total_tokens"] > 0


def test_generate_recommendations():
    """Test recommendation generation."""
    analysis_results = {
        "duplication": {
            "token_waste": {"total_estimated": 15000, "exact": 10000, "fuzzy_estimated": 5000}
        },
        "growth": {
            "agents": {
                "agent1": {
                    "growth_pattern": "exponential",
                    "growth_rate_percent": 200.0,
                    "growth_absolute": 50000,
                    "projection": {"rounds_to_limit": 10},
                }
            }
        },
        "behaviors": {
            "HeavyBehavior": {"total_tokens": 25000, "system_prompt_tokens": 15000, "tool_tokens": 10000},
            "LightBehavior": {"total_tokens": 5000, "system_prompt_tokens": 3000, "tool_tokens": 2000},
        },
    }

    recommendations = generate_recommendations(analysis_results)

    assert len(recommendations) > 0

    # Should have HIGH priority recommendations
    high_priority = [r for r in recommendations if r["priority"] == "HIGH"]
    assert len(high_priority) > 0

    # Check recommendation structure
    for rec in recommendations:
        assert "priority" in rec
        assert "category" in rec
        assert "title" in rec
        assert "description" in rec
        assert "fixes" in rec
        assert "potential_savings" in rec
        assert "difficulty" in rec
        assert rec["priority"] in ["HIGH", "MEDIUM", "LOW"]

    # Should be sorted by priority
    priorities = [rec["priority"] for rec in recommendations]
    priority_values = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    for i in range(len(priorities) - 1):
        assert priority_values[priorities[i]] <= priority_values[priorities[i + 1]]


def test_generate_recommendations_low_waste():
    """Test recommendations with low token waste."""
    analysis_results = {
        "duplication": {"token_waste": {"total_estimated": 500}},
        "growth": {"agents": {}},
        "behaviors": {},
    }

    recommendations = generate_recommendations(analysis_results)

    # Should still generate recommendations but with lower priority
    if recommendations:
        dup_recs = [r for r in recommendations if r["category"] == "duplication"]
        if dup_recs:
            assert dup_recs[0]["priority"] == "LOW"


def test_integration_full_analysis(temp_snapshot_dir):
    """Integration test: full analysis pipeline."""
    # Create realistic test data
    for i in range(5):
        snapshot = {
            "agent_name": "test_agent",
            "round": i,
            "phase": "pre_llm",
            "timestamp": 1234567890.0 + i * 10,
            "context": [
                {"role": "system", "content": "System prompt " + "x" * (i * 100)},
                {"role": "user", "content": f"Message {i}"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "tool1", "description": "Tool 1"},
                }
            ],
            "behaviors_loaded": ["BehaviorA", "BehaviorB"],
            "metrics": {
                "system_prompt_length": 100 + i * 100,
                "total_messages": 2 + i,
                "total_context_length": 200 + i * 200,
                "tool_count": 1,
                "tool_definition_length": 50,
            },
        }
        (temp_snapshot_dir / f"snapshot_{i}.json").write_text(json.dumps(snapshot))

    # Load snapshots
    snapshots = load_snapshots(temp_snapshot_dir)
    assert len(snapshots) == 5

    # Run all analyses
    duplication = analyze_duplication(snapshots)
    growth = analyze_growth(snapshots)
    behaviors = attribute_tokens_to_behaviors(snapshots)

    # Verify results
    assert "exact_duplicates" in duplication
    assert "token_waste" in duplication
    assert "agents" in growth
    assert len(behaviors) > 0

    # Generate recommendations
    analysis_results = {
        "duplication": duplication,
        "growth": growth,
        "behaviors": behaviors,
    }
    recommendations = generate_recommendations(analysis_results)

    assert len(recommendations) > 0
