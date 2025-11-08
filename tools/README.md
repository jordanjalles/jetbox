# Context Inspection Tools

Tools for analyzing and reporting on Jetbox agent context window usage.

## Overview

The Context Inspection System helps identify inefficiencies, duplication, and optimization opportunities in agent context windows.

## Quick Start

```bash
# Analyze captured snapshots
python tools/analyze_context.py .context_inspection/test_data

# Run interactive demo
python tools/demo_phase3.py

# Run tests
pytest tests/test_analyze_context.py -v
```

## Tools

### analyze_context.py (Phase 3 - NEW!)

**Comprehensive analysis engine for context snapshots.**

**Features:**
- Duplication detection (exact and fuzzy >80% similarity)
- Growth pattern analysis (linear/exponential/stable)
- Per-behavior token attribution
- Prioritized recommendations (HIGH/MEDIUM/LOW)
- Markdown report generation

**Usage:**
```bash
python tools/analyze_context.py .context_inspection
python tools/analyze_context.py .context_inspection --output report.md
```

See `PHASE3_USAGE_EXAMPLES.md` in docs/context_inspection/ for details.

### demo_phase3.py (NEW!)

**Interactive demo showing all analysis capabilities.**

```bash
python tools/demo_phase3.py
```

### report_generator.py (Phase 5)

Generate comprehensive, actionable markdown reports from context inspection data.

**Usage:**

```bash
# Generate sample report (for testing/demo)
python tools/report_generator.py

# Generate report from analysis data
python tools/report_generator.py input_data.json output_report.md
```

**Input Format:**

The tool expects JSON data with this structure:

```json
{
  "scenarios": [
    {
      "name": "scenario_name",
      "total_rounds": 10,
      "avg_context_size": 25000,
      "max_context_size": 35000,
      "avg_system_prompt_size": 10000,
      "avg_tool_size": 5000,
      "duplication": {
        "total_duplicated_tokens": 2000
      },
      "growth": {
        "context_sizes": [15000, 20000, 25000, 30000, 35000],
        "growth_rate": "linear"
      },
      "behaviors": {
        "BehaviorName": {
          "total_tokens": 5000,
          "roi_score": 0.85
        }
      }
    }
  ],
  "duplication": {
    "exact": [
      {
        "content": "duplicated text...",
        "count": 5,
        "locations": [{"round": 2}, {"round": 4}],
        "token_waste": 150
      }
    ],
    "fuzzy": [
      {
        "content": "similar text...",
        "similarity": 0.85,
        "locations": [{"round": 3}],
        "token_waste": 100
      }
    ]
  },
  "behaviors": {
    "BehaviorName": {
      "system_prompt_tokens": 800,
      "tool_definition_tokens": 3000,
      "context_injection_tokens": 1200,
      "total_tokens": 5000,
      "roi_score": 0.85
    }
  },
  "recommendations": [
    {
      "priority": "HIGH",
      "title": "Recommendation title",
      "description": "Detailed description",
      "token_savings": 5000,
      "difficulty": "Medium",
      "file_path": "path/to/file.py",
      "line_number": "123",
      "implementation": "code snippet..."
    }
  ]
}
```

**Output:**

Generates a markdown report with:

1. **Executive Summary**
   - Key metrics across all scenarios
   - Top 3 recommendations

2. **Per-Scenario Analysis**
   - Metrics table
   - Context growth chart (ASCII art)
   - Behavior contribution chart

3. **Duplication Deep Dive**
   - Exact duplicates table
   - Fuzzy duplicates table
   - Token waste calculations

4. **Behavior Contribution Matrix**
   - Per-behavior token breakdown
   - ROI scores

5. **Prioritized Recommendations**
   - HIGH/MEDIUM/LOW priority (color-coded)
   - Impact (token savings)
   - Difficulty estimate
   - File paths and line numbers
   - Implementation code snippets

6. **Comparative Analysis**
   - Cross-scenario comparison
   - Growth rate analysis
   - Key insights

7. **Usage Instructions**
   - How to interpret the report
   - Next steps

## Report Features

### Visualizations

- **ASCII art line charts** - Context growth over rounds
- **Horizontal bar charts** - Behavior comparisons
- **Priority emojis** - 🔴 HIGH, 🟡 MEDIUM, 🟢 LOW
- **Token formatting** - Human-readable (5.0K instead of 5000)

### Actionable Content

- Specific file paths and line numbers
- Token savings estimates
- Implementation difficulty ratings
- Code snippets for fixes
- ROI scores for cost/benefit analysis

### Data Handling

- Gracefully handles missing data
- Supports both real analysis data and mock data
- Validates input structure
- Provides helpful error messages

## Testing

Run the test suite:

```bash
python tools/test_report_generator.py
```

Tests cover:
- ASCII chart generation
- Bar chart generation
- Token formatting
- Priority emoji selection
- Empty/missing data handling
- Real data structure processing
- Full report generation

## Example Output

See `/workspace/docs/context_inspection/sample_context_inspection_report.md` for a complete example report.

## Integration with Analysis Engine

The report generator is designed to work with Phase 3's analysis engine:

```bash
# Phase 3: Analyze captured context snapshots
python tools/analyze_context.py .context_inspection > analysis.json

# Phase 5: Generate report from analysis
python tools/report_generator.py analysis.json context_report.md
```

## Development

**Adding New Sections:**

1. Add method to `ContextInspectionReport` class:
   ```python
   def _generate_my_section(self) -> str:
       section = "## My New Section\n\n"
       # ... generate content
       return section
   ```

2. Add to `generate()` method's sections list:
   ```python
   sections = [
       self._generate_header(),
       # ... other sections
       self._generate_my_section(),
       self._generate_footer(),
   ]
   ```

**Adding New Visualizations:**

Create helper functions similar to `generate_ascii_chart()` and `generate_bar_chart()`.

**Customizing Output:**

All formatting functions (token formatting, emojis, etc.) are in the module globals for easy customization.

## Future Enhancements

- [ ] HTML report generation
- [ ] Interactive charts (if rendered in notebook)
- [ ] CSV export for metrics
- [ ] Trend comparison across multiple runs
- [ ] Auto-recommendations based on patterns
- [ ] Integration with CI/CD for regression detection

## License

Part of the Jetbox project.
