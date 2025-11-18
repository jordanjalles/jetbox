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

---

# Code Analysis Tools

This section contains code analysis tools to help maintain code quality and identify refactoring opportunities in the Jetbox codebase.

## Tools Overview

### detect_duplicates.py - Duplicate Code Detector

Finds duplicate code blocks across the codebase using AST-based comparison (ignores whitespace and comments).

**Features:**
- AST-based comparison (structural similarity)
- Configurable minimum block size (default: 5 lines)
- Ignore test files option
- Sort by duplication impact
- Console and markdown output formats
- Progress indicators for large scans

**Usage:**
```bash
# Scan behaviors directory with 5+ line minimum
python tools/detect_duplicates.py --min-lines 5 --path behaviors/

# Scan multiple directories, ignore tests, save to markdown
python tools/detect_duplicates.py --path src/ behaviors/ --ignore-tests --output duplicates.md

# Scan entire codebase with 10+ line minimum
python tools/detect_duplicates.py --min-lines 10

# Scan specific file
python tools/detect_duplicates.py --path behaviors/delegation.py --min-lines 7
```

**Options:**
- `--path <paths>` - Files or directories to scan (default: current directory)
- `--min-lines <n>` - Minimum lines for duplicate detection (default: 5)
- `--ignore-tests` - Skip test files (test_*.py, *_test.py, tests/)
- `--output <file>` - Save markdown report to file
- `--ignore-patterns <patterns>` - Additional patterns to ignore (default: archive, cache, venv)

### find_unused.py - Unused Code Finder

Finds functions and methods that aren't called anywhere in the codebase.

**Features:**
- Detects unused functions and methods
- Handles event system patterns (on_*, get_*, handle_*)
- Supports private method filtering
- Shows where functions ARE used (helpful context)
- Console and markdown output formats
- Progress indicators

**Usage:**
```bash
# Check single file for unused methods
python tools/find_unused.py behaviors/workspace_task_notes.py

# Check directory, save to markdown
python tools/find_unused.py behaviors/ --output unused_report.md

# Include private methods, show what IS used
python tools/find_unused.py base_agent.py --include-private --show-used

# Check multiple directories
python tools/find_unused.py src/ behaviors/ --output unused.md

# Check file and search entire codebase
python tools/find_unused.py behaviors/delegation.py --search-path .
```

**Options:**
- `paths` - Files or directories to analyze (positional argument)
- `--search-path <paths>` - Where to search for references (default: same as target paths)
- `--include-private` - Include private methods (starting with `_`)
- `--include-dunder` - Include dunder methods (`__init__`, `__str__`, etc.)
- `--show-used` - Show information about used functions too
- `--output <file>` - Save markdown report to file
- `--ignore-patterns <patterns>` - Patterns to ignore

### code_analysis_utils.py - Shared Utilities

Common utilities used by both analysis tools:

**Key Functions:**
- `scan_python_files()` - Recursively find Python files with filtering
- `parse_ast_safe()` - Parse Python files to AST with error handling
- `extract_code_blocks()` - Extract functions/classes as code blocks
- `extract_functions()` - Extract function definitions with metadata
- `find_references()` - Find references to functions in codebase
- `create_markdown_report()` - Generate markdown reports
- `print_progress()` - Display progress bars

**Classes:**
- `CodeBlock` - Represents a code block with location and AST hash
- `FunctionInfo` - Metadata about a function (name, location, references)

## Integration into Development Workflow

### Pre-Refactoring Checklist
```bash
# Before major refactoring, run both tools
python tools/detect_duplicates.py --min-lines 5 --output reports/duplicates.md
python tools/find_unused.py src/ behaviors/ --output reports/unused.md

# Review reports for:
# 1. Consolidation opportunities (duplicates)
# 2. Dead code removal (unused)
# 3. Refactoring priorities (impact scores)
```

### CI/CD Integration
```bash
# Add to pre-commit or CI pipeline
python tools/detect_duplicates.py --min-lines 10 || echo "Duplicates found"
python tools/find_unused.py src/ behaviors/ || echo "Unused code found"
```

### Periodic Code Health Checks
```bash
# Weekly/monthly code health report
mkdir -p reports/$(date +%Y-%m-%d)
python tools/detect_duplicates.py --output reports/$(date +%Y-%m-%d)/duplicates.md
python tools/find_unused.py src/ behaviors/ --output reports/$(date +%Y-%m-%d)/unused.md
```
