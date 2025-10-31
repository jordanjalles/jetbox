# Project Evaluation Suite

Comprehensive test suite for evaluating the Jetbox orchestrator system on L5-L8 coding tasks and projects.

## What is This?

This evaluation suite tests the orchestrator's ability to complete real-world coding tasks of increasing complexity:

- **L5**: Simple utilities (file converters, validators, calculators)
- **L6**: Multi-file modules (API clients, data pipelines, config managers)
- **L7**: Complete packages (Python packages, libraries, CLI tools)
- **L8**: Full systems (microservices, web apps, distributed systems)

## Quick Start

### 1. Validate the suite

```bash
python test_eval_suite_quick.py
```

This confirms the test infrastructure works without running the orchestrator.

### 2. Run a single test

```bash
pytest tests/test_project_evaluation.py::test_l5_task[L5_cli_calculator] -v
```

This runs one L5 task (CLI calculator) and validates the result.

### 3. Run all L5 tests

```bash
python run_project_evaluation.py --level L5
```

This runs all 3 L5 tasks and generates a report.

### 4. Run the full suite

```bash
python run_project_evaluation.py
```

This runs all 15 tests (12 unique tasks, L8 tasks tested with/without architect) and generates a comprehensive report.

## Files

### Test Suite
- `tests/test_project_evaluation.py` - 15 pytest test cases covering L5-L8 tasks
- `run_project_evaluation.py` - Runner script with reporting
- `test_eval_suite_quick.py` - Quick validation (no orchestrator)

### Results
- `evaluation_results/project_eval_results.jsonl` - Raw results (one JSON per line)
- `evaluation_results/PROJECT_EVAL_SUMMARY.md` - Generated markdown report
- `evaluation_results/PROJECT_EVALUATION_GUIDE.md` - Detailed documentation

## Test Tasks

The suite includes 12 unique tasks:

### L5 Tasks (3)
1. **JSON/CSV Converter** - Convert between JSON and CSV formats
2. **Data Validator** - Validate emails, phones, URLs
3. **CLI Calculator** - Interactive calculator with history

### L6 Tasks (3)
1. **REST API Client** - HTTP client with auth and rate limiting
2. **Data Pipeline** - Multi-stage data processing pipeline
3. **Config Manager** - Configuration management with validation

### L7 Tasks (3)
1. **Python Package** - Complete package with setup.py, tests, docs
2. **Multi-Module Library** - Library with 4+ modules and dependencies
3. **CLI Tool** - Command-line tool with config and integration tests

### L8 Tasks (3)
1. **Microservices** - User and product services with Docker
2. **Web Application** - Todo app with backend, frontend, database
3. **Distributed System** - Task queue with API, worker, storage

Each L8 task is tested twice: with architect and without architect (6 total test cases).

## What Gets Validated?

### For L5-L7 Tasks
- Required classes/functions exist in code
- Tests pass (if included)
- Code quality (ruff checks pass)
- File structure (setup.py, README for L7)

### For L8 Tasks
- All components/services present
- Architecture matches requirements
- Docker/deployment config (if specified)
- Integration tests (if included)

## Example Output

```
================================================================================
JETBOX PROJECT EVALUATION SUITE
================================================================================

Running: python -m pytest tests/test_project_evaluation.py -v --tb=short

collected 15 items

tests/test_project_evaluation.py::test_l5_task[L5_json_csv_converter] PASSED
tests/test_project_evaluation.py::test_l5_task[L5_data_validator] PASSED
tests/test_project_evaluation.py::test_l5_task[L5_cli_calculator] PASSED
...

================================================================================
EVALUATION COMPLETED IN 45.2m
================================================================================

Generating summary report...

================================================================================
OVERALL SUMMARY
================================================================================
Total tasks:       15
Successful:        12 (80.0%)
Failed:            3
Total duration:    2712.3s (45.2m)
Average duration:  180.8s per task

...
```

## Understanding Results

After running tests, check:

1. **Console output** - See which tests passed/failed in real-time
2. **project_eval_results.jsonl** - Raw results for programmatic analysis
3. **PROJECT_EVAL_SUMMARY.md** - Human-readable report with details

Each result includes:
- Task completion status
- Duration and timeout
- Files created
- Validation details (missing classes/functions)
- Test results and code quality
- Error messages (if failed)

## Customization

### Add a new task

Edit `tests/test_project_evaluation.py` and add to the appropriate list:

```python
L5_TASKS.append({
    "id": "L5_my_task",
    "level": "L5",
    "name": "My Task",
    "description": "Brief description",
    "goal": """Detailed requirements...""",
    "validation": {
        "classes": ["MyClass"],
        "functions": ["my_function"],
    },
    "timeout": 300,  # 5 minutes
})
```

### Adjust timeouts

Change the `timeout` field (in seconds):
- L5-L7: 300 seconds (5 minutes) typical
- L8: 600 seconds (10 minutes) typical

### Custom validation

Modify `validate_task_result()` in `tests/test_project_evaluation.py` to add custom validation logic.

## Use Cases

### 1. Development Testing
Test orchestrator changes don't break existing functionality:
```bash
python run_project_evaluation.py --level L5
```

### 2. Performance Benchmarking
Compare orchestrator versions:
```bash
# Run full suite and save results
python run_project_evaluation.py
cp evaluation_results/project_eval_results.jsonl results_v1.jsonl

# Make changes, then re-run
python run_project_evaluation.py
cp evaluation_results/project_eval_results.jsonl results_v2.jsonl

# Compare
diff results_v1.jsonl results_v2.jsonl
```

### 3. Architect Evaluation
Compare L8 results with/without architect:
```bash
# Results include both architect and no-architect runs
python run_project_evaluation.py --level L8
# Check PROJECT_EVAL_SUMMARY.md for comparison
```

### 4. CI/CD Integration
Run in CI pipeline:
```bash
python run_project_evaluation.py
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Some tests failed - check report"
fi
```

## Troubleshooting

**Test times out**
- Increase `timeout` in task definition
- Check orchestrator logs for issues

**Workspace not found**
- Check `.agent_workspace/` directory
- Verify workspace creation logic

**Validation fails**
- Check `validation_result.details` in results
- Review actual files vs. requirements
- Check test output for errors

**No results file**
- Ensure `evaluation_results/` exists
- Check pytest runs successfully
- Look for permission errors

## Documentation

- **PROJECT_EVALUATION_GUIDE.md** - Comprehensive documentation
- **evaluation_results/README.md** - Results directory info
- This file - Quick reference

## Next Steps

1. **Run validation**: `python test_eval_suite_quick.py`
2. **Try one task**: `pytest tests/test_project_evaluation.py::test_l5_task[L5_cli_calculator] -v`
3. **Run level**: `python run_project_evaluation.py --level L5`
4. **Full suite**: `python run_project_evaluation.py`
5. **Review results**: Check `evaluation_results/PROJECT_EVAL_SUMMARY.md`

---

**Total:** 15 tests across 12 unique tasks (L5-L8)
**Duration:** ~45-60 minutes for full suite
**Output:** JSONL results + Markdown summary report
