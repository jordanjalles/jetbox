# Project Evaluation Suite Guide

Comprehensive test suite for evaluating the Jetbox orchestrator system on L5-L7 coding tasks and L8 full projects.

## Overview

The project evaluation suite tests the orchestrator's ability to complete increasingly complex tasks:

- **L5 Tasks**: Simple single-file utilities (file parsers, validators, CLI tools)
- **L6 Tasks**: Multi-file modules (REST clients, data pipelines, config managers)
- **L7 Tasks**: Complete packages (Python packages with setup.py, multi-module libraries, CLI tools)
- **L8 Tasks**: Full multi-component systems (microservices, web applications, distributed systems)

## Files

### Core Files

- **`tests/test_project_evaluation.py`** - Pytest test suite with all test cases
- **`run_project_evaluation.py`** - Runner script to execute tests and generate reports
- **`test_eval_suite_quick.py`** - Quick validation script (no orchestrator runs)

### Results

- **`evaluation_results/project_eval_results.jsonl`** - Raw test results (JSONL format)
- **`evaluation_results/PROJECT_EVAL_SUMMARY.md`** - Generated summary report

## Quick Start

### 1. Validate the suite is working

```bash
python test_eval_suite_quick.py
```

This tests the suite infrastructure without running the orchestrator.

### 2. Run a single task test

```bash
# Test a specific L5 task
pytest tests/test_project_evaluation.py::test_l5_task[L5_cli_calculator] -v

# Test a specific L6 task
pytest tests/test_project_evaluation.py::test_l6_task[L6_rest_api_client] -v
```

### 3. Run all tests for a level

```bash
# Run all L5 tests
python run_project_evaluation.py --level L5

# Run all L6 tests
python run_project_evaluation.py --level L6
```

### 4. Run the full suite

```bash
python run_project_evaluation.py
```

This runs all 12 tasks (3 per level) and generates a comprehensive report.

## Test Tasks

### L5 Tasks (Single-file utilities)

1. **L5_json_csv_converter**: File converter between JSON and CSV formats
   - Required: FileConverter class with json_to_csv, csv_to_json methods
   - Timeout: 5 minutes

2. **L5_data_validator**: Data validation for email, phone, URL formats
   - Required: DataValidator class with validate_email, validate_phone, validate_url
   - Timeout: 5 minutes

3. **L5_cli_calculator**: Command-line calculator with history
   - Required: Calculator class with evaluate method
   - Timeout: 5 minutes

### L6 Tasks (Multi-file modules)

1. **L6_rest_api_client**: REST API client with auth and rate limiting
   - Required: APIClient, AuthHandler classes with get/post methods
   - Timeout: 5 minutes

2. **L6_data_pipeline**: Data processing pipeline with stages
   - Required: Pipeline, Stage, FilterStage, TransformStage classes
   - Timeout: 5 minutes

3. **L6_config_manager**: Configuration manager with validation
   - Required: ConfigManager, ConfigSchema classes with load_config, validate
   - Timeout: 5 minutes

### L7 Tasks (Complete packages)

1. **L7_python_package**: Python package 'textutils' with setup.py
   - Required: Package structure, setup.py, README.md, tests
   - Timeout: 5 minutes

2. **L7_multi_module_library**: Library 'datalib' with multiple modules
   - Required: 4+ modules with readers, writers, transformers, validators
   - Timeout: 5 minutes

3. **L7_cli_tool**: CLI tool 'filetool' with commands
   - Required: CLI commands (search, replace, stats), setup.py
   - Timeout: 5 minutes

### L8 Tasks (Full projects)

1. **L8_microservices**: Microservices with user-service and product-service
   - Required: 2 services, docker-compose.yml, integration tests
   - Timeout: 10 minutes

2. **L8_web_application**: Todo web app with backend, frontend, database
   - Required: Backend API, frontend UI, database schema
   - Timeout: 10 minutes

3. **L8_distributed_system**: Distributed task processing with queue
   - Required: API, worker, storage components, queue implementation
   - Timeout: 10 minutes

## Validation Criteria

### L5-L7 Tasks

Tests check for:
1. **Required symbols** (classes/functions) present in code
2. **Tests pass** (if test files included)
3. **Code quality** (ruff checks pass)
4. **File structure** (setup.py, README for L7)

### L8 Tasks

Tests check for:
1. **All components** present (services, modules)
2. **Architecture** matches requirements
3. **Docker/deployment** configuration (if specified)
4. **Integration tests** (if included)

## Result Format

Each test result in `project_eval_results.jsonl` contains:

```json
{
  "task_id": "L5_cli_calculator",
  "level": "L5",
  "name": "CLI Calculator",
  "description": "Create a command-line calculator with history",
  "timestamp": "2025-10-31T20:00:00",
  "run_result": {
    "success": true,
    "duration": 123.45,
    "error": null
  },
  "validation_result": {
    "success": true,
    "files_created": ["calculator.py", "test_calculator.py"],
    "tests_passed": true,
    "code_quality": true,
    "details": {
      "symbols": {
        "found": {"classes": ["Calculator"], "functions": ["evaluate"]},
        "missing": {"classes": [], "functions": []}
      }
    }
  }
}
```

## Summary Report

After running tests, a markdown summary is generated at `evaluation_results/PROJECT_EVAL_SUMMARY.md` with:

1. **Overall statistics** - Total pass rate, duration, etc.
2. **Results by level** - Pass rates and average durations for each level
3. **Task table** - Summary table of all tasks
4. **Detailed results** - Full details for each task including files, errors, validation

## Customizing Tests

### Adding New Tasks

Edit `tests/test_project_evaluation.py` and add to the appropriate task list:

```python
L5_TASKS.append({
    "id": "L5_my_task",
    "level": "L5",
    "name": "My Custom Task",
    "description": "Description of task",
    "goal": """Detailed goal with requirements...""",
    "validation": {
        "classes": ["MyClass"],
        "functions": ["my_function"],
    },
    "timeout": 300,
})
```

### Adjusting Timeouts

Edit the `timeout` field in task definitions (in seconds):
- L5-L7: 300 seconds (5 minutes) typical
- L8: 600 seconds (10 minutes) typical

### Custom Validation

For special validation needs, modify `validate_task_result()` in the test file.

## Troubleshooting

### Test times out

- Increase the `timeout` field in the task definition
- Check orchestrator is running correctly
- Review orchestrator logs for issues

### Workspace not found

- Check `.agent_workspace/` directory for created workspaces
- Workspace slug is created from task goal (lowercase, hyphens)
- Adjust `find_workspace_for_task()` logic if needed

### Validation fails

- Check `validation_result.details` for specific failures
- Review files created vs. requirements
- Check test output for specific errors

### No results file

- Ensure `evaluation_results/` directory exists
- Check pytest runs successfully
- Look for permission errors

## Performance Benchmarking

The suite can be used to benchmark:

1. **Completion rates** - What percentage of tasks complete successfully?
2. **Duration** - How long does each level/task take?
3. **Quality** - Do tests pass? Does code quality check pass?
4. **Architect value** - L8 tests run with and without architect (compare results)

## CI/CD Integration

To run in CI:

```bash
# Run full suite
python run_project_evaluation.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Some tests failed"
fi

# Parse results
python -c "
import json
results = []
with open('evaluation_results/project_eval_results.jsonl') as f:
    for line in f:
        results.append(json.loads(line))
success_rate = sum(r['validation_result']['success'] for r in results) / len(results)
print(f'Success rate: {success_rate:.1%}')
"
```

## Future Enhancements

Potential improvements:

1. **Parallel execution** - Run multiple tests concurrently
2. **Retry logic** - Retry failed tests with different seeds
3. **Regression tracking** - Compare results across runs
4. **Performance profiles** - Track token usage, LLM calls, etc.
5. **Custom validators** - Task-specific validation logic
6. **Interactive reports** - HTML reports with charts

## Support

For issues or questions:
- Check test output and logs
- Review orchestrator behavior
- Examine workspace files directly
- Adjust validation criteria as needed
