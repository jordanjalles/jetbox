# Evaluation Results

This directory contains results from the Jetbox project evaluation suite.

## Files

- `project_eval_results.jsonl` - Raw test results in JSONL format (one result per line)
- `PROJECT_EVAL_SUMMARY.md` - Human-readable summary report with statistics and detailed results

## Running Evaluations

Run the full suite:
```bash
python run_project_evaluation.py
```

Run specific level:
```bash
python run_project_evaluation.py --level L5
```

Run specific task:
```bash
python run_project_evaluation.py --task L5_json_csv_converter
```

## Result Format

Each line in `project_eval_results.jsonl` contains:
- `task_id` - Unique task identifier
- `level` - Task level (L5, L6, L7, L8)
- `name` - Human-readable task name
- `description` - Task description
- `timestamp` - When test was run
- `run_result` - Orchestrator execution results
- `validation_result` - Code validation results

## Evaluation Criteria

### L5-L7 Tasks (Code-focused)
- Required classes/functions present
- Tests pass (if included)
- Code quality (ruff checks)
- File structure

### L8 Tasks (Architecture-focused)
- All components present
- Architecture matches requirements
- Integration between components
- Docker/deployment configuration (if required)
