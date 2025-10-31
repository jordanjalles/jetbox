#!/bin/bash
# Example: Run a single L5 test to demonstrate the suite

echo "=========================================="
echo "EXAMPLE: Running L5_cli_calculator test"
echo "=========================================="
echo ""
echo "This demonstrates the evaluation suite without running a full test."
echo "To actually run the test (takes ~5 minutes):"
echo ""
echo "  pytest tests/test_project_evaluation.py::test_l5_task[L5_cli_calculator] -v"
echo ""
echo "The test will:"
echo "  1. Call orchestrator_main.py --once with the task goal"
echo "  2. Wait for completion (up to 5 min timeout)"
echo "  3. Find the workspace in .agent_workspace/"
echo "  4. Validate the created code:"
echo "     - Check for Calculator class"
echo "     - Check for evaluate() method"
echo "     - Run pytest if tests exist"
echo "     - Run ruff for code quality"
echo "  5. Save results to evaluation_results/project_eval_results.jsonl"
echo ""
echo "Expected result format:"
echo '{'
echo '  "task_id": "L5_cli_calculator",'
echo '  "level": "L5",'
echo '  "name": "CLI Calculator",'
echo '  "timestamp": "2025-10-31T20:00:00",'
echo '  "run_result": {'
echo '    "success": true,'
echo '    "duration": 234.5'
echo '  },'
echo '  "validation_result": {'
echo '    "success": true,'
echo '    "files_created": ["calculator.py", "test_calculator.py"],'
echo '    "tests_passed": true,'
echo '    "code_quality": true,'
echo '    "details": { ... }'
echo '  }'
echo '}'
echo ""
echo "=========================================="
