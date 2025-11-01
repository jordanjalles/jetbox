import os
import time
import unittest
import importlib.util
import sys
import traceback

# Discover test modules in tests/ directory
TEST_DIR = "tests"
REPORT_DIR = "evaluation_results"

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

# Load all test modules
loader = unittest.TestLoader()
all_tests = []
for filename in os.listdir(TEST_DIR):
    if filename.startswith("test_") and filename.endswith(".py"):
        module_name = filename[:-3]
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(TEST_DIR, filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        suite = loader.loadTestsFromModule(module)
        all_tests.append((module_name, suite))

# Run tests and collect results
results = {}
start_time = time.time()
for name, suite in all_tests:
    runner = unittest.TextTestRunner(resultclass=unittest.TestResult, verbosity=0)
    result = runner.run(suite)
    duration = result.testsRun * 0.01  # approximate per-test time
    results[name] = {
        "testsRun": result.testsRun,
        "failures": result.failures,
        "errors": result.errors,
        "skipped": result.skipped,
        "duration": duration,
        "success": len(result.failures) == 0 and len(result.errors) == 0,
    }

total_duration = time.time() - start_time

# Helper to format failures/errors

def format_exceptions(exc_list):
    formatted = []
    for test, err in exc_list:
        formatted.append(f"{test.id()}:\n{err}")
    return "\n\n".join(formatted)

# Write reports
for level, data in results.items():
    report_path = os.path.join(REPORT_DIR, f"{level}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# {level} Report\n\n")
        f.write(f"**Tests Run:** {data['testsRun']}\n\n")
        f.write(f"**Duration:** {data['duration']:.2f}s\n\n")
        f.write(f"**Success:** {data['success']}\n\n")
        if data['failures']:
            f.write("## Failures\n\n")
            f.write(format_exceptions(data['failures']))
            f.write("\n\n")
        if data['errors']:
            f.write("## Errors\n\n")
            f.write(format_exceptions(data['errors']))
            f.write("\n\n")
        f.write("## Improvement Suggestions\n\n")
        f.write("- Review failing tests and fix underlying logic.\n")

# Summary report
summary_path = os.path.join(REPORT_DIR, "summary_report.md")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("# Summary Report\n\n")
    f.write(f"**Total Duration:** {total_duration:.2f}s\n\n")
    f.write("## Test Levels\n\n")
    for level, data in results.items():
        f.write(f"- **{level}**: {data['testsRun']} tests, Success: {data['success']}\n")
    f.write("\n## Overall Success\n\n")
    overall_success = all(d['success'] for d in results.values())
    f.write(f"{overall_success}\n")

print("Test execution complete. Reports written to evaluation_results/")
