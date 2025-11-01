import unittest
import time
import logging
import os

# Setup logging
os.makedirs('evaluation_results', exist_ok=True)
logging.basicConfig(filename='evaluation_results/log.txt', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# Load tests
loader = unittest.TestLoader()
suite = loader.discover('tests')

# Run tests and capture results
runner = unittest.TextTestRunner(resultclass=unittest.TestResult, verbosity=2)
start_time = time.time()
result = runner.run(suite)
end_time = time.time()

# Gather statistics
total_tests = result.testsRun
passed = total_tests - len(result.failures) - len(result.errors) - len(result.skipped)
failed = len(result.failures)
errors = len(result.errors)
skipped = len(result.skipped)

# Prepare markdown report
report_lines = []
report_lines.append('# Test Harness Report')
report_lines.append('')
report_lines.append(f'**Total tests run:** {total_tests}')
report_lines.append(f'**Passed:** {passed}')
report_lines.append(f'**Failed:** {failed}')
report_lines.append(f'**Errors:** {errors}')
report_lines.append(f'**Skipped:** {skipped}')
report_lines.append('')
report_lines.append(f'**Total execution time:** {end_time - start_time:.3f} seconds')
report_lines.append('')

# Bugs / Failures
if failed or errors:
    report_lines.append('## Failures and Errors')
    for failed_test, err in result.failures + result.errors:
        report_lines.append(f'- **{failed_test.id()}**')
        report_lines.append('  ```')
        report_lines.append(err)
        report_lines.append('  ```')
        report_lines.append('')
else:
    report_lines.append('## No failures or errors detected.')
    report_lines.append('')

# Improvements (simple suggestion)
report_lines.append('## Suggested Improvements')
if failed:
    report_lines.append('- Investigate the failure in the test cases and fix the underlying logic.')
else:
    report_lines.append('- All tests passed. Consider adding more edge case tests for robustness.')

# Write report
report_path = 'evaluation_results/report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

logging.info('Test harness completed. Report written to %s', report_path)
