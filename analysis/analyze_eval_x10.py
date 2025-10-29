#!/usr/bin/env python3
"""Analyze eval suite x10 results and generate detailed failure analysis."""

import json
from collections import defaultdict
from pathlib import Path

def main():
    with open('eval_suite_results_x10.json') as f:
        results = json.load(f)

    # Overall stats
    total = len(results)
    passed = sum(1 for r in results if r.get('success'))
    failed = total - passed

    print("="*80)
    print("EVAL SUITE X10 - DETAILED ANALYSIS")
    print("="*80)
    print(f"\nOverall: {passed}/{total} passed ({passed/total*100:.1f}%)\n")

    # Group by test ID
    by_test = defaultdict(lambda: {'passed': 0, 'failed': 0, 'failures': []})

    for result in results:
        test_id = result['id']
        if result.get('success'):
            by_test[test_id]['passed'] += 1
        else:
            by_test[test_id]['failed'] += 1
            by_test[test_id]['failures'].append({
                'iteration': result.get('iteration', 0),
                'failure_mode': result.get('failure_mode', 'unknown'),
                'duration': result.get('duration', 0),
                'rounds': result.get('rounds', 0),
                'error': result.get('error', '')
            })

    # Print test-by-test analysis
    print("TEST-BY-TEST RESULTS:")
    print("-"*80)
    for test_id in sorted(by_test.keys()):
        data = by_test[test_id]
        total_runs = data['passed'] + data['failed']
        pass_rate = data['passed'] / total_runs * 100 if total_runs > 0 else 0

        status = "✓" if pass_rate == 100 else "⚠" if pass_rate >= 50 else "✗"
        print(f"\n{status} {test_id}: {data['passed']}/{total_runs} passed ({pass_rate:.0f}%)")

        if data['failures']:
            print(f"  Failures:")
            failure_modes = defaultdict(list)
            for f in data['failures']:
                failure_modes[f['failure_mode']].append(f['iteration'])

            for mode, iterations in failure_modes.items():
                print(f"    - {mode}: iterations {sorted(iterations)}")

    # Failure mode analysis
    print("\n" + "="*80)
    print("FAILURE MODE ANALYSIS:")
    print("-"*80)

    failure_modes = defaultdict(lambda: {'count': 0, 'tests': set()})
    for result in results:
        if not result.get('success'):
            mode = result.get('failure_mode', 'unknown')
            failure_modes[mode]['count'] += 1
            failure_modes[mode]['tests'].add(result['id'])

    for mode in sorted(failure_modes.keys(), key=lambda x: failure_modes[x]['count'], reverse=True):
        data = failure_modes[mode]
        print(f"\n{mode}: {data['count']} occurrences")
        print(f"  Affected tests: {', '.join(sorted(data['tests']))}")

    # Most problematic tests
    print("\n" + "="*80)
    print("MOST PROBLEMATIC TESTS:")
    print("-"*80)

    problem_tests = []
    for test_id, data in by_test.items():
        total_runs = data['passed'] + data['failed']
        fail_rate = data['failed'] / total_runs if total_runs > 0 else 0
        if fail_rate > 0:
            problem_tests.append((test_id, fail_rate, data['failed'], total_runs))

    problem_tests.sort(key=lambda x: x[1], reverse=True)

    for test_id, fail_rate, failed, total in problem_tests[:5]:
        print(f"\n{test_id}: {failed}/{total} failures ({fail_rate*100:.0f}% fail rate)")

        # Show failure details
        for failure in by_test[test_id]['failures'][:3]:  # First 3 failures
            print(f"  - Iter {failure['iteration']}: {failure['failure_mode']} "
                  f"({failure['duration']:.1f}s, {failure['rounds']} rounds)")

    # Level performance
    print("\n" + "="*80)
    print("LEVEL PERFORMANCE:")
    print("-"*80)

    by_level = defaultdict(lambda: {'passed': 0, 'total': 0})
    for result in results:
        level = result.get('level', 0)
        by_level[level]['total'] += 1
        if result.get('success'):
            by_level[level]['passed'] += 1

    for level in sorted(by_level.keys()):
        data = by_level[level]
        pct = data['passed'] / data['total'] * 100 if data['total'] > 0 else 0
        print(f"Level {level}: {data['passed']}/{data['total']} ({pct:.1f}%)")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
