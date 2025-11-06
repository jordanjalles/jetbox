#!/usr/bin/env python3
"""
Phase 1 Test Runner - MetaProgrammer Baseline Validation

Runs all 5 Phase 1 tests and generates summary report.
"""
from pathlib import Path
import subprocess
import time
import json


def run_test(test_file, test_name):
    """Run a single test and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            ['python', str(test_file)],
            capture_output=True,
            text=True,
            timeout=180
        )

        duration = time.time() - start_time

        passed = result.returncode == 0

        return {
            'name': test_name,
            'file': str(test_file),
            'passed': passed,
            'duration': duration,
            'stdout': result.stdout[-2000:] if result.stdout else '',  # Last 2000 chars
            'stderr': result.stderr[-1000:] if result.stderr else ''
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            'name': test_name,
            'file': str(test_file),
            'passed': False,
            'duration': duration,
            'error': 'TIMEOUT (>180s)',
            'stdout': '',
            'stderr': ''
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            'name': test_name,
            'file': str(test_file),
            'passed': False,
            'duration': duration,
            'error': str(e),
            'stdout': '',
            'stderr': ''
        }


def main():
    """Run all Phase 1 tests."""
    print("\n" + "="*60)
    print("PHASE 1: BASELINE VALIDATION")
    print("="*60)
    print("Testing MetaProgrammer's ability to generate:")
    print("- 3 simple behaviors (no hooks)")
    print("- 2 simple agents (standard behaviors)")
    print()

    tests = [
        ('tests/test_meta_1_1_http_request_behavior.py', 'Test 1.1: HttpRequestBehavior'),
        ('tests/test_meta_1_2_json_tools_behavior.py', 'Test 1.2: JsonToolsBehavior'),
        ('tests/test_meta_1_3_environment_behavior.py', 'Test 1.3: EnvironmentBehavior'),
        ('tests/test_meta_3_2_doc_generator_agent.py', 'Test 3.2: DocGeneratorAgent'),
        ('tests/test_meta_3_3_test_generator_agent.py', 'Test 3.3: TestGeneratorAgent'),
    ]

    results = []
    start_time = time.time()

    for test_file, test_name in tests:
        result = run_test(Path(test_file), test_name)
        results.append(result)

        # Show immediate result
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"\n{status}: {test_name} ({result['duration']:.1f}s)")

        if not result['passed']:
            if 'error' in result:
                print(f"  Error: {result['error']}")
            elif result['stderr']:
                print(f"  Last error output:\n{result['stderr'][:500]}")

    total_duration = time.time() - start_time

    # Generate summary
    print("\n" + "="*60)
    print("PHASE 1 SUMMARY")
    print("="*60)

    passed = sum(1 for r in results if r['passed'])
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Duration: {total_duration:.1f}s")
    print()

    for result in results:
        status = "✅" if result['passed'] else "❌"
        print(f"{status} {result['name']}: {result['duration']:.1f}s")

    # Success gate: 4/5 must pass
    success_gate = passed >= 4

    print()
    print(f"Success Gate (4/5): {'✅ PASS' if success_gate else '❌ FAIL'}")

    # Save detailed results
    report_file = Path('evaluation_results/phase1_test_results.json')
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w') as f:
        json.dump({
            'phase': 1,
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total,
            'total_duration': total_duration,
            'success_gate_met': success_gate,
            'tests': results
        }, f, indent=2)

    print(f"\nDetailed results saved to: {report_file}")

    # Check generated artifacts
    print("\n" + "="*60)
    print("GENERATED ARTIFACTS")
    print("="*60)

    behaviors_dir = Path('behaviors')
    generated_behaviors = [
        'HttpRequestBehavior.py',
        'JsonToolsBehavior.py',
        'EnvironmentBehavior.py'
    ]

    print("\nBehaviors:")
    for behavior in generated_behaviors:
        exists = (behaviors_dir / behavior).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {behavior}")

    configs_dir = Path('config/agents')
    generated_agents = [
        'DocGeneratorAgent.yaml',
        'TestGeneratorAgent.yaml'
    ]

    print("\nAgent Configs:")
    for agent in generated_agents:
        exists = (configs_dir / agent).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {agent}")

    print()

    if success_gate:
        print("✅ PHASE 1 COMPLETE - Ready for Phase 2")
        return 0
    else:
        print("❌ PHASE 1 INCOMPLETE - Review failures before proceeding")
        return 1


if __name__ == "__main__":
    exit(main())
