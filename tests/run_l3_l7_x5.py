#!/usr/bin/env python3
"""Run L3-L7 tests x5 iterations."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from run_stress_tests import TESTS, run_test

# Filter to L3-L7 tests only
l3_to_l7_tests = [t for t in TESTS if 3 <= t["level"] <= 7]

print(f"Running {len(l3_to_l7_tests)} tests (L3-L7) x 5 iterations = {len(l3_to_l7_tests) * 5} total runs")
print()

all_results = []

for iteration in range(1, 6):
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}/5")
    print(f"{'='*80}\n")

    for test in l3_to_l7_tests:
        print(f"Running {test['id']}: {test['name']}...")
        result = run_test(test)
        result["iteration"] = iteration
        all_results.append(result)

        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"  {status} ({result['rounds']} rounds, {result['duration']:.1f}s)")

# Write results
with open("l3_l7_x5_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Summary
total = len(all_results)
passed = sum(1 for r in all_results if r["success"])
print(f"\n{'='*80}")
print(f"FINAL RESULTS: {passed}/{total} passed ({passed/total*100:.1f}%)")
print(f"{'='*80}")

# By level summary
from collections import defaultdict
by_level = defaultdict(lambda: {"passed": 0, "total": 0})
for r in all_results:
    level = r["level"]
    by_level[level]["total"] += 1
    if r["success"]:
        by_level[level]["passed"] += 1

print("\nResults by level:")
for level in sorted(by_level.keys()):
    stats = by_level[level]
    rate = stats["passed"] / stats["total"] * 100
    print(f"  L{level}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
