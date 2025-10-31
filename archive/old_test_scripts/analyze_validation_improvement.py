#!/usr/bin/env python3
"""Analyze validation improvement potential from fixed validators."""

import json

# Load current results
with open("l3_l7_context_strategy_results.json") as f:
    results = json.load(f)

print("="*70)
print("VALIDATION ANALYSIS - BEFORE vs AFTER VALIDATOR FIX")
print("="*70)

# Group by task
tasks_by_name = {}
for result in results:
    task_name = result["task"]
    if task_name not in tasks_by_name:
        tasks_by_name[task_name] = {"hierarchical": None, "append_until_full": None}
    tasks_by_name[task_name][result["strategy"]] = result

# Analyze each task
print("\nCURRENT VALIDATION STATUS:")
print("-" * 70)

validators_needed = [
    "calculator",
    "file_processor", 
    "todo_list",
    "stack",
    "lru_cache"
]

has_validation = []
needs_validation = []

for task_name in sorted(tasks_by_name.keys()):
    hier = tasks_by_name[task_name]["hierarchical"]
    append = tasks_by_name[task_name]["append_until_full"]
    
    validator = task_name.split("_", 1)[1]  # L3_calculator -> calculator
    
    hier_passed = hier.get("validation_passed", 0)
    hier_failed = hier.get("validation_failed", 0)
    append_passed = append.get("validation_passed", 0) if append else 0
    append_failed = append.get("validation_failed", 0) if append else 0
    
    total_checks = hier_passed + hier_failed + append_passed + append_failed
    
    if total_checks > 0:
        has_validation.append(task_name)
        print(f"✓ {task_name:20s} - Has validation ({total_checks} checks)")
    else:
        needs_validation.append(task_name)
        was_fixed = validator in validators_needed
        status = "FIXED" if was_fixed else "MISSING"
        print(f"✗ {task_name:20s} - No validation ({status})")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nTasks with validation: {len(has_validation)}/10")
print(f"Tasks needing validation: {len(needs_validation)}/10")

print(f"\nValidators added in fix:")
for v in validators_needed:
    task = f"L?_{v}"
    status = "✓" if any(v in t for t in needs_validation) else "⚠️"
    print(f"  {status} {v}")

print(f"\n{'='*70}")
print("EXPECTED IMPACT")
print("="*70)

fixed_count = sum(1 for task in needs_validation if any(v in task for v in validators_needed))
print(f"\nTasks that will get validation: {fixed_count}/{len(needs_validation)}")
print(f"Overall validation coverage: {len(has_validation)}/10 → {len(has_validation) + fixed_count}/10")
print(f"Improvement: {(len(has_validation) + fixed_count)/10*100:.0f}% (from {len(has_validation)/10*100:.0f}%)")

print(f"\nNOTE: To see actual validation results, need to re-run evaluation")
print("with fixed validators, or manually validate workspaces if they still exist.")
