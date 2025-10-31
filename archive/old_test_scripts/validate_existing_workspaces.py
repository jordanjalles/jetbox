#!/usr/bin/env python3
"""Validate existing workspace outputs from L3-L7 evaluation."""

import json
from pathlib import Path
from semantic_validator import validate_workspace

# Load the results
with open("l3_l7_context_strategy_results.json") as f:
    results = json.load(f)

print("="*70)
print("VALIDATING EXISTING WORKSPACES")
print("="*70)

# Find workspace directories
workspace_base = Path(".agent_workspace")
if not workspace_base.exists():
    print(f"ERROR: {workspace_base} not found")
    exit(1)

# Group by task name
tasks = {}
for result in results:
    task_name = result["task"]
    strategy = result["strategy"]
    if task_name not in tasks:
        tasks[task_name] = {}
    tasks[task_name][strategy] = result

# Validate each workspace
updated_results = []
for result in results:
    task_name = result["task"]
    strategy = result["strategy"]
    validator_name = result.get("validator", task_name.split("_", 1)[1])  # L3_calculator -> calculator
    
    print(f"\n{'='*70}")
    print(f"Task: {task_name} | Strategy: {strategy}")
    print(f"Validator: {validator_name}")
    
    # Find workspace directory
    # Workspaces are named after the goal, which varies by task
    workspace_dirs = list(workspace_base.glob(f"*{task_name}*"))
    if not workspace_dirs:
        workspace_dirs = list(workspace_base.glob(f"*{validator_name}*"))
    
    if not workspace_dirs:
        print(f"  ⚠️  No workspace found (task may have crashed)")
        updated_results.append(result)
        continue
    
    # Use most recent workspace
    workspace_dir = max(workspace_dirs, key=lambda p: p.stat().st_mtime)
    print(f"  Workspace: {workspace_dir.name}")
    
    # Run validation
    validation = validate_workspace(workspace_dir, validator_name)
    
    # Update result
    old_passed = result.get("validation_passed", 0)
    old_failed = result.get("validation_failed", 0)
    
    # Calculate new counts
    new_passed = 0
    new_failed = 0
    if "found" in validation:
        for symbol_type, symbols in validation["found"].items():
            new_passed += len(symbols)
    if "missing" in validation:
        for symbol_type, symbols in validation["missing"].items():
            new_failed += len(symbols)
    
    result["validation_passed"] = new_passed
    result["validation_failed"] = new_failed
    result["validation_found"] = validation.get("found", {})
    result["validation_missing"] = validation.get("missing", {})
    result["success"] = validation.get("success", False)
    
    print(f"  Old validation: {old_passed}/{old_passed + old_failed} passed")
    print(f"  New validation: {new_passed}/{new_passed + new_failed} passed")
    
    if new_passed > old_passed:
        print(f"  ✓ IMPROVEMENT: +{new_passed - old_passed} symbols found")
    elif new_passed == old_passed and new_passed > 0:
        print(f"  ✓ Same validation (working)")
    elif new_passed == 0 and old_passed == 0:
        print(f"  ⚠️  Still no validation data")
    else:
        print(f"  ⚠️  Validation regressed")
    
    updated_results.append(result)

# Save updated results
output_file = "l3_l7_context_strategy_results_revalidated.json"
with open(output_file, "w") as f:
    json.dump(updated_results, f, indent=2)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

# Calculate stats
total_tasks = len(updated_results)
tasks_with_validation = sum(1 for r in updated_results if r["validation_passed"] + r["validation_failed"] > 0)
tasks_without_validation = total_tasks - tasks_with_validation

print(f"Total tasks: {total_tasks}")
print(f"Tasks with validation data: {tasks_with_validation}/{total_tasks} ({tasks_with_validation/total_tasks*100:.1f}%)")
print(f"Tasks without validation: {tasks_without_validation}/{total_tasks} ({tasks_without_validation/total_tasks*100:.1f}%)")

# By strategy
for strategy in ["hierarchical", "append_until_full"]:
    strat_results = [r for r in updated_results if r["strategy"] == strategy]
    strat_with_val = sum(1 for r in strat_results if r["validation_passed"] + r["validation_failed"] > 0)
    print(f"\n{strategy}:")
    print(f"  With validation: {strat_with_val}/{len(strat_results)}")

print(f"\nUpdated results saved to: {output_file}")
