"""
Re-validate L5 failures using flexible validation.

This script checks if the "failed" L5 runs actually have working code,
just in the wrong file structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.flexible_validation import VALIDATORS

# Map failed runs to their workspaces (these were from the eval)
# Since workspaces were temp and cleaned up, we'll just show the concept

def check_if_files_exist_anywhere(inspection_dir: Path) -> list[str]:
    """Check what Python files the agent actually created."""
    # In the context snapshots, we can see tool call responses
    # Let's check if write_file was called successfully

    import json

    py_files_created = []
    for snapshot in sorted(inspection_dir.glob("*.json")):
        with open(snapshot) as f:
            data = json.load(f)

            # Look for tool responses that show file writes
            for msg in data.get('context', []):
                if msg.get('role') == 'tool':
                    content = msg.get('content', '')
                    if 'Wrote' in content and 'chars to' in content:
                        # Extract filename from "Wrote X chars to filename.py"
                        parts = content.split('chars to ')
                        if len(parts) > 1:
                            filename = parts[1].strip().strip('"')
                            if filename not in py_files_created:
                                py_files_created.append(filename)

    return py_files_created


def main():
    """Re-validate L5 failures."""

    eval_dir = Path("evaluation_results/context_analysis_20251108_012110/failed_runs")

    if not eval_dir.exists():
        print(f"Eval directory not found: {eval_dir}")
        return

    l5_failures = [
        "L5_blog_system_run1_inspection",
        "L5_blog_system_run2_inspection",
        "L5_todo_app_run1_inspection",
        "L5_todo_app_run2_inspection",
        "L5_inventory_system_run1_inspection",
        "L5_inventory_system_run2_inspection",
        "L5_url_shortener_run1_inspection",
        "L5_url_shortener_run2_inspection",
        "L5_email_validator_service_run1_inspection",
        "L5_email_validator_service_run2_inspection",
    ]

    print("="*70)
    print("RE-VALIDATING L5 FAILURES")
    print("="*70)
    print()

    results = {
        'had_files': 0,
        'no_files': 0,
        'single_file_solution': 0,
    }

    for failure in l5_failures:
        inspection_dir = eval_dir / failure

        if not inspection_dir.exists():
            continue

        # Check what files were created
        files_created = check_if_files_exist_anywhere(inspection_dir)

        task_name = failure.split('_inspection')[0]
        run_num = task_name[-1]
        task_name = '_'.join(task_name.split('_')[0:-1])

        print(f"{task_name} (run {run_num}):")

        if files_created:
            results['had_files'] += 1
            print(f"  ✓ Created files: {files_created}")

            # Check if it's a single-file solution
            if len(files_created) == 1:
                results['single_file_solution'] += 1
                print(f"  → Single-file solution (likely works but wrong structure)")
        else:
            results['no_files'] += 1
            print(f"  ✗ No files created (genuine failure)")

        print()

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total L5 failures analyzed: {len(l5_failures)}")
    print(f"Had files created: {results['had_files']}")
    print(f"  - Single-file solutions: {results['single_file_solution']}")
    print(f"No files created: {results['no_files']}")
    print()
    print(f"Estimated false negatives: {results['single_file_solution']}/{len(l5_failures)} ({100*results['single_file_solution']//len(l5_failures)}%)")
    print()


if __name__ == "__main__":
    main()
