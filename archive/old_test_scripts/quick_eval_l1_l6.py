#!/usr/bin/env python3
"""Quick evaluation: one task from each level L1-L6."""
import sys
import tempfile
import subprocess
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))

from task_executor_agent import TaskExecutorAgent


# One task from each level for quick testing
TASKS = [
    {
        "level": "L1",
        "name": "simple_function",
        "goal": "Create a Python file called greet.py with a function greet(name) that returns 'Hello, {name}!'",
        "expected_files": ["greet.py"],
        "validation": ["python", "-c", "from greet import greet; assert greet('World') == 'Hello, World!'"],
    },
    {
        "level": "L2",
        "name": "class_definition",
        "goal": "Create person.py with a Person class. Include __init__(name, age), get_info() method returning formatted string, and birthday() method incrementing age.",
        "expected_files": ["person.py"],
        "validation": ["python", "-c", "from person import Person; p = Person('Alice', 30); p.birthday(); assert p.age == 31"],
    },
    {
        "level": "L3",
        "name": "file_io",
        "goal": "Create file_processor.py with functions: read_lines(filename) returning list of lines, write_lines(filename, lines), count_words(filename) returning word count, find_in_file(filename, pattern) returning matching lines.",
        "expected_files": ["file_processor.py"],
        "validation": ["python", "-c", "from file_processor import count_words, write_lines; write_lines('test.txt', ['hello world']); assert count_words('test.txt') == 2"],
    },
    {
        "level": "L4",
        "name": "csv_processor",
        "goal": "Create csv_analyzer.py with functions: load_csv(filename) returning list of dicts, calculate_average(data, column), filter_rows(data, column, value), save_csv(data, filename). Use csv module.",
        "expected_files": ["csv_analyzer.py"],
        "validation": ["python", "-c", "from csv_analyzer import load_csv, calculate_average, save_csv; data = [{'a': 1}, {'a': 3}]; save_csv(data, 't.csv'); assert abs(calculate_average(data, 'a') - 2.0) < 0.01"],
    },
    {
        "level": "L5",
        "name": "rest_api_mock",
        "goal": "Create api.py with a Flask app. Add GET /users returning JSON list, POST /users accepting JSON and returning created user with id, GET /users/<id> returning specific user. Include simple in-memory storage.",
        "expected_files": ["api.py"],
        "validation": ["python", "-c", "import api; assert hasattr(api, 'app')"],
    },
    {
        "level": "L6",
        "name": "async_downloader",
        "goal": "Create async_downloader.py using asyncio and aiohttp. Include async download_url(url, session), async download_multiple(urls) returning list of contents, and main() that demonstrates downloading 3 URLs concurrently.",
        "expected_files": ["async_downloader.py"],
        "validation": ["python", "-c", "import async_downloader; assert hasattr(async_downloader, 'download_multiple')"],
    },
]


def run_task(task_def):
    """Run a single task and return results."""
    print(f"\n{'='*70}")
    print(f"LEVEL {task_def['level']}: {task_def['name']}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create and run agent
            agent = TaskExecutorAgent(
                workspace=workspace,
                goal=task_def['goal'],
                max_rounds=20,
                model="gpt-oss:20b"
            )

            result = agent.run()
            duration = time.time() - start_time

            workspace_dir = agent.workspace_manager.workspace_dir

            # Check files exist
            files_created = []
            for expected_file in task_def['expected_files']:
                file_path = workspace_dir / expected_file
                if file_path.exists():
                    files_created.append(expected_file)

            all_files_exist = len(files_created) == len(task_def['expected_files'])

            # Run validation
            validation_passed = False
            if all_files_exist:
                import os
                original_cwd = os.getcwd()
                try:
                    os.chdir(workspace_dir)
                    validation_result = subprocess.run(
                        task_def['validation'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    validation_passed = validation_result.returncode == 0
                except Exception as e:
                    print(f"  Validation error: {e}")
                finally:
                    os.chdir(original_cwd)

            success = all_files_exist and validation_passed

            status_icon = "✓" if success else "✗"
            print(f"{status_icon} {task_def['level']}: {task_def['name']} - {duration:.1f}s")

            return {
                "level": task_def['level'],
                "name": task_def['name'],
                "success": success,
                "duration": duration,
                "files_ok": all_files_exist,
                "validation_ok": validation_passed,
            }

    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ {task_def['level']}: {task_def['name']} - ERROR: {e}")
        return {
            "level": task_def['level'],
            "name": task_def['name'],
            "success": False,
            "duration": duration,
            "files_ok": False,
            "validation_ok": False,
            "error": str(e)
        }


def main():
    """Run quick L1-L6 evaluation."""
    print("="*70)
    print("QUICK EVALUATION: L1-L6 (1 task per level)")
    print("="*70)

    results = []

    for task_def in TASKS:
        result = run_task(task_def)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    by_level = {}
    for r in results:
        level = r['level']
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(r)

    for level in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        if level in by_level:
            level_results = by_level[level]
            passed = sum(1 for r in level_results if r['success'])
            total = len(level_results)
            avg_time = sum(r['duration'] for r in level_results) / total

            status = "✓" if passed == total else "✗"
            print(f"{status} {level}: {passed}/{total} passed (avg {avg_time:.1f}s)")

    total_passed = sum(1 for r in results if r['success'])
    total_tasks = len(results)
    pass_rate = (total_passed / total_tasks * 100) if total_tasks > 0 else 0

    print(f"\nOverall: {total_passed}/{total_tasks} ({pass_rate:.1f}%)")

    # Save results
    output_file = Path("quick_eval_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "summary": {
                "total_passed": total_passed,
                "total_tasks": total_tasks,
                "pass_rate": pass_rate,
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return 0 if pass_rate >= 66 else 1


if __name__ == "__main__":
    sys.exit(main())
