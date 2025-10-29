#!/usr/bin/env python3
"""
Run L1-L7 evaluation, 3 runs per level.

Tests one task per difficulty level with 3 repetitions
to measure consistency and gather statistics.
"""
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from task_executor_agent import TaskExecutorAgent


# Select one representative task per level
TASKS = {
    "L1": {
        "name": "simple_function",
        "goal": "Create a Python file called greet.py with a function greet(name) that returns 'Hello, {name}!'",
        "expected_files": ["greet.py"],
        "validation": lambda ws: (
            (ws / "greet.py").exists()
            and "def greet" in (ws / "greet.py").read_text()
            and "Hello" in (ws / "greet.py").read_text()
        )
    },
    "L2": {
        "name": "class_definition",
        "goal": "Create person.py with a Person class. Include __init__(name, age), get_info() method returning formatted string, and birthday() method incrementing age.",
        "expected_files": ["person.py"],
        "validation": lambda ws: (
            (ws / "person.py").exists()
            and "class Person" in (ws / "person.py").read_text()
            and "def __init__" in (ws / "person.py").read_text()
            and "def get_info" in (ws / "person.py").read_text()
            and "def birthday" in (ws / "person.py").read_text()
        )
    },
    "L3": {
        "name": "file_io",
        "goal": "Create file_processor.py with functions: read_lines(filename) returning list of lines, write_lines(filename, lines), count_words(filename) returning word count, find_in_file(filename, pattern) returning matching lines.",
        "expected_files": ["file_processor.py"],
        "validation": lambda ws: (
            (ws / "file_processor.py").exists()
            and "def read_lines" in (ws / "file_processor.py").read_text()
            and "def write_lines" in (ws / "file_processor.py").read_text()
            and "def count_words" in (ws / "file_processor.py").read_text()
            and "def find_in_file" in (ws / "file_processor.py").read_text()
        )
    },
    "L4": {
        "name": "csv_processor",
        "goal": "Create csv_analyzer.py with functions: load_csv(filename) returning list of dicts, calculate_average(data, column), filter_rows(data, column, value), save_csv(data, filename). Use csv module.",
        "expected_files": ["csv_analyzer.py"],
        "validation": lambda ws: (
            (ws / "csv_analyzer.py").exists()
            and "def load_csv" in (ws / "csv_analyzer.py").read_text()
            and "def calculate_average" in (ws / "csv_analyzer.py").read_text()
            and "def filter_rows" in (ws / "csv_analyzer.py").read_text()
            and "def save_csv" in (ws / "csv_analyzer.py").read_text()
        )
    },
    "L5": {
        "name": "rest_api_mock",
        "goal": "Create api.py with a Flask app. Add GET /users returning JSON list, POST /users accepting JSON and returning created user with id, GET /users/<id> returning specific user. Include simple in-memory storage.",
        "expected_files": ["api.py"],
        "validation": lambda ws: (
            (ws / "api.py").exists()
            and "Flask" in (ws / "api.py").read_text()
            and "@app.route" in (ws / "api.py").read_text()
            and "/users" in (ws / "api.py").read_text()
        )
    },
    "L6": {
        "name": "async_downloader",
        "goal": "Create async_downloader.py using asyncio and aiohttp. Include async download_url(url, session), async download_multiple(urls) returning list of contents, and main() that demonstrates downloading 3 URLs concurrently.",
        "expected_files": ["async_downloader.py"],
        "validation": lambda ws: (
            (ws / "async_downloader.py").exists()
            and "async def" in (ws / "async_downloader.py").read_text()
            and "asyncio" in (ws / "async_downloader.py").read_text()
            and "aiohttp" in (ws / "async_downloader.py").read_text()
        )
    },
    "L7": {
        "name": "web_scraper",
        "goal": "Create web_scraper.py with BeautifulSoup. Include scrape_page(url) extracting title/links, scrape_multiple(urls) using ThreadPoolExecutor, save_results(data, filename) in JSON format. Add error handling for network failures.",
        "expected_files": ["web_scraper.py"],
        "validation": lambda ws: (
            (ws / "web_scraper.py").exists()
            and "def scrape_page" in (ws / "web_scraper.py").read_text()
            and "def scrape_multiple" in (ws / "web_scraper.py").read_text()
            and "def save_results" in (ws / "web_scraper.py").read_text()
        )
    }
}


def run_task(level: str, task_config: dict, run_number: int, max_rounds: int = 20):
    """
    Run a single task.

    Args:
        level: Level name (L1, L2, etc)
        task_config: Task configuration dict
        run_number: Which run this is (1-3)
        max_rounds: Maximum rounds before timeout

    Returns:
        Result dict with success, duration, files_ok, validation_ok
    """
    print(f"\n{'='*70}")
    print(f"LEVEL {level}: {task_config['name']} (Run {run_number}/3)")
    print(f"{'='*70}")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        agent = TaskExecutorAgent(
            workspace=workspace,
            goal=task_config["goal"],
            max_rounds=max_rounds,
            model="gpt-oss:20b",
            temperature=0.2
        )

        # Run agent
        import time
        start = time.time()
        try:
            result = agent.run()
            duration = time.time() - start

            # Check files
            actual_workspace = agent.workspace_manager.workspace_dir
            files_ok = all(
                (actual_workspace / f).exists()
                for f in task_config["expected_files"]
            )

            # Run validation
            validation_ok = False
            if files_ok:
                try:
                    validation_ok = task_config["validation"](actual_workspace)
                except Exception as e:
                    print(f"Validation error: {e}")
                    validation_ok = False

            success = files_ok and validation_ok

            return {
                "level": level,
                "name": task_config["name"],
                "run": run_number,
                "success": success,
                "duration": duration,
                "files_ok": files_ok,
                "validation_ok": validation_ok,
                "rounds": agent.state.total_rounds,
            }

        except Exception as e:
            duration = time.time() - start
            print(f"Error: {e}")
            return {
                "level": level,
                "name": task_config["name"],
                "run": run_number,
                "success": False,
                "duration": duration,
                "files_ok": False,
                "validation_ok": False,
                "error": str(e),
                "rounds": agent.state.total_rounds if hasattr(agent.state, 'total_rounds') else 0,
            }


def main():
    """Run L1-L7 x3 evaluation."""
    print("="*70)
    print("EVALUATION: L1-L7 x3")
    print("="*70)
    print()

    all_results = []

    # Run each level 3 times
    for level in ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]:
        task_config = TASKS[level]

        for run_num in range(1, 4):
            result = run_task(level, task_config, run_num)
            all_results.append(result)

            # Show result
            status = "✓" if result["success"] else "✗"
            print(f"{status} {level} run {run_num}: {result['duration']:.1f}s, {result['rounds']} rounds")

    # Calculate statistics
    print("\n" + "="*70)
    print("SUMMARY BY LEVEL")
    print("="*70)

    for level in ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]:
        level_results = [r for r in all_results if r["level"] == level]

        successes = sum(1 for r in level_results if r["success"])
        avg_duration = sum(r["duration"] for r in level_results) / len(level_results)
        avg_rounds = sum(r["rounds"] for r in level_results) / len(level_results)

        print(f"\n{level}: {successes}/3 passed (avg {avg_duration:.1f}s, {avg_rounds:.1f} rounds)")
        for r in level_results:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} Run {r['run']}: {r['duration']:.1f}s, {r['rounds']} rounds")

    # Overall stats
    total_success = sum(1 for r in all_results if r["success"])
    total_tasks = len(all_results)
    pass_rate = (total_success / total_tasks) * 100

    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"Total passed: {total_success}/{total_tasks} ({pass_rate:.1f}%)")
    print(f"Average duration: {sum(r['duration'] for r in all_results) / len(all_results):.1f}s")
    print(f"Average rounds: {sum(r['rounds'] for r in all_results) / len(all_results):.1f}")

    # Save results
    output_file = Path("eval_l1_l7_x3_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-oss:20b",
            "results": all_results,
            "summary": {
                "total_passed": total_success,
                "total_tasks": total_tasks,
                "pass_rate": pass_rate,
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
