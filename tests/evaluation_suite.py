"""
Evaluation Suite: Progressive Difficulty Testing for Jetbox Agent

Tests the orchestrator and task executor with progressively harder tasks to:
1. Identify what types of mistakes the agent makes
2. Highlight what types of problems cause failures
3. Categorize failure patterns
4. Generate detailed reports

Task Levels:
- Level 1: Basic (single file, simple logic)
- Level 2: Intermediate (multi-file, OOP, file I/O)
- Level 3: Advanced (dependencies, algorithms, workflows)
- Level 4: Complex (optimization, edge cases, integration)
"""
import sys
from pathlib import Path
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_executor_agent import TaskExecutorAgent


@dataclass
class TaskDefinition:
    """Definition of an evaluation task."""
    level: int  # 1-4
    name: str
    description: str
    goal: str  # What we tell the agent
    expected_files: list[str]  # Files we expect to exist
    validation_commands: list[list[str]]  # Commands to run for validation
    timeout_rounds: int = 20  # Max rounds before timeout
    tags: list[str] = field(default_factory=list)  # Categories: "single-file", "multi-file", "algorithm", etc


@dataclass
class TaskResult:
    """Result of running a task."""
    task: TaskDefinition
    success: bool
    duration: float
    rounds_used: int
    files_created: list[str]
    validation_passed: bool
    validation_output: str
    failure_category: str | None
    error_message: str | None
    agent_output: str


class FailureCategory:
    """Categories for classifying failures."""
    SYNTAX_ERROR = "syntax_error"
    LOGIC_ERROR = "logic_error"
    MISSING_FILES = "missing_files"
    IMPORT_ERROR = "import_error"
    TEST_FAILURE = "test_failure"
    INCOMPLETE = "incomplete_implementation"
    WRONG_APPROACH = "wrong_approach"
    TIMEOUT = "timeout_exceeded"
    MISUNDERSTOOD_SPEC = "misunderstood_specification"
    TOOL_ERROR = "tool_usage_error"
    UNKNOWN = "unknown_failure"


class EvaluationSuite:
    """Runs progressive difficulty evaluation."""

    def __init__(self, model: str = None, output_dir: Path = None):
        self.model = model
        self.output_dir = output_dir or Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        self.tasks = self._define_tasks()
        self.results: list[TaskResult] = []

    def _define_tasks(self) -> list[TaskDefinition]:
        """Define all evaluation tasks."""
        return [
            # ===== LEVEL 1: BASIC =====
            TaskDefinition(
                level=1,
                name="simple_function",
                description="Single file with one simple function",
                goal="Create a Python file called greet.py with a function greet(name) that returns 'Hello, {name}!'",
                expected_files=["greet.py"],
                validation_commands=[
                    ["python", "-c", "from greet import greet; assert greet('World') == 'Hello, World!'"]
                ],
                tags=["single-file", "basic"]
            ),
            TaskDefinition(
                level=1,
                name="simple_math",
                description="Basic arithmetic operations",
                goal="Create math_utils.py with functions: add(a,b), subtract(a,b), multiply(a,b), divide(a,b). Include proper error handling for division by zero.",
                expected_files=["math_utils.py"],
                validation_commands=[
                    ["python", "-c", "from math_utils import add, divide; assert add(2,3)==5; assert divide(10,2)==5"]
                ],
                tags=["single-file", "basic", "error-handling"]
            ),
            TaskDefinition(
                level=1,
                name="list_operations",
                description="Simple list manipulation",
                goal="Create list_utils.py with functions: get_max(lst), get_min(lst), get_average(lst), remove_duplicates(lst)",
                expected_files=["list_utils.py"],
                validation_commands=[
                    ["python", "-c", "from list_utils import get_max, remove_duplicates; assert get_max([1,5,3])==5; assert remove_duplicates([1,2,2,3])==[1,2,3]"]
                ],
                tags=["single-file", "data-structures"]
            ),

            # ===== LEVEL 2: INTERMEDIATE =====
            TaskDefinition(
                level=2,
                name="class_definition",
                description="Basic OOP with class",
                goal="Create a Person class in person.py with attributes: name, age, email. Include __init__, __str__, and a method is_adult() that returns True if age >= 18.",
                expected_files=["person.py"],
                validation_commands=[
                    ["python", "-c", "from person import Person; p=Person('Alice',25,'a@b.com'); assert p.is_adult()==True; assert 'Alice' in str(p)"]
                ],
                tags=["oop", "single-file"]
            ),
            TaskDefinition(
                level=2,
                name="multi_file_package",
                description="Package with multiple modules",
                goal="Create a package called 'shapes' with: shapes/__init__.py (exports all), shapes/circle.py (Circle class with area() method), shapes/rectangle.py (Rectangle class with area() method)",
                expected_files=["shapes/__init__.py", "shapes/circle.py", "shapes/rectangle.py"],
                validation_commands=[
                    ["python", "-c", "from shapes import Circle, Rectangle; import math; c=Circle(5); assert abs(c.area()-math.pi*25)<0.1"]
                ],
                tags=["multi-file", "oop", "package"],
                timeout_rounds=25
            ),
            TaskDefinition(
                level=2,
                name="file_io",
                description="Read and write files",
                goal="Create file_processor.py with: write_lines(filename, lines), read_lines(filename), count_words(filename). Include proper error handling for missing files.",
                expected_files=["file_processor.py"],
                validation_commands=[
                    ["python", "-c", "from file_processor import write_lines, read_lines; write_lines('test.txt',['hello','world']); assert len(read_lines('test.txt'))==2"]
                ],
                tags=["file-io", "error-handling"]
            ),
            TaskDefinition(
                level=2,
                name="data_validation",
                description="Input validation and error handling",
                goal="Create validator.py with: validate_email(email), validate_phone(phone), validate_url(url). Each returns True/False. Use regex patterns.",
                expected_files=["validator.py"],
                validation_commands=[
                    ["python", "-c", "from validator import validate_email, validate_phone; assert validate_email('test@example.com'); assert not validate_email('invalid')"]
                ],
                tags=["validation", "regex"]
            ),

            # ===== LEVEL 3: ADVANCED =====
            TaskDefinition(
                level=3,
                name="sorting_algorithms",
                description="Implement sorting algorithms",
                goal="Create sorting.py with implementations of: bubble_sort(lst), quick_sort(lst), merge_sort(lst). Include docstrings explaining the algorithms.",
                expected_files=["sorting.py"],
                validation_commands=[
                    ["python", "-c", "from sorting import bubble_sort, quick_sort, merge_sort; assert bubble_sort([3,1,2])==[1,2,3]; assert quick_sort([3,1,2])==[1,2,3]"]
                ],
                tags=["algorithms", "advanced"],
                timeout_rounds=30
            ),
            TaskDefinition(
                level=3,
                name="json_api_client",
                description="REST API client with JSON",
                goal="Create api_client.py with a class JSONClient that has methods: get(url), post(url, data), parse_response(). Use requests library. Include error handling for network errors.",
                expected_files=["api_client.py"],
                validation_commands=[
                    ["python", "-c", "from api_client import JSONClient; c=JSONClient(); assert hasattr(c, 'get'); assert hasattr(c, 'post')"]
                ],
                tags=["api", "dependencies", "error-handling"],
                timeout_rounds=30
            ),
            TaskDefinition(
                level=3,
                name="csv_processor",
                description="CSV file processing with statistics",
                goal="Create csv_analyzer.py with: load_csv(filename), get_column_stats(data, column), filter_rows(data, condition), export_csv(data, filename). Handle missing values.",
                expected_files=["csv_analyzer.py"],
                validation_commands=[
                    ["python", "-c", "from csv_analyzer import load_csv, get_column_stats; import csv; assert callable(load_csv)"]
                ],
                tags=["data-processing", "csv"],
                timeout_rounds=30
            ),
            TaskDefinition(
                level=3,
                name="cache_decorator",
                description="Implement memoization decorator",
                goal="Create decorators.py with a @memoize decorator that caches function results. Include a function to clear the cache. Test with fibonacci function.",
                expected_files=["decorators.py"],
                validation_commands=[
                    ["python", "-c", "from decorators import memoize; @memoize\ndef fib(n): return n if n<2 else fib(n-1)+fib(n-2)\nassert fib(10)==55"]
                ],
                tags=["decorators", "caching", "advanced"],
                timeout_rounds=30
            ),

            # ===== LEVEL 4: COMPLEX =====
            TaskDefinition(
                level=4,
                name="database_orm",
                description="Simple ORM for SQLite",
                goal="Create database.py with a simple ORM: Model base class, Database connection manager, methods for save(), find(), delete(). Use SQLite. Include a User model example.",
                expected_files=["database.py"],
                validation_commands=[
                    ["python", "-c", "from database import Database; db=Database(':memory:'); assert hasattr(db, 'save')"]
                ],
                tags=["database", "orm", "complex"],
                timeout_rounds=40
            ),
            TaskDefinition(
                level=4,
                name="async_downloader",
                description="Async file downloader with progress",
                goal="Create async_downloader.py using asyncio. Implement: download_file(url, filename), download_multiple(urls), with progress tracking. Handle errors gracefully.",
                expected_files=["async_downloader.py"],
                validation_commands=[
                    ["python", "-c", "import asyncio; from async_downloader import download_file; assert callable(download_file)"]
                ],
                tags=["async", "concurrency", "complex"],
                timeout_rounds=40
            ),
            TaskDefinition(
                level=4,
                name="test_framework",
                description="Mini testing framework",
                goal="Create test_framework.py with: TestCase base class, assertEqual/assertTrue/assertRaises methods, TestRunner that discovers and runs tests, generates reports.",
                expected_files=["test_framework.py"],
                validation_commands=[
                    ["python", "-c", "from test_framework import TestCase; class T(TestCase): pass\nt=T(); assert hasattr(t, 'assertEqual')"]
                ],
                tags=["testing", "framework", "complex"],
                timeout_rounds=40
            ),
        ]

    def run_task(self, task: TaskDefinition, workspace: Path) -> TaskResult:  # noqa: C901
        """Run a single task and evaluate results."""
        print(f"\n{'='*70}")
        print(f"Level {task.level}: {task.name}")
        print(f"Goal: {task.goal}")
        print(f"{'='*70}")

        start_time = time.time()
        error_msg = None
        failure_category = None
        agent_output = ""

        try:
            # Create task executor
            executor = TaskExecutorAgent(
                workspace=workspace,
                goal=task.goal,
                max_rounds=task.timeout_rounds,
                model=self.model
            )

            # Note: We would normally run executor.run() here, but for testing
            # we'll just verify the setup works and manually check files
            # In a real eval, you'd run: result = executor.run()

            # For now, simulate running by checking if workspace was set up
            workspace_dir = executor.workspace_manager.workspace_dir
            print(f"Workspace: {workspace_dir}")

            # Check expected files exist (would be created by agent.run())
            files_created = []
            all_files_exist = True
            for expected_file in task.expected_files:
                file_path = workspace_dir / expected_file
                if file_path.exists():
                    files_created.append(expected_file)
                else:
                    all_files_exist = False

            # Run validation commands
            validation_passed = False
            validation_output = ""

            if all_files_exist:
                # Change to workspace directory for validation
                import os
                original_cwd = os.getcwd()
                try:
                    os.chdir(workspace_dir)

                    # Run validation commands
                    import subprocess
                    for cmd in task.validation_commands:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        validation_output += f"Command: {' '.join(cmd)}\n"
                        validation_output += f"Exit code: {result.returncode}\n"
                        validation_output += f"Output: {result.stdout}\n"
                        if result.stderr:
                            validation_output += f"Error: {result.stderr}\n"

                        if result.returncode == 0:
                            validation_passed = True
                finally:
                    os.chdir(original_cwd)
            else:
                failure_category = FailureCategory.MISSING_FILES
                validation_output = f"Missing files: {set(task.expected_files) - set(files_created)}"

            duration = time.time() - start_time
            success = validation_passed and all_files_exist

            # Categorize failures
            if not success and not failure_category:
                if "SyntaxError" in validation_output or "IndentationError" in validation_output:
                    failure_category = FailureCategory.SYNTAX_ERROR
                elif "ImportError" in validation_output or "ModuleNotFoundError" in validation_output:
                    failure_category = FailureCategory.IMPORT_ERROR
                elif "AssertionError" in validation_output:
                    failure_category = FailureCategory.TEST_FAILURE
                elif not all_files_exist:
                    failure_category = FailureCategory.MISSING_FILES
                else:
                    failure_category = FailureCategory.UNKNOWN

            return TaskResult(
                task=task,
                success=success,
                duration=duration,
                rounds_used=0,  # Would get from executor
                files_created=files_created,
                validation_passed=validation_passed,
                validation_output=validation_output,
                failure_category=failure_category,
                error_message=error_msg,
                agent_output=agent_output
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            failure_category = FailureCategory.UNKNOWN

            return TaskResult(
                task=task,
                success=False,
                duration=duration,
                rounds_used=0,
                files_created=[],
                validation_passed=False,
                validation_output="",
                failure_category=failure_category,
                error_message=error_msg,
                agent_output=agent_output
            )

    def run_suite(self, max_level: int = 4, specific_tasks: list[str] = None):
        """Run the full evaluation suite."""
        print("\n" + "="*70)
        print("JETBOX EVALUATION SUITE")
        print("="*70)

        tasks_to_run = self.tasks
        if specific_tasks:
            tasks_to_run = [t for t in self.tasks if t.name in specific_tasks]
        if max_level < 4:
            tasks_to_run = [t for t in tasks_to_run if t.level <= max_level]

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            for task in tasks_to_run:
                result = self.run_task(task, workspace)
                self.results.append(result)

                # Print result
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"\n{status} - {result.task.name} ({result.duration:.1f}s)")
                if not result.success:
                    print(f"  Failure: {result.failure_category}")
                    if result.error_message:
                        print(f"  Error: {result.error_message[:100]}")

        self.generate_report()

    def generate_report(self):  # noqa: C901
        """Generate detailed evaluation report."""
        report_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_file, "w") as f:
            f.write("# Jetbox Agent Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary statistics
            total = len(self.results)
            passed = sum(1 for r in self.results if r.success)
            failed = total - passed

            f.write("## Summary\n\n")
            f.write(f"- **Total Tasks**: {total}\n")
            f.write(f"- **Passed**: {passed} ({100*passed/total:.1f}%)\n")
            f.write(f"- **Failed**: {failed} ({100*failed/total:.1f}%)\n\n")

            # By level
            f.write("## Results by Level\n\n")
            for level in range(1, 5):
                level_results = [r for r in self.results if r.task.level == level]
                if level_results:
                    level_passed = sum(1 for r in level_results if r.success)
                    f.write(f"- **Level {level}**: {level_passed}/{len(level_results)} passed\n")

            # Failure categories
            f.write("\n## Failure Categories\n\n")
            failure_counts = {}
            for result in self.results:
                if not result.success and result.failure_category:
                    failure_counts[result.failure_category] = failure_counts.get(result.failure_category, 0) + 1

            for category, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{category}**: {count} failures\n")

            # Detailed results
            f.write("\n## Detailed Results\n\n")
            for result in self.results:
                status = "✅ PASS" if result.success else "❌ FAIL"
                f.write(f"### {status} Level {result.task.level}: {result.task.name}\n\n")
                f.write(f"**Description**: {result.task.description}\n\n")
                f.write(f"**Duration**: {result.duration:.2f}s\n\n")

                if not result.success:
                    f.write(f"**Failure Category**: {result.failure_category}\n\n")
                    if result.error_message:
                        f.write(f"**Error Message**:\n```\n{result.error_message}\n```\n\n")
                    f.write(f"**Validation Output**:\n```\n{result.validation_output}\n```\n\n")

                f.write(f"**Files Created**: {', '.join(result.files_created) if result.files_created else 'None'}\n\n")
                f.write("---\n\n")

            # Insights
            f.write("## Insights\n\n")
            f.write("### Common Failure Patterns\n\n")
            # Analyze patterns
            if failed > 0:
                most_common = max(failure_counts.items(), key=lambda x: x[1]) if failure_counts else (None, 0)
                if most_common[0]:
                    f.write(f"- Most common failure: **{most_common[0]}** ({most_common[1]} occurrences)\n")

                # Level analysis
                level_4_results = [r for r in self.results if r.task.level == 4]
                if level_4_results:
                    level_4_failed = sum(1 for r in level_4_results if not r.success)
                    f.write(f"- Level 4 (Complex) tasks had {level_4_failed}/{len(level_4_results)} failures\n")

        print(f"\n📊 Report generated: {report_file}")
        return report_file


if __name__ == "__main__":
    suite = EvaluationSuite()

    # For now, just run setup/framework test (not actual agent execution)
    print("\n🧪 Evaluation Suite Framework Test")
    print("Note: This tests the evaluation framework setup, not actual agent execution.")
    print("To run full evaluation, you would call suite.run_suite() with a working agent.\n")

    # Show available tasks
    print(f"Total tasks defined: {len(suite.tasks)}")
    for level in range(1, 5):
        level_tasks = [t for t in suite.tasks if t.level == level]
        print(f"  Level {level}: {len(level_tasks)} tasks")

    print("\nTask Categories:")
    all_tags = set()
    for task in suite.tasks:
        all_tags.update(task.tags)
    for tag in sorted(all_tags):
        count = sum(1 for t in suite.tasks if tag in t.tags)
        print(f"  {tag}: {count} tasks")

    print("\n✅ Evaluation framework ready!")
    print("To run: suite.run_suite(max_level=2)  # Run levels 1-2")
