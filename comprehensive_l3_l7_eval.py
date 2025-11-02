#!/usr/bin/env python3
"""
Comprehensive L3-L7 Evaluation
- 5 levels (L3-L7)
- 3 problems per level
- 3 runs per problem
- Total: 45 test runs

Provides detailed statistics, failure analysis, and optimization insights.
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
import statistics

# Set model for testing
os.environ["OLLAMA_MODEL"] = "gpt-oss:20b"

from task_executor_agent import TaskExecutorAgent
from orchestrator_agent import OrchestratorAgent
from architect_agent import ArchitectAgent


@dataclass
class TestRun:
    """Single test run result"""
    run_number: int
    status: str  # success, failed, error
    duration: float
    rounds_used: int
    files_created: int
    files_expected: int
    errors: List[str] = field(default_factory=list)


@dataclass
class TestProblem:
    """Test problem with multiple runs"""
    level: str
    name: str
    description: str
    max_rounds: int
    expected_files: List[str]
    runs: List[TestRun] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return len([r for r in self.runs if r.status == 'success']) / len(self.runs)

    @property
    def avg_duration(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean([r.duration for r in self.runs])

    @property
    def avg_rounds(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean([r.rounds_used for r in self.runs])


class ComprehensiveEvaluator:
    """L3-L7 comprehensive evaluation"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.problems: List[TestProblem] = []
        self.start_time = time.time()

    def define_test_problems(self):
        """Define all test problems L3-L7"""

        # L3: Multi-file packages (TaskExecutor)
        self.problems.extend([
            TestProblem(
                level="L3",
                name="Math Package",
                description="Create mathx package with add/subtract/multiply/divide in separate files, with tests",
                max_rounds=20,
                expected_files=["mathx/__init__.py", "mathx/add.py", "mathx/subtract.py", "mathx/multiply.py", "mathx/divide.py", "tests/test_mathx.py"]
            ),
            TestProblem(
                level="L3",
                name="String Utils Package",
                description="Create strutils package with capitalize/reverse/count_words functions in separate files, with tests",
                max_rounds=20,
                expected_files=["strutils/__init__.py", "strutils/capitalize.py", "strutils/reverse.py", "strutils/count_words.py", "tests/test_strutils.py"]
            ),
            TestProblem(
                level="L3",
                name="File Utils Package",
                description="Create fileutils package with read/write/append/delete functions in separate files, with tests",
                max_rounds=20,
                expected_files=["fileutils/__init__.py", "fileutils/read.py", "fileutils/write.py", "fileutils/append.py", "fileutils/delete.py", "tests/test_fileutils.py"]
            ),
        ])

        # L4: Packages with dependencies (TaskExecutor)
        self.problems.extend([
            TestProblem(
                level="L4",
                name="HTTP Wrapper",
                description="Create http_wrapper package with retry logic and timeout handling. Include tests.",
                max_rounds=25,
                expected_files=["http_wrapper/__init__.py", "http_wrapper/client.py", "tests/test_client.py"]
            ),
            TestProblem(
                level="L4",
                name="JSON Validator",
                description="Create json_validator package that validates JSON schemas. Include tests with sample schemas.",
                max_rounds=25,
                expected_files=["json_validator/__init__.py", "json_validator/validator.py", "tests/test_validator.py"]
            ),
            TestProblem(
                level="L4",
                name="CSV Parser",
                description="Create csv_parser package with header detection and type inference. Include tests.",
                max_rounds=25,
                expected_files=["csv_parser/__init__.py", "csv_parser/parser.py", "tests/test_parser.py"]
            ),
        ])

        # L5: Simple orchestration (Orchestrator + TaskExecutor)
        self.problems.extend([
            TestProblem(
                level="L5",
                name="Web API",
                description="Create a simple REST API with Flask: user CRUD endpoints, with tests",
                max_rounds=30,
                expected_files=["app.py", "tests/test_api.py", "requirements.txt"]
            ),
            TestProblem(
                level="L5",
                name="CLI Tool",
                description="Create a CLI tool with argparse: file converter (JSON<->CSV), with tests",
                max_rounds=30,
                expected_files=["cli.py", "converter.py", "tests/test_cli.py", "tests/test_converter.py"]
            ),
            TestProblem(
                level="L5",
                name="Data Pipeline",
                description="Create a data processing pipeline: read CSV, transform, write JSON, with tests",
                max_rounds=30,
                expected_files=["pipeline.py", "transforms.py", "tests/test_pipeline.py"]
            ),
        ])

        # L6: Architecture + implementation (Orchestrator + Architect + TaskExecutor)
        self.problems.extend([
            TestProblem(
                level="L6",
                name="Microservice",
                description="Design and implement a user microservice with database layer, API, and tests",
                max_rounds=40,
                expected_files=["architecture.md", "app.py", "models.py", "api.py", "tests/test_api.py"]
            ),
            TestProblem(
                level="L6",
                name="Plugin System",
                description="Design and implement a plugin system with dynamic loading, hooks, and tests",
                max_rounds=40,
                expected_files=["architecture.md", "plugin_manager.py", "plugin_base.py", "tests/test_plugins.py"]
            ),
            TestProblem(
                level="L6",
                name="Event Bus",
                description="Design and implement an event bus with pub/sub, filtering, and tests",
                max_rounds=40,
                expected_files=["architecture.md", "event_bus.py", "subscriber.py", "tests/test_event_bus.py"]
            ),
        ])

        # L7: Complex multi-agent workflows
        self.problems.extend([
            TestProblem(
                level="L7",
                name="Full Stack App",
                description="Design and build a task tracker: backend API + frontend + database schema + tests",
                max_rounds=50,
                expected_files=["architecture.md", "backend/app.py", "backend/models.py", "frontend/index.html", "tests/test_backend.py"]
            ),
            TestProblem(
                level="L7",
                name="Distributed System",
                description="Design and implement a distributed key-value store with replication and tests",
                max_rounds=50,
                expected_files=["architecture.md", "node.py", "replication.py", "client.py", "tests/test_replication.py"]
            ),
            TestProblem(
                level="L7",
                name="Message Queue",
                description="Design and build a message queue system with persistence, acknowledgments, and tests",
                max_rounds=50,
                expected_files=["architecture.md", "queue.py", "broker.py", "persistence.py", "tests/test_queue.py"]
            ),
        ])

    def run_test(self, problem: TestProblem, run_num: int) -> TestRun:
        """Run a single test"""
        print(f"\n  Run {run_num}/3: {problem.name}")

        workspace = Path(f".agent_workspace/eval_{problem.level.lower()}_{problem.name.lower().replace(' ', '_')}_run{run_num}")

        start_time = time.time()
        errors = []

        try:
            # Determine agent type based on level
            if problem.level in ["L3", "L4"]:
                # Direct TaskExecutor
                agent = TaskExecutorAgent(workspace=workspace, goal=problem.description)
                result = agent.run(max_rounds=problem.max_rounds)

            elif problem.level == "L5":
                # Orchestrator + TaskExecutor
                agent = OrchestratorAgent(workspace=workspace)
                # Simulate user message
                agent.add_message({"role": "user", "content": problem.description})
                result = agent.run(max_rounds=problem.max_rounds)

            elif problem.level in ["L6", "L7"]:
                # Full stack: Orchestrator + Architect + TaskExecutor
                agent = OrchestratorAgent(workspace=workspace)
                agent.add_message({"role": "user", "content": problem.description})
                result = agent.run(max_rounds=problem.max_rounds)

            else:
                raise ValueError(f"Unknown level: {problem.level}")

            # Check files created
            files_created = 0
            for expected_file in problem.expected_files:
                if (workspace / expected_file).exists():
                    files_created += 1

            # Get performance data
            perf_stats = agent.perf_stats if hasattr(agent, 'perf_stats') else None
            rounds_used = len(perf_stats.llm_call_times) if perf_stats and hasattr(perf_stats, 'llm_call_times') else 0

            duration = time.time() - start_time
            status = result.get('status', 'unknown')

            run_result = TestRun(
                run_number=run_num,
                status=status,
                duration=duration,
                rounds_used=rounds_used,
                files_created=files_created,
                files_expected=len(problem.expected_files),
                errors=errors
            )

            # Display result
            icon = "✅" if status == "success" else "❌"
            print(f"    {icon} {status} - {duration:.1f}s, {rounds_used} rounds, {files_created}/{len(problem.expected_files)} files")

            return run_result

        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)}")
            duration = time.time() - start_time

            print(f"    ❌ ERROR - {type(e).__name__}: {str(e)}")

            return TestRun(
                run_number=run_num,
                status="error",
                duration=duration,
                rounds_used=0,
                files_created=0,
                files_expected=len(problem.expected_files),
                errors=errors
            )

    def run_all_tests(self):
        """Run all test problems"""
        print("="*80)
        print("COMPREHENSIVE L3-L7 EVALUATION")
        print("="*80)
        print(f"Model: {os.environ.get('OLLAMA_MODEL', 'default')}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total tests: {len(self.problems)} problems × 3 runs = {len(self.problems) * 3} runs")
        print("="*80)

        for level in ["L3", "L4", "L5", "L6", "L7"]:
            level_problems = [p for p in self.problems if p.level == level]

            print(f"\n{'='*80}")
            print(f"{level}: {level_problems[0].description.split(':')[0] if level_problems else 'Unknown'}")
            print(f"{'='*80}")

            for problem in level_problems:
                print(f"\n[{problem.level}] {problem.name}")
                print(f"  Goal: {problem.description}")
                print(f"  Expected files: {len(problem.expected_files)}")

                # Run 3 times
                for run_num in range(1, 4):
                    run_result = self.run_test(problem, run_num)
                    problem.runs.append(run_result)

                # Display summary for this problem
                success_rate = problem.success_rate
                print(f"\n  Summary: {success_rate*100:.0f}% success ({sum(1 for r in problem.runs if r.status=='success')}/3)")
                print(f"  Avg duration: {problem.avg_duration:.1f}s")
                print(f"  Avg rounds: {problem.avg_rounds:.1f}")

    def generate_report(self):
        """Generate comprehensive report"""
        total_duration = time.time() - self.start_time
        total_runs = sum(len(p.runs) for p in self.problems)
        successful_runs = sum(len([r for r in p.runs if r.status == 'success']) for p in self.problems)

        # Calculate per-level statistics
        level_stats = {}
        for level in ["L3", "L4", "L5", "L6", "L7"]:
            level_problems = [p for p in self.problems if p.level == level]
            if level_problems:
                level_runs = [r for p in level_problems for r in p.runs]
                level_stats[level] = {
                    'problems': len(level_problems),
                    'runs': len(level_runs),
                    'success_rate': len([r for r in level_runs if r.status == 'success']) / len(level_runs) if level_runs else 0,
                    'avg_duration': statistics.mean([r.duration for r in level_runs]) if level_runs else 0,
                    'avg_rounds': statistics.mean([r.rounds_used for r in level_runs]) if level_runs else 0,
                }

        # Generate report
        report = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'total_duration': total_duration,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
                'total_problems': len(self.problems),
            },
            'level_statistics': level_stats,
            'problems': [
                {
                    'level': p.level,
                    'name': p.name,
                    'description': p.description,
                    'success_rate': p.success_rate,
                    'avg_duration': p.avg_duration,
                    'avg_rounds': p.avg_rounds,
                    'runs': [asdict(r) for r in p.runs]
                }
                for p in self.problems
            ]
        }

        # Write JSON report
        json_path = self.output_dir / f"l3_l7_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Write markdown report
        md_path = self.output_dir / f"l3_l7_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(md_path, 'w') as f:
            self._write_markdown_report(f, report)

        # Display summary
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total runs: {total_runs}")
        print(f"Successful: {successful_runs} ({report['summary']['success_rate']*100:.1f}%)")
        print(f"Duration: {total_duration/60:.1f} minutes")
        print(f"\nPer-level success rates:")
        for level in ["L3", "L4", "L5", "L6", "L7"]:
            if level in level_stats:
                stats = level_stats[level]
                print(f"  {level}: {stats['success_rate']*100:.0f}% ({stats['runs']} runs, avg {stats['avg_duration']:.1f}s)")
        print(f"\nReports written to:")
        print(f"  - {json_path}")
        print(f"  - {md_path}")

        return report

    def _write_markdown_report(self, f, report: Dict):
        """Write markdown report"""
        f.write("# L3-L7 Comprehensive Evaluation Report\n\n")

        # Summary
        f.write("## Executive Summary\n\n")
        summary = report['summary']
        f.write(f"**Timestamp**: {summary['timestamp']}\n")
        f.write(f"**Total Duration**: {summary['total_duration']/60:.1f} minutes\n")
        f.write(f"**Total Runs**: {summary['total_runs']}\n")
        f.write(f"**Success Rate**: {summary['success_rate']*100:.1f}%\n")
        f.write(f"**Problems Tested**: {summary['total_problems']}\n\n")

        # Per-level statistics
        f.write("## Per-Level Statistics\n\n")
        f.write("| Level | Problems | Runs | Success Rate | Avg Duration | Avg Rounds |\n")
        f.write("|-------|----------|------|--------------|--------------|------------|\n")
        for level in ["L3", "L4", "L5", "L6", "L7"]:
            if level in report['level_statistics']:
                stats = report['level_statistics'][level]
                f.write(f"| {level} | {stats['problems']} | {stats['runs']} | {stats['success_rate']*100:.0f}% | {stats['avg_duration']:.1f}s | {stats['avg_rounds']:.1f} |\n")
        f.write("\n")

        # Detailed results
        f.write("## Detailed Results\n\n")
        for level in ["L3", "L4", "L5", "L6", "L7"]:
            level_problems = [p for p in report['problems'] if p['level'] == level]
            if level_problems:
                f.write(f"### {level}\n\n")
                for problem in level_problems:
                    icon = "✅" if problem['success_rate'] >= 0.67 else "⚠️" if problem['success_rate'] > 0 else "❌"
                    f.write(f"#### {icon} {problem['name']}\n\n")
                    f.write(f"**Description**: {problem['description']}\n\n")
                    f.write(f"**Success Rate**: {problem['success_rate']*100:.0f}% ({sum(1 for r in problem['runs'] if r['status']=='success')}/3)\n\n")
                    f.write(f"**Avg Duration**: {problem['avg_duration']:.1f}s\n\n")
                    f.write(f"**Avg Rounds**: {problem['avg_rounds']:.1f}\n\n")

                    # Run details
                    f.write("**Runs**:\n")
                    for run in problem['runs']:
                        run_icon = "✅" if run['status'] == 'success' else "❌"
                        f.write(f"- Run {run['run_number']}: {run_icon} {run['status']} ({run['duration']:.1f}s, {run['rounds_used']} rounds, {run['files_created']}/{run['files_expected']} files)\n")
                    f.write("\n")


def main():
    evaluator = ComprehensiveEvaluator(output_dir=Path("evaluation_results"))

    # Define test problems
    evaluator.define_test_problems()

    # Run all tests
    evaluator.run_all_tests()

    # Generate report
    report = evaluator.generate_report()

    # Return exit code based on success rate
    success_rate = report['summary']['success_rate']
    return 0 if success_rate >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
