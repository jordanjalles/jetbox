#!/usr/bin/env python3
"""
Enhanced L3-L7 Comprehensive Evaluation with Performance Breakdown

Performance tracking:
- LLM thinking time
- Tool execution time
- Idle/hanging time
- Context management overhead
- Round-by-round timing
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
class PerformanceBreakdown:
    """Detailed performance breakdown"""
    total_time: float = 0.0
    llm_thinking_time: float = 0.0  # Total time in LLM calls
    tool_execution_time: float = 0.0  # Total time executing tools
    context_management_time: float = 0.0  # Time building/compacting context
    idle_time: float = 0.0  # Time between rounds (overhead)

    # Per-round timing
    round_times: List[float] = field(default_factory=list)
    llm_call_times: List[float] = field(default_factory=list)
    tool_call_times: List[float] = field(default_factory=list)

    # Tool breakdown
    tool_time_breakdown: Dict[str, float] = field(default_factory=dict)
    tool_call_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def overhead_percentage(self) -> float:
        """Percentage of time spent on overhead (not LLM or tools)"""
        if self.total_time == 0:
            return 0.0
        productive_time = self.llm_thinking_time + self.tool_execution_time
        overhead = self.total_time - productive_time
        return (overhead / self.total_time) * 100

    @property
    def llm_percentage(self) -> float:
        """Percentage of time in LLM"""
        if self.total_time == 0:
            return 0.0
        return (self.llm_thinking_time / self.total_time) * 100

    @property
    def tool_percentage(self) -> float:
        """Percentage of time in tools"""
        if self.total_time == 0:
            return 0.0
        return (self.tool_execution_time / self.total_time) * 100


@dataclass
class TestRun:
    """Single test run with performance details"""
    run_number: int
    status: str
    duration: float
    rounds_used: int
    files_created: int
    files_expected: int
    performance: PerformanceBreakdown = field(default_factory=PerformanceBreakdown)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TestProblem:
    """Test problem with statistics"""
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
    def avg_llm_time(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean([r.performance.llm_thinking_time for r in self.runs])

    @property
    def avg_tool_time(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean([r.performance.tool_execution_time for r in self.runs])

    @property
    def avg_overhead(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean([r.performance.overhead_percentage for r in self.runs])


class EnhancedEvaluator:
    """Enhanced evaluator with detailed performance tracking"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.problems: List[TestProblem] = []
        self.start_time = time.time()

    def define_test_problems(self):
        """Define all test problems L3-L7"""

        # L3: Multi-file packages
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
                name="String Utils",
                description="Create strutils package with capitalize/reverse/count_words in separate files, with tests",
                max_rounds=20,
                expected_files=["strutils/__init__.py", "strutils/capitalize.py", "strutils/reverse.py", "strutils/count_words.py", "tests/test_strutils.py"]
            ),
            TestProblem(
                level="L3",
                name="File Utils",
                description="Create fileutils package with read/write/append/delete in separate files, with tests",
                max_rounds=20,
                expected_files=["fileutils/__init__.py", "fileutils/read.py", "fileutils/write.py", "fileutils/append.py", "fileutils/delete.py", "tests/test_fileutils.py"]
            ),
        ])

        # L4: Packages with dependencies
        self.problems.extend([
            TestProblem(
                level="L4",
                name="HTTP Wrapper",
                description="Create http_wrapper with retry logic and timeout. Include tests.",
                max_rounds=25,
                expected_files=["http_wrapper/__init__.py", "http_wrapper/client.py", "tests/test_client.py"]
            ),
            TestProblem(
                level="L4",
                name="JSON Validator",
                description="Create json_validator for schema validation. Include tests with sample schemas.",
                max_rounds=25,
                expected_files=["json_validator/__init__.py", "json_validator/validator.py", "tests/test_validator.py"]
            ),
            TestProblem(
                level="L4",
                name="CSV Parser",
                description="Create csv_parser with header detection and type inference. Include tests.",
                max_rounds=25,
                expected_files=["csv_parser/__init__.py", "csv_parser/parser.py", "tests/test_parser.py"]
            ),
        ])

        # L5: Orchestration
        self.problems.extend([
            TestProblem(
                level="L5",
                name="Web API",
                description="Create REST API with Flask: user CRUD endpoints, with tests",
                max_rounds=30,
                expected_files=["app.py", "tests/test_api.py", "requirements.txt"]
            ),
            TestProblem(
                level="L5",
                name="CLI Tool",
                description="Create CLI with argparse: file converter (JSON<->CSV), with tests",
                max_rounds=30,
                expected_files=["cli.py", "converter.py", "tests/test_cli.py"]
            ),
            TestProblem(
                level="L5",
                name="Data Pipeline",
                description="Create data pipeline: read CSV, transform, write JSON, with tests",
                max_rounds=30,
                expected_files=["pipeline.py", "transforms.py", "tests/test_pipeline.py"]
            ),
        ])

        # L6: Architecture + implementation
        self.problems.extend([
            TestProblem(
                level="L6",
                name="Microservice",
                description="Design and implement user microservice with database, API, tests",
                max_rounds=40,
                expected_files=["architecture.md", "app.py", "models.py", "tests/test_api.py"]
            ),
            TestProblem(
                level="L6",
                name="Plugin System",
                description="Design and implement plugin system with dynamic loading, hooks, tests",
                max_rounds=40,
                expected_files=["architecture.md", "plugin_manager.py", "plugin_base.py", "tests/test_plugins.py"]
            ),
            TestProblem(
                level="L6",
                name="Event Bus",
                description="Design and implement event bus with pub/sub, filtering, tests",
                max_rounds=40,
                expected_files=["architecture.md", "event_bus.py", "subscriber.py", "tests/test_event_bus.py"]
            ),
        ])

        # L7: Complex workflows
        self.problems.extend([
            TestProblem(
                level="L7",
                name="Full Stack App",
                description="Design and build task tracker: backend API + frontend + database + tests",
                max_rounds=50,
                expected_files=["architecture.md", "backend/app.py", "frontend/index.html", "tests/test_backend.py"]
            ),
            TestProblem(
                level="L7",
                name="Distributed System",
                description="Design and implement distributed KV store with replication, tests",
                max_rounds=50,
                expected_files=["architecture.md", "node.py", "replication.py", "tests/test_replication.py"]
            ),
            TestProblem(
                level="L7",
                name="Message Queue",
                description="Design and build message queue with persistence, acknowledgments, tests",
                max_rounds=50,
                expected_files=["architecture.md", "queue.py", "broker.py", "tests/test_queue.py"]
            ),
        ])

    def run_test(self, problem: TestProblem, run_num: int) -> TestRun:
        """Run test with detailed performance tracking"""
        print(f"\n  Run {run_num}/3: {problem.name}")

        workspace = Path(f".agent_workspace/eval_{problem.level.lower()}_{problem.name.lower().replace(' ', '_')}_run{run_num}")

        test_start = time.time()
        errors = []
        warnings = []
        performance = PerformanceBreakdown()

        try:
            # Create agent
            if problem.level in ["L3", "L4"]:
                agent = TaskExecutorAgent(workspace=workspace, goal=problem.description)
            elif problem.level == "L5":
                agent = OrchestratorAgent(workspace=workspace)
                agent.add_message({"role": "user", "content": problem.description})
            else:  # L6, L7
                agent = OrchestratorAgent(workspace=workspace)
                agent.add_message({"role": "user", "content": problem.description})

            # Run agent
            run_start = time.time()
            result = agent.run(max_rounds=problem.max_rounds)
            run_duration = time.time() - run_start

            # Extract performance data from agent
            perf_stats = agent.perf_stats if hasattr(agent, 'perf_stats') else None

            if perf_stats and hasattr(perf_stats, 'llm_call_times'):
                performance.llm_call_times = list(perf_stats.llm_call_times)
                performance.llm_thinking_time = sum(perf_stats.llm_call_times)

            if perf_stats and hasattr(perf_stats, 'tool_call_times'):
                performance.tool_call_times = list(perf_stats.tool_call_times)
                performance.tool_execution_time = sum(perf_stats.tool_call_times)

            # Calculate overhead
            performance.total_time = run_duration
            productive_time = performance.llm_thinking_time + performance.tool_execution_time
            performance.idle_time = max(0, run_duration - productive_time)

            # Check files
            files_created = sum(1 for f in problem.expected_files if (workspace / f).exists())

            # Get rounds used
            rounds_used = len(performance.llm_call_times)

            # Detect performance issues
            if performance.overhead_percentage > 30:
                warnings.append(f"High overhead: {performance.overhead_percentage:.1f}%")

            if performance.idle_time > 10:
                warnings.append(f"Long idle time: {performance.idle_time:.1f}s")

            test_duration = time.time() - test_start
            status = result.get('status', 'unknown')

            run_result = TestRun(
                run_number=run_num,
                status=status,
                duration=test_duration,
                rounds_used=rounds_used,
                files_created=files_created,
                files_expected=len(problem.expected_files),
                performance=performance,
                errors=errors,
                warnings=warnings
            )

            # Display result
            icon = "✅" if status == "success" else "❌"
            print(f"    {icon} {status} - {test_duration:.1f}s ({performance.llm_percentage:.0f}% LLM, {performance.tool_percentage:.0f}% tools, {performance.overhead_percentage:.0f}% overhead)")
            print(f"       {rounds_used} rounds, {files_created}/{len(problem.expected_files)} files")

            if warnings:
                for w in warnings:
                    print(f"       ⚠️  {w}")

            return run_result

        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)}")
            test_duration = time.time() - test_start

            print(f"    ❌ ERROR - {type(e).__name__}: {str(e)}")

            return TestRun(
                run_number=run_num,
                status="error",
                duration=test_duration,
                rounds_used=0,
                files_created=0,
                files_expected=len(problem.expected_files),
                performance=performance,
                errors=errors
            )

    def run_all_tests(self):
        """Run all tests"""
        print("="*80)
        print("ENHANCED L3-L7 EVALUATION WITH PERFORMANCE TRACKING")
        print("="*80)
        print(f"Model: {os.environ.get('OLLAMA_MODEL', 'default')}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total: {len(self.problems)} problems × 3 runs = {len(self.problems) * 3} tests")
        print("="*80)

        for level in ["L3", "L4", "L5", "L6", "L7"]:
            level_problems = [p for p in self.problems if p.level == level]

            if level_problems:
                print(f"\n{'='*80}")
                print(f"{level}: {self._get_level_description(level)}")
                print(f"{'='*80}")

                for problem in level_problems:
                    print(f"\n[{problem.level}] {problem.name}")
                    print(f"  Goal: {problem.description}")

                    # Run 3 times
                    for run_num in range(1, 4):
                        run_result = self.run_test(problem, run_num)
                        problem.runs.append(run_result)

                    # Problem summary
                    print(f"\n  📊 Summary: {problem.success_rate*100:.0f}% success")
                    print(f"     Duration: {problem.avg_duration:.1f}s avg")
                    print(f"     Performance: {problem.avg_llm_time:.1f}s LLM, {problem.avg_tool_time:.1f}s tools, {problem.avg_overhead:.1f}% overhead")

    def _get_level_description(self, level: str) -> str:
        """Get level description"""
        descriptions = {
            "L3": "Multi-file packages",
            "L4": "Packages with dependencies",
            "L5": "Simple orchestration",
            "L6": "Architecture + implementation",
            "L7": "Complex multi-agent workflows"
        }
        return descriptions.get(level, "Unknown")

    def generate_report(self):
        """Generate comprehensive report with performance breakdown"""
        total_duration = time.time() - self.start_time
        total_runs = sum(len(p.runs) for p in self.problems)
        successful_runs = sum(len([r for r in p.runs if r.status == 'success']) for p in self.problems)

        # Per-level statistics
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
                    'avg_llm_time': statistics.mean([r.performance.llm_thinking_time for r in level_runs]) if level_runs else 0,
                    'avg_tool_time': statistics.mean([r.performance.tool_execution_time for r in level_runs]) if level_runs else 0,
                    'avg_overhead': statistics.mean([r.performance.overhead_percentage for r in level_runs]) if level_runs else 0,
                }

        # Generate report
        report = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'total_duration_minutes': total_duration / 60,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
                'total_problems': len(self.problems),
            },
            'level_statistics': level_stats,
            'problems': [self._problem_to_dict(p) for p in self.problems]
        }

        # Write reports
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.output_dir / f"l3_l7_enhanced_{timestamp}.json"
        md_path = self.output_dir / f"l3_l7_enhanced_{timestamp}.md"

        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        with open(md_path, 'w') as f:
            self._write_markdown(f, report)

        # Display summary
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Duration: {total_duration/60:.1f} minutes")
        print(f"Success Rate: {report['summary']['success_rate']*100:.1f}% ({successful_runs}/{total_runs})")
        print(f"\nPer-Level Results:")
        for level in ["L3", "L4", "L5", "L6", "L7"]:
            if level in level_stats:
                s = level_stats[level]
                print(f"  {level}: {s['success_rate']*100:.0f}% success | {s['avg_duration']:.1f}s avg | {s['avg_llm_time']:.1f}s LLM | {s['avg_tool_time']:.1f}s tools | {s['avg_overhead']:.1f}% overhead")
        print(f"\nReports:")
        print(f"  JSON: {json_path}")
        print(f"  MD:   {md_path}")

        return report

    def _problem_to_dict(self, p: TestProblem) -> Dict:
        """Convert problem to dict for JSON"""
        return {
            'level': p.level,
            'name': p.name,
            'description': p.description,
            'success_rate': p.success_rate,
            'avg_duration': p.avg_duration,
            'avg_llm_time': p.avg_llm_time,
            'avg_tool_time': p.avg_tool_time,
            'avg_overhead': p.avg_overhead,
            'runs': [self._run_to_dict(r) for r in p.runs]
        }

    def _run_to_dict(self, r: TestRun) -> Dict:
        """Convert run to dict for JSON"""
        return {
            'run_number': r.run_number,
            'status': r.status,
            'duration': r.duration,
            'rounds_used': r.rounds_used,
            'files_created': r.files_created,
            'files_expected': r.files_expected,
            'performance': {
                'total_time': r.performance.total_time,
                'llm_thinking_time': r.performance.llm_thinking_time,
                'tool_execution_time': r.performance.tool_execution_time,
                'idle_time': r.performance.idle_time,
                'llm_percentage': r.performance.llm_percentage,
                'tool_percentage': r.performance.tool_percentage,
                'overhead_percentage': r.performance.overhead_percentage,
            },
            'errors': r.errors,
            'warnings': r.warnings
        }

    def _write_markdown(self, f, report: Dict):
        """Write markdown report"""
        f.write("# L3-L7 Enhanced Evaluation Report\n\n")
        f.write("## Executive Summary\n\n")

        summary = report['summary']
        f.write(f"**Duration**: {summary['total_duration_minutes']:.1f} minutes\n")
        f.write(f"**Success Rate**: {summary['success_rate']*100:.1f}% ({summary['successful_runs']}/{summary['total_runs']})\n")
        f.write(f"**Problems**: {summary['total_problems']}\n\n")

        f.write("## Performance Breakdown by Level\n\n")
        f.write("| Level | Success | Avg Time | LLM Time | Tool Time | Overhead |\n")
        f.write("|-------|---------|----------|----------|-----------|----------|\n")
        for level in ["L3", "L4", "L5", "L6", "L7"]:
            if level in report['level_statistics']:
                s = report['level_statistics'][level]
                f.write(f"| {level} | {s['success_rate']*100:.0f}% | {s['avg_duration']:.1f}s | {s['avg_llm_time']:.1f}s | {s['avg_tool_time']:.1f}s | {s['avg_overhead']:.1f}% |\n")
        f.write("\n")

        f.write("## Detailed Results\n\n")
        for problem in report['problems']:
            icon = "✅" if problem['success_rate'] >= 0.67 else "⚠️" if problem['success_rate'] > 0 else "❌"
            f.write(f"### {icon} [{problem['level']}] {problem['name']}\n\n")
            f.write(f"**Success**: {problem['success_rate']*100:.0f}%\n")
            f.write(f"**Avg Duration**: {problem['avg_duration']:.1f}s\n")
            f.write(f"**Performance**: {problem['avg_llm_time']:.1f}s LLM, {problem['avg_tool_time']:.1f}s tools, {problem['avg_overhead']:.1f}% overhead\n\n")

            for run in problem['runs']:
                run_icon = "✅" if run['status'] == 'success' else "❌"
                perf = run['performance']
                f.write(f"- Run {run['run_number']}: {run_icon} {run['status']} | {run['duration']:.1f}s | {perf['llm_percentage']:.0f}% LLM, {perf['tool_percentage']:.0f}% tools, {perf['overhead_percentage']:.0f}% overhead\n")
            f.write("\n")


def main():
    evaluator = EnhancedEvaluator(output_dir=Path("evaluation_results"))
    evaluator.define_test_problems()
    evaluator.run_all_tests()
    report = evaluator.generate_report()

    return 0 if report['summary']['success_rate'] >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
