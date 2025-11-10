#!/usr/bin/env python3
"""
Targeted L5-L7 Re-evaluation After Delegation Fix

Tests orchestration levels to validate delegation bug fix.
- L5: Orchestrator + TaskExecutor (3 problems × 3 runs = 9 tests)
- L6: Orchestrator + Architect + TaskExecutor (3 problems × 3 runs = 9 tests)
- L7: Complex workflows (3 problems × 3 runs = 9 tests)
Total: 27 tests
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict
import statistics

os.environ["OLLAMA_MODEL"] = "gpt-oss:20b"

from agents.orchestrator_agent import OrchestratorAgent


@dataclass
class TestRun:
    """Single test run result"""
    run_number: int
    status: str
    duration: float
    files_created: int
    files_expected: int
    errors: List[str] = field(default_factory=list)


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


class L5L7Evaluator:
    """Targeted L5-L7 evaluation"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.problems: List[TestProblem] = []
        self.start_time = time.time()

    def define_problems(self):
        """Define L5-L7 test problems"""

        # L5: Simple orchestration
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
        """Run single test"""
        print(f"\n  Run {run_num}/3: {problem.name}")

        workspace = Path(f".agent_workspaces/rerun_{problem.level.lower()}_{problem.name.lower().replace(' ', '_')}_run{run_num}")

        start_time = time.time()
        errors = []

        try:
            # Create orchestrator
            agent = OrchestratorAgent(workspace=workspace)
            agent.add_message({"role": "user", "content": problem.description})

            # Run
            result = agent.run(max_rounds=problem.max_rounds)

            # Check files
            files_created = sum(1 for f in problem.expected_files if (workspace / f).exists())

            duration = time.time() - start_time
            status = result.get('status', 'unknown')

            run_result = TestRun(
                run_number=run_num,
                status=status,
                duration=duration,
                files_created=files_created,
                files_expected=len(problem.expected_files),
                errors=errors
            )

            icon = "✅" if status == "success" else "❌"
            print(f"    {icon} {status} - {duration:.1f}s, {files_created}/{len(problem.expected_files)} files")

            return run_result

        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)}")
            duration = time.time() - start_time

            print(f"    ❌ ERROR - {type(e).__name__}: {str(e)}")

            return TestRun(
                run_number=run_num,
                status="error",
                duration=duration,
                files_created=0,
                files_expected=len(problem.expected_files),
                errors=errors
            )

    def run_all(self):
        """Run all tests"""
        print("="*80)
        print("L5-L7 RE-EVALUATION (POST DELEGATION FIX)")
        print("="*80)
        print(f"Model: {os.environ.get('OLLAMA_MODEL', 'default')}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Tests: 9 problems × 3 runs = 27 tests")
        print("="*80)

        for level in ["L5", "L6", "L7"]:
            level_problems = [p for p in self.problems if p.level == level]

            if level_problems:
                print(f"\n{'='*80}")
                print(f"{level}: {self._get_level_desc(level)}")
                print(f"{'='*80}")

                for problem in level_problems:
                    print(f"\n[{problem.level}] {problem.name}")
                    print(f"  Goal: {problem.description}")

                    for run_num in range(1, 4):
                        run_result = self.run_test(problem, run_num)
                        problem.runs.append(run_result)

                    # Summary
                    print(f"\n  📊 {problem.success_rate*100:.0f}% success ({sum(1 for r in problem.runs if r.status=='success')}/3)")
                    print(f"     {problem.avg_duration:.1f}s avg")

    def _get_level_desc(self, level: str) -> str:
        return {
            "L5": "Simple orchestration",
            "L6": "Architecture + implementation",
            "L7": "Complex multi-agent workflows"
        }.get(level, "Unknown")

    def generate_report(self):
        """Generate report"""
        total_duration = time.time() - self.start_time
        total_runs = sum(len(p.runs) for p in self.problems)
        successful = sum(len([r for r in p.runs if r.status == 'success']) for p in self.problems)

        # Per-level stats
        level_stats = {}
        for level in ["L5", "L6", "L7"]:
            level_problems = [p for p in self.problems if p.level == level]
            if level_problems:
                level_runs = [r for p in level_problems for r in p.runs]
                level_stats[level] = {
                    'problems': len(level_problems),
                    'runs': len(level_runs),
                    'success_rate': len([r for r in level_runs if r.status == 'success']) / len(level_runs) if level_runs else 0,
                    'avg_duration': statistics.mean([r.duration for r in level_runs]) if level_runs else 0,
                }

        report = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'total_duration_minutes': total_duration / 60,
                'total_runs': total_runs,
                'successful_runs': successful,
                'success_rate': successful / total_runs if total_runs > 0 else 0,
            },
            'level_statistics': level_stats,
            'problems': [
                {
                    'level': p.level,
                    'name': p.name,
                    'success_rate': p.success_rate,
                    'avg_duration': p.avg_duration,
                    'runs': [
                        {
                            'run_number': r.run_number,
                            'status': r.status,
                            'duration': r.duration,
                            'files_created': r.files_created,
                            'files_expected': r.files_expected
                        }
                        for r in p.runs
                    ]
                }
                for p in self.problems
            ]
        }

        # Write reports
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.output_dir / f"l5_l7_rerun_{timestamp}.json"
        md_path = self.output_dir / f"l5_l7_rerun_{timestamp}.md"

        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        with open(md_path, 'w') as f:
            self._write_md(f, report)

        # Display summary
        print(f"\n{'='*80}")
        print("RE-EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Duration: {total_duration/60:.1f} minutes")
        print(f"Success Rate: {report['summary']['success_rate']*100:.1f}% ({successful}/{total_runs})")
        print("\nPer-Level Results:")
        for level in ["L5", "L6", "L7"]:
            if level in level_stats:
                s = level_stats[level]
                print(f"  {level}: {s['success_rate']*100:.0f}% ({int(s['success_rate']*s['runs'])}/{s['runs']} tests)")
        print("\nReports:")
        print(f"  JSON: {json_path}")
        print(f"  MD:   {md_path}")

        return report

    def _write_md(self, f, report: Dict):
        """Write markdown report"""
        f.write("# L5-L7 Re-Evaluation Results (Post Delegation Fix)\n\n")
        f.write("## Summary\n\n")

        summary = report['summary']
        f.write(f"**Duration**: {summary['total_duration_minutes']:.1f} minutes\n")
        f.write(f"**Success Rate**: {summary['success_rate']*100:.1f}% ({summary['successful_runs']}/{summary['total_runs']})\n\n")

        f.write("## Per-Level Results\n\n")
        f.write("| Level | Success Rate | Tests Passed |\n")
        f.write("|-------|--------------|-------------|\n")
        for level in ["L5", "L6", "L7"]:
            if level in report['level_statistics']:
                s = report['level_statistics'][level]
                passed = int(s['success_rate'] * s['runs'])
                f.write(f"| {level} | {s['success_rate']*100:.0f}% | {passed}/{s['runs']} |\n")
        f.write("\n")

        f.write("## Detailed Results\n\n")
        for problem in report['problems']:
            icon = "✅" if problem['success_rate'] >= 0.67 else "⚠️" if problem['success_rate'] > 0 else "❌"
            f.write(f"### {icon} [{problem['level']}] {problem['name']}\n\n")
            f.write(f"**Success**: {problem['success_rate']*100:.0f}%\n")
            f.write(f"**Avg Duration**: {problem['avg_duration']:.1f}s\n\n")

            for run in problem['runs']:
                run_icon = "✅" if run['status'] == 'success' else "❌"
                f.write(f"- Run {run['run_number']}: {run_icon} {run['status']} | {run['duration']:.1f}s | {run['files_created']}/{run['files_expected']} files\n")
            f.write("\n")


def main():
    evaluator = L5L7Evaluator(output_dir=Path("evaluation_results"))
    evaluator.define_problems()
    evaluator.run_all()
    report = evaluator.generate_report()

    return 0 if report['summary']['success_rate'] >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
