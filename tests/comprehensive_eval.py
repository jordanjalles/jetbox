#!/usr/bin/env python3
"""
Comprehensive Three-Level Evaluation with Detailed Logging and Analysis

Tracks:
- Bugs encountered
- Agent failures with root cause analysis
- Time breakdown by phase
- Performance metrics
- Optimization opportunities
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# Set model for testing
os.environ["OLLAMA_MODEL"] = "gpt-oss:20b"

from agents.task_executor_agent import TaskExecutorAgent


@dataclass
class ToolCall:
    """Tool call record"""
    round_num: int
    tool_name: str
    args: Dict[str, Any]
    result: str
    success: bool
    duration: float


@dataclass
class TestResult:
    """Comprehensive test result"""
    test_name: str
    level: str
    goal: str
    status: str  # success, failed, error
    failure_reason: str
    duration: float
    rounds_used: int
    max_rounds: int

    # Files
    expected_files: List[str]
    created_files: List[str]
    files_score: float  # percentage of expected files created

    # Performance
    avg_llm_time: float
    total_llm_time: float
    llm_calls: int
    total_tokens: int

    # Tool usage
    tool_calls: List[Dict[str, Any]]
    tool_call_count: int
    unique_tools_used: List[str]

    # Errors
    errors: List[str]
    warnings: List[str]

    # Optimization insights
    inefficiencies: List[str]
    suggestions: List[str]


class ComprehensiveEvaluator:
    """Comprehensive evaluation with detailed tracking"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results: List[TestResult] = []
        self.global_errors: List[str] = []
        self.start_time = time.time()

    def run_level1_test(
        self,
        test_name: str,
        goal: str,
        max_rounds: int,
        expected_files: List[str]
    ) -> TestResult:
        """Run Level 1 test with comprehensive tracking"""
        print(f"\n{'='*80}")
        print(f"LEVEL 1 - {test_name}")
        print(f"{'='*80}")

        workspace = Path(f".agent_workspace/eval_{test_name.lower().replace(' ', '_').replace(':', '')}")

        test_start = time.time()
        errors = []
        warnings = []
        tool_calls_log = []
        inefficiencies = []
        suggestions = []

        try:
            # Create agent
            agent = TaskExecutorAgent(
                workspace=workspace,
                goal=goal,
            )

            # Track initial state
            print(f"Workspace: {workspace}")
            print(f"Goal: {goal}")
            print(f"Max rounds: {max_rounds}")
            print(f"Expected files: {expected_files}")

            # Run agent with tracking
            run_start = time.time()
            result = agent.run(max_rounds=max_rounds)
            run_duration = time.time() - run_start

            # Extract performance data
            perf_stats = agent.perf_stats if hasattr(agent, 'perf_stats') else None

            llm_calls = len(perf_stats.llm_call_times) if perf_stats and hasattr(perf_stats, 'llm_call_times') else 0
            total_llm_time = sum(perf_stats.llm_call_times) if perf_stats and hasattr(perf_stats, 'llm_call_times') else 0
            avg_llm_time = total_llm_time / llm_calls if llm_calls > 0 else 0
            total_tokens = perf_stats.total_tokens_estimated if perf_stats and hasattr(perf_stats, 'total_tokens_estimated') else 0

            # Check files
            created_files = []
            for expected in expected_files:
                file_path = workspace / expected
                if file_path.exists():
                    created_files.append(expected)

            files_score = len(created_files) / len(expected_files) if expected_files else 0

            # Analyze tool usage from history
            if hasattr(agent, 'state') and hasattr(agent.state, 'messages'):
                tool_usage = self._analyze_tool_usage(agent.state.messages)
                tool_calls_log = tool_usage['calls']
                unique_tools = tool_usage['unique_tools']

                # Detect inefficiencies
                inefficiencies = self._detect_inefficiencies(tool_usage, run_duration)
            else:
                unique_tools = []

            # Determine status and failure reason
            status = result.get('status', 'unknown')
            failure_reason = result.get('reason', '')

            if status != 'success':
                if files_score == 1.0:
                    warnings.append("Agent didn't signal completion but all files were created")
                    suggestions.append("Improve completion detection/nudging")

                if run_duration > 60:
                    inefficiencies.append(f"Test took {run_duration:.1f}s but failed")
                    suggestions.append("Investigate timeout or stuck behavior")

            # Build result
            test_result = TestResult(
                test_name=test_name,
                level="L1",
                goal=goal,
                status=status,
                failure_reason=failure_reason,
                duration=time.time() - test_start,
                rounds_used=result.get('rounds_used', 0) or llm_calls,
                max_rounds=max_rounds,
                expected_files=expected_files,
                created_files=created_files,
                files_score=files_score,
                avg_llm_time=avg_llm_time,
                total_llm_time=total_llm_time,
                llm_calls=llm_calls,
                total_tokens=total_tokens,
                tool_calls=tool_calls_log,
                tool_call_count=len(tool_calls_log),
                unique_tools_used=unique_tools,
                errors=errors,
                warnings=warnings,
                inefficiencies=inefficiencies,
                suggestions=suggestions
            )

            # Display result
            status_icon = "✅" if status == "success" else "❌"
            print(f"\n{status_icon} {test_name}: {status.upper()}")
            print(f"   Duration: {test_result.duration:.1f}s")
            print(f"   Rounds: {test_result.rounds_used}/{max_rounds}")
            print(f"   Files: {len(created_files)}/{len(expected_files)}")
            print(f"   LLM calls: {llm_calls} (avg {avg_llm_time:.2f}s)")

            if inefficiencies:
                print(f"   ⚠️  Inefficiencies: {len(inefficiencies)}")
            if errors:
                print(f"   🐛 Errors: {len(errors)}")

            return test_result

        except Exception as e:
            # Handle errors
            error_trace = traceback.format_exc()
            errors.append(f"{type(e).__name__}: {str(e)}")
            self.global_errors.append(f"{test_name}: {error_trace}")

            test_result = TestResult(
                test_name=test_name,
                level="L1",
                goal=goal,
                status="error",
                failure_reason=str(e),
                duration=time.time() - test_start,
                rounds_used=0,
                max_rounds=max_rounds,
                expected_files=expected_files,
                created_files=[],
                files_score=0,
                avg_llm_time=0,
                total_llm_time=0,
                llm_calls=0,
                total_tokens=0,
                tool_calls=[],
                tool_call_count=0,
                unique_tools_used=[],
                errors=errors,
                warnings=warnings,
                inefficiencies=inefficiencies,
                suggestions=["Fix error: " + str(e)]
            )

            print(f"\n❌ {test_name}: ERROR")
            print(f"   {type(e).__name__}: {str(e)}")

            return test_result

    def _analyze_tool_usage(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze tool usage from message history"""
        tool_calls = []
        unique_tools = set()

        for msg in messages:
            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                for tc in msg.get('tool_calls', []):
                    tool_name = tc.get('function', {}).get('name', 'unknown')
                    tool_calls.append({
                        'tool': tool_name,
                        'args': tc.get('function', {}).get('arguments', {})
                    })
                    unique_tools.add(tool_name)

        return {
            'calls': tool_calls,
            'unique_tools': sorted(list(unique_tools))
        }

    def _detect_inefficiencies(self, tool_usage: Dict, duration: float) -> List[str]:
        """Detect inefficiencies in agent behavior"""
        inefficiencies = []

        # Repeated tool calls
        tool_names = [tc['tool'] for tc in tool_usage['calls']]
        if len(tool_names) > 3:
            for tool in set(tool_names):
                count = tool_names.count(tool)
                if count > 5:
                    inefficiencies.append(f"Repeated {tool} {count} times (possible loop)")

        # Long duration
        if duration > 120:
            inefficiencies.append(f"Test took {duration:.1f}s (>2 minutes)")

        # No write_file calls
        if 'write_file' not in tool_usage['unique_tools']:
            inefficiencies.append("No files written (agent didn't attempt creation)")

        return inefficiencies

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        total_duration = time.time() - self.start_time

        # Calculate statistics
        total_tests = len(self.results)
        successful = len([r for r in self.results if r.status == 'success'])
        failed = len([r for r in self.results if r.status == 'failed'])
        errors = len([r for r in self.results if r.status == 'error'])

        avg_duration = sum(r.duration for r in self.results) / total_tests if total_tests > 0 else 0
        total_llm_time = sum(r.total_llm_time for r in self.results)
        total_llm_calls = sum(r.llm_calls for r in self.results)
        total_tokens = sum(r.total_tokens for r in self.results)

        # Generate report
        report = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'total_duration': total_duration,
                'total_tests': total_tests,
                'successful': successful,
                'failed': failed,
                'errors': errors,
                'success_rate': successful / total_tests if total_tests > 0 else 0,
                'avg_test_duration': avg_duration,
                'total_llm_time': total_llm_time,
                'total_llm_calls': total_llm_calls,
                'total_tokens': total_tokens,
                'avg_llm_time': total_llm_time / total_llm_calls if total_llm_calls > 0 else 0
            },
            'tests': [asdict(r) for r in self.results],
            'global_errors': self.global_errors,
            'optimization_insights': self._generate_optimization_insights()
        }

        # Write JSON report
        json_path = self.output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Write human-readable report
        text_path = self.output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(text_path, 'w') as f:
            self._write_text_report(f, report)

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Results: {successful}/{total_tests} passed ({report['summary']['success_rate']*100:.1f}%)")
        print(f"Duration: {total_duration:.1f}s")
        print("Reports written to:")
        print(f"  - {json_path}")
        print(f"  - {text_path}")

        return report

    def _generate_optimization_insights(self) -> Dict[str, Any]:
        """Generate optimization insights from all test results"""
        insights = {
            'common_inefficiencies': [],
            'common_errors': [],
            'slow_tests': [],
            'high_token_tests': [],
            'recommendations': []
        }

        # Aggregate inefficiencies
        all_inefficiencies = {}
        for result in self.results:
            for ineff in result.inefficiencies:
                all_inefficiencies[ineff] = all_inefficiencies.get(ineff, 0) + 1

        insights['common_inefficiencies'] = [
            {'issue': k, 'count': v}
            for k, v in sorted(all_inefficiencies.items(), key=lambda x: -x[1])
        ]

        # Aggregate errors
        all_errors = {}
        for result in self.results:
            for error in result.errors:
                all_errors[error] = all_errors.get(error, 0) + 1

        insights['common_errors'] = [
            {'error': k, 'count': v}
            for k, v in sorted(all_errors.items(), key=lambda x: -x[1])
        ]

        # Slow tests (>60s)
        insights['slow_tests'] = [
            {'test': r.test_name, 'duration': r.duration}
            for r in sorted(self.results, key=lambda x: -x.duration)
            if r.duration > 60
        ]

        # High token tests (>20K)
        insights['high_token_tests'] = [
            {'test': r.test_name, 'tokens': r.total_tokens}
            for r in sorted(self.results, key=lambda x: -x.total_tokens)
            if r.total_tokens > 20000
        ]

        # Generate recommendations
        if insights['common_inefficiencies']:
            top_ineff = insights['common_inefficiencies'][0]
            insights['recommendations'].append(
                f"Fix most common inefficiency: {top_ineff['issue']} (affects {top_ineff['count']} tests)"
            )

        if insights['slow_tests']:
            insights['recommendations'].append(
                f"Optimize slow tests: {len(insights['slow_tests'])} tests >60s"
            )

        failed_rate = len([r for r in self.results if r.status == 'failed']) / len(self.results) if self.results else 0
        if failed_rate > 0.5:
            insights['recommendations'].append(
                f"High failure rate ({failed_rate*100:.1f}%) - investigate root causes"
            )

        return insights

    def _write_text_report(self, f, report: Dict):
        """Write human-readable text report"""
        f.write("# COMPREHENSIVE EVALUATION REPORT\n\n")

        # Summary
        f.write("## Summary\n\n")
        summary = report['summary']
        f.write(f"**Timestamp**: {summary['timestamp']}\n")
        f.write(f"**Total Duration**: {summary['total_duration']:.1f}s\n")
        f.write(f"**Tests**: {summary['total_tests']} ({summary['successful']} passed, {summary['failed']} failed, {summary['errors']} errors)\n")
        f.write(f"**Success Rate**: {summary['success_rate']*100:.1f}%\n")
        f.write(f"**Avg Test Duration**: {summary['avg_test_duration']:.1f}s\n")
        f.write(f"**Total LLM Time**: {summary['total_llm_time']:.1f}s\n")
        f.write(f"**Total LLM Calls**: {summary['total_llm_calls']}\n")
        f.write(f"**Total Tokens**: {summary['total_tokens']:,}\n")
        f.write(f"**Avg LLM Time**: {summary['avg_llm_time']:.2f}s\n\n")

        # Test Results
        f.write("## Test Results\n\n")
        for test in report['tests']:
            icon = "✅" if test['status'] == 'success' else "❌"
            f.write(f"### {icon} {test['test_name']}\n\n")
            f.write(f"- **Status**: {test['status']}\n")
            f.write(f"- **Duration**: {test['duration']:.1f}s\n")
            f.write(f"- **Rounds**: {test['rounds_used']}/{test['max_rounds']}\n")
            f.write(f"- **Files**: {len(test['created_files'])}/{len(test['expected_files'])} ({test['files_score']*100:.0f}%)\n")
            f.write(f"- **LLM Calls**: {test['llm_calls']} (avg {test['avg_llm_time']:.2f}s)\n")
            f.write(f"- **Tokens**: {test['total_tokens']:,}\n")

            if test['failure_reason']:
                f.write(f"- **Failure Reason**: {test['failure_reason']}\n")

            if test['errors']:
                f.write(f"- **Errors**: {len(test['errors'])}\n")
                for error in test['errors']:
                    f.write(f"  - {error}\n")

            if test['warnings']:
                f.write(f"- **Warnings**: {len(test['warnings'])}\n")
                for warning in test['warnings']:
                    f.write(f"  - {warning}\n")

            if test['inefficiencies']:
                f.write("- **Inefficiencies**:\n")
                for ineff in test['inefficiencies']:
                    f.write(f"  - {ineff}\n")

            if test['suggestions']:
                f.write("- **Suggestions**:\n")
                for sugg in test['suggestions']:
                    f.write(f"  - {sugg}\n")

            f.write("\n")

        # Optimization Insights
        f.write("## Optimization Insights\n\n")
        insights = report['optimization_insights']

        if insights['common_inefficiencies']:
            f.write("### Common Inefficiencies\n\n")
            for item in insights['common_inefficiencies'][:5]:
                f.write(f"- {item['issue']} ({item['count']} tests)\n")
            f.write("\n")

        if insights['common_errors']:
            f.write("### Common Errors\n\n")
            for item in insights['common_errors'][:5]:
                f.write(f"- {item['error']} ({item['count']} tests)\n")
            f.write("\n")

        if insights['slow_tests']:
            f.write("### Slow Tests (>60s)\n\n")
            for item in insights['slow_tests'][:5]:
                f.write(f"- {item['test']}: {item['duration']:.1f}s\n")
            f.write("\n")

        if insights['high_token_tests']:
            f.write("### High Token Usage (>20K)\n\n")
            for item in insights['high_token_tests'][:5]:
                f.write(f"- {item['test']}: {item['tokens']:,} tokens\n")
            f.write("\n")

        if insights['recommendations']:
            f.write("### Recommendations\n\n")
            for rec in insights['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")


def main():
    print("="*80)
    print("COMPREHENSIVE THREE-LEVEL EVALUATION")
    print("="*80)
    print(f"Model: {os.environ.get('OLLAMA_MODEL', 'default')}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    evaluator = ComprehensiveEvaluator(output_dir=Path("evaluation_results"))

    # Level 1 Tests
    print("\n" + "="*80)
    print("LEVEL 1: DIRECT TASKEXECUTOR (L1-L4)")
    print("="*80)

    level1_tests = [
        ("L1: Simple File", "Create a file hello.py that prints 'Hello World'", 5, ["hello.py"]),
        ("L2: File with Function", "Create calculator.py with an add(a, b) function and a test for it", 10, ["calculator.py", "test_calculator.py"]),
        ("L3: Multi-File Package", "Create a Python package 'mathx' with add, subtract, multiply, divide functions in separate files, with tests for all functions", 20, ["mathx/__init__.py", "mathx/add.py", "mathx/subtract.py", "mathx/multiply.py", "mathx/divide.py", "tests/test_mathx.py"]),
        ("L4: Package with Dependencies", "Create a 'requests_wrapper' package that wraps HTTP requests with retry logic. Include tests.", 20, ["requests_wrapper/__init__.py", "requests_wrapper/wrapper.py", "tests/test_wrapper.py"]),
    ]

    for test_name, goal, max_rounds, expected_files in level1_tests:
        result = evaluator.run_level1_test(test_name, goal, max_rounds, expected_files)
        evaluator.results.append(result)

    # Generate final report
    report = evaluator.generate_report()

    # Return exit code based on success rate
    success_rate = report['summary']['success_rate']
    return 0 if success_rate >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
