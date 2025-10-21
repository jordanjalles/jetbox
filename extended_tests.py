#!/usr/bin/env python3
"""Extended stress tests for overnight autonomous testing (L6+)."""

from pathlib import Path

# L6: Extreme Challenge Tests - Push agent to limits
L6_TESTS = [
    {
        "id": "L6-1",
        "level": 6,
        "name": "Self-Improving Code",
        "task": "Create a simple function optimizer (optimizer.py) that uses cProfile to identify slow operations and suggests improvements. Include test_optimizer.py with tests and performance benchmarks.",
        "expected_files": ["optimizer.py", "test_optimizer.py"],
        "timeout": 600,
        "verify_cmd": ["python", "-m", "pytest", "test_optimizer.py", "-q"],
    },
    {
        "id": "L6-2",
        "level": 6,
        "name": "Multi-Module Dependencies",
        "task": "Create a project with 3 interdependent modules: parser.py (parses config), validator.py (validates parsed data using parser), executor.py (executes validated commands). Add tests for each. Ensure no circular dependencies.",
        "expected_files": ["parser.py", "validator.py", "executor.py", "test_parser.py", "test_validator.py", "test_executor.py"],
        "timeout": 600,
        "verify_cmd": ["python", "-m", "pytest", "-q"],
    },
    {
        "id": "L6-3",
        "level": 6,
        "name": "Error Recovery System",
        "task": "Implement a resilient file processor (processor.py) with error_handler.py for retry logic and state_manager.py for persistence. Must handle corrupt data gracefully. Include comprehensive tests.",
        "expected_files": ["processor.py", "error_handler.py", "state_manager.py"],
        "timeout": 720,
    },
    {
        "id": "L6-4",
        "level": 6,
        "name": "Code Migration Task",
        "task": "Create legacy_code.py with Python 2 style code (print statements, old string formatting). Then create modern_code.py migrating it to Python 3.11 with type hints, f-strings, and pathlib. Add tests proving equivalence.",
        "expected_files": ["legacy_code.py", "modern_code.py", "test_migration.py"],
        "timeout": 600,
        "setup": lambda: Path("legacy_code.py").write_text(
            "# Python 2 style code\n"
            "def process_file(filename):\n"
            "    f = open(filename, 'r')\n"
            "    data = f.read()\n"
            "    f.close()\n"
            "    print 'Processing:', filename\n"
            "    return data.upper()\n\n"
            "def format_output(name, count):\n"
            "    return 'Name: %s, Count: %d' % (name, count)\n"
        ),
    },
    {
        "id": "L6-5",
        "level": 6,
        "name": "Documentation Generator",
        "task": "Create doc_generator.py that reads Python files and generates Markdown documentation with function signatures, docstrings, and examples. Test it on itself. Output to docs/ directory.",
        "expected_files": ["doc_generator.py", "test_doc_generator.py"],
        "timeout": 600,
    },
    {
        "id": "L6-6",
        "level": 6,
        "name": "Concurrent Task Manager",
        "task": "Create task_manager.py with a ThreadPoolExecutor-based system for running tasks concurrently. Must handle task dependencies, timeouts, and error propagation. Include tests with real concurrent tasks.",
        "expected_files": ["task_manager.py", "test_task_manager.py"],
        "timeout": 600,
    },
]

# L7: Algorithmic Challenge Tests
L7_TESTS = [
    {
        "id": "L7-1",
        "level": 7,
        "name": "LRU Cache Implementation",
        "task": "Implement an LRU (Least Recently Used) cache in lru_cache.py with O(1) get and put operations. Use OrderedDict or custom doubly-linked list. Include comprehensive tests with edge cases.",
        "expected_files": ["lru_cache.py", "test_lru_cache.py"],
        "timeout": 480,
        "verify_cmd": ["python", "-m", "pytest", "test_lru_cache.py", "-q"],
    },
    {
        "id": "L7-2",
        "level": 7,
        "name": "Balanced Binary Search Tree",
        "task": "Implement a balanced BST (AVL or Red-Black tree) in bst.py with insert, delete, search, and rebalancing. Include tests verifying balance property and correctness.",
        "expected_files": ["bst.py", "test_bst.py"],
        "timeout": 600,
    },
    {
        "id": "L7-3",
        "level": 7,
        "name": "Trie-Based Autocomplete",
        "task": "Implement a Trie data structure in trie.py for autocomplete functionality. Support insert, search, and prefix matching with autocomplete suggestions. Include tests and performance benchmarks.",
        "expected_files": ["trie.py", "test_trie.py"],
        "timeout": 480,
    },
]

# L8: System Design Tests
L8_TESTS = [
    {
        "id": "L8-1",
        "level": 8,
        "name": "Rate Limiter Implementation",
        "task": "Design a rate limiter in rate_limiter.py supporting multiple strategies: token bucket, sliding window, fixed window. Include tests simulating high request rates.",
        "expected_files": ["rate_limiter.py", "test_rate_limiter.py"],
        "timeout": 600,
    },
    {
        "id": "L8-2",
        "level": 8,
        "name": "Caching Layer with TTL",
        "task": "Create cache.py implementing a caching layer with TTL (time-to-live), LRU eviction, and size limits. Support multiple backends (memory, file). Include comprehensive tests.",
        "expected_files": ["cache.py", "test_cache.py"],
        "timeout": 600,
    },
    {
        "id": "L8-3",
        "level": 8,
        "name": "Priority Job Queue",
        "task": "Implement a priority job queue in job_queue.py with support for priority levels, retries, and dead letter queue. Include worker simulation and tests.",
        "expected_files": ["job_queue.py", "test_job_queue.py"],
        "timeout": 720,
    },
]

# Combine all extended tests
EXTENDED_TESTS = L6_TESTS + L7_TESTS + L8_TESTS


def get_test_by_level(level: int) -> list[dict]:
    """Get all tests for a specific level."""
    return [t for t in EXTENDED_TESTS if t["level"] == level]


def get_all_tests() -> list[dict]:
    """Get all extended tests."""
    return EXTENDED_TESTS
