#!/usr/bin/env python3
"""
Comprehensive project evaluation suite for Jetbox orchestrator.

Tests L5-L7 coding tasks and L8 full projects with both orchestrator-only
and orchestrator+architect scenarios.
"""
import pytest
import tempfile
import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_validator import validate_workspace, SemanticValidator


# ===== TEST TASK DEFINITIONS =====

L5_TASKS = [
    {
        "id": "L5_json_csv_converter",
        "level": "L5",
        "name": "JSON/CSV File Converter",
        "description": "Create a file converter that converts between JSON and CSV formats",
        "goal": """Create a file converter utility:
1. FileConverter class with:
   - json_to_csv(input_path, output_path) -> converts JSON to CSV
   - csv_to_json(input_path, output_path) -> converts CSV to JSON
   - validate_file(path, format) -> checks if file is valid JSON or CSV
2. Handle nested JSON (flatten to dot notation for CSV)
3. Handle CSV headers and proper quoting
4. Write tests for both conversions and error cases""",
        "validation": {
            "classes": ["FileConverter"],
            "functions": ["json_to_csv", "csv_to_json"],
        },
        "timeout": 300,  # 5 minutes
    },
    {
        "id": "L5_data_validator",
        "level": "L5",
        "name": "Data Validator",
        "description": "Create validators for email, phone, and other common formats",
        "goal": """Create data validation system:
1. DataValidator class with:
   - validate_email(email) -> returns True/False with error message
   - validate_phone(phone, country='US') -> validates phone numbers
   - validate_url(url) -> checks URL format
   - validate_date(date_str, format='%Y-%m-%d') -> validates date strings
2. Include regex patterns for common formats
3. Write tests covering valid/invalid cases for each validator""",
        "validation": {
            "classes": ["DataValidator"],
            "functions": ["validate_email", "validate_phone", "validate_url"],
        },
        "timeout": 300,
    },
    {
        "id": "L5_cli_calculator",
        "level": "L5",
        "name": "CLI Calculator",
        "description": "Create a command-line calculator with history",
        "goal": """Create CLI calculator with history:
1. Calculator class with:
   - evaluate(expression) -> evaluates math expressions (e.g., "2 + 3 * 4")
   - history -> list of (expression, result) tuples
   - clear_history() -> clears calculation history
2. Support: +, -, *, /, parentheses, and basic functions (sqrt, abs)
3. Main function with interactive loop (type expressions, get results)
4. Write tests for evaluation, history tracking, and edge cases""",
        "validation": {
            "classes": ["Calculator"],
            "functions": ["evaluate"],
        },
        "timeout": 300,
    },
]

L6_TASKS = [
    {
        "id": "L6_rest_api_client",
        "level": "L6",
        "name": "REST API Client with Auth",
        "description": "Create a REST API client with authentication and rate limiting",
        "goal": """Create REST API client library:
1. APIClient class with:
   - __init__(base_url, api_key) -> initializes with auth
   - get(endpoint, params) -> makes GET request
   - post(endpoint, data) -> makes POST request
   - _authenticate() -> handles auth headers
   - _rate_limit() -> enforces rate limiting (5 requests per second)
2. AuthHandler class for managing tokens and refresh
3. Include retry logic with exponential backoff
4. Write tests with mock responses and rate limit verification""",
        "validation": {
            "classes": ["APIClient", "AuthHandler"],
            "functions": ["get", "post"],
        },
        "timeout": 300,
    },
    {
        "id": "L6_data_pipeline",
        "level": "L6",
        "name": "Data Processing Pipeline",
        "description": "Create a multi-stage data processing pipeline",
        "goal": """Create data processing pipeline with 3+ stages:
1. Pipeline class with:
   - add_stage(stage) -> adds processing stage
   - process(data) -> runs data through all stages
   - get_stats() -> returns processing statistics
2. Stage base class with process(data) method
3. At least 3 concrete stages:
   - FilterStage(condition) -> filters data by condition
   - TransformStage(transformer) -> applies transformation
   - AggregateStage(aggregator) -> aggregates results
4. Write tests for pipeline execution and stage composition""",
        "validation": {
            "classes": ["Pipeline", "Stage", "FilterStage", "TransformStage"],
            "functions": ["add_stage", "process"],
        },
        "timeout": 300,
    },
    {
        "id": "L6_config_manager",
        "level": "L6",
        "name": "Configuration Manager",
        "description": "Create a configuration manager with validation and environment support",
        "goal": """Create configuration management system:
1. ConfigManager class with:
   - load_config(path, env='dev') -> loads config from YAML/JSON
   - get(key, default=None) -> retrieves config value (supports dot notation)
   - set(key, value) -> sets config value
   - validate() -> validates config against schema
2. ConfigSchema class for defining validation rules
3. Support environment-specific configs (dev, staging, prod)
4. Write tests for loading, validation, and environment overrides""",
        "validation": {
            "classes": ["ConfigManager", "ConfigSchema"],
            "functions": ["load_config", "get", "validate"],
        },
        "timeout": 300,
    },
]

L7_TASKS = [
    {
        "id": "L7_python_package",
        "level": "L7",
        "name": "Python Package with Setup",
        "description": "Create a complete Python package with setup.py, tests, and docs",
        "goal": """Create complete Python package called 'textutils':
1. Package structure:
   - textutils/__init__.py -> main module
   - textutils/string_ops.py -> string operations (reverse, count_words, etc.)
   - textutils/file_ops.py -> file operations (read, write, search)
   - tests/test_string_ops.py -> tests for string operations
   - tests/test_file_ops.py -> tests for file operations
   - setup.py -> package metadata and installation
   - README.md -> usage documentation
2. At least 5 functions across modules
3. All tests must pass
4. Package should be installable with 'pip install -e .'""",
        "validation": {
            "classes": [],
            "functions": ["reverse", "count_words"],
            "has_setup": True,
            "has_readme": True,
        },
        "timeout": 300,
    },
    {
        "id": "L7_multi_module_library",
        "level": "L7",
        "name": "Multi-Module Library",
        "description": "Create a library with multiple modules and dependencies",
        "goal": """Create 'datalib' library with multiple modules:
1. datalib/ package with:
   - readers.py -> JSONReader, CSVReader, XMLReader classes
   - writers.py -> JSONWriter, CSVWriter, XMLWriter classes
   - transformers.py -> DataTransformer with map/filter/reduce
   - validators.py -> DataValidator with type checking
2. Each module should have 2-3 classes
3. Include requirements.txt with dependencies
4. Write comprehensive tests covering all modules
5. Create examples/ directory with usage examples""",
        "validation": {
            "classes": ["JSONReader", "CSVReader", "JSONWriter", "DataTransformer"],
            "functions": [],
            "min_modules": 4,
        },
        "timeout": 300,
    },
    {
        "id": "L7_cli_tool",
        "level": "L7",
        "name": "CLI Tool with Config",
        "description": "Create a CLI tool with configuration, commands, and tests",
        "goal": """Create 'filetool' CLI application:
1. Commands:
   - filetool search <pattern> <path> -> searches for pattern in files
   - filetool replace <old> <new> <path> -> replaces text in files
   - filetool stats <path> -> shows file statistics (count, size, types)
2. Configuration file support (~/.filetool.yaml)
3. Proper argument parsing with help text
4. Progress bars for long operations
5. Write integration tests that test each command
6. Create setup.py to install as 'filetool' command""",
        "validation": {
            "classes": ["FileTool", "SearchCommand", "ReplaceCommand"],
            "functions": ["search", "replace", "stats"],
            "has_cli": True,
        },
        "timeout": 300,
    },
]

L8_TASKS = [
    {
        "id": "L8_microservices",
        "level": "L8",
        "name": "Microservices System",
        "description": "Create a microservices system with 2-3 services",
        "goal": """Create microservices system with user service and product service:
1. user-service/:
   - Flask/FastAPI app with user CRUD endpoints
   - User model with SQLite database
   - /users GET/POST/PUT/DELETE endpoints
2. product-service/:
   - Flask/FastAPI app with product CRUD endpoints
   - Product model with SQLite database
   - /products GET/POST/PUT/DELETE endpoints
3. Service communication via HTTP
4. Each service should have its own requirements.txt
5. Docker-compose.yml to run both services
6. Write integration tests that test cross-service communication""",
        "validation": {
            "services": ["user-service", "product-service"],
            "has_docker": True,
            "min_endpoints": 4,
        },
        "timeout": 600,  # 10 minutes for complex project
        "needs_architect": True,
    },
    {
        "id": "L8_web_application",
        "level": "L8",
        "name": "Web Application",
        "description": "Create a web application with frontend, backend, and database",
        "goal": """Create todo web application:
1. backend/:
   - Flask/FastAPI REST API
   - SQLite database with Todo model
   - Endpoints: GET/POST/PUT/DELETE /todos
   - User authentication (simple JWT)
2. frontend/:
   - Simple HTML/CSS/JS interface
   - List todos, add todo, mark complete
   - Connect to backend API
3. database/:
   - Schema definition
   - Migration scripts
4. Write tests for API endpoints and authentication
5. Include README with setup instructions""",
        "validation": {
            "components": ["backend", "frontend"],
            "has_database": True,
            "has_api": True,
        },
        "timeout": 600,
        "needs_architect": True,
    },
    {
        "id": "L8_distributed_system",
        "level": "L8",
        "name": "Distributed Task System",
        "description": "Create a distributed system with worker queue, API, and storage",
        "goal": """Create distributed task processing system:
1. api/:
   - REST API to submit tasks
   - Task model and job queue
   - Status tracking endpoints
2. worker/:
   - Background worker that processes tasks
   - Pulls from queue and executes
   - Updates task status
3. storage/:
   - Redis/SQLite for task storage
   - Job queue implementation
4. Communication between components via queue
5. Write tests for task submission, processing, and status tracking
6. Include docker-compose.yml to run all components""",
        "validation": {
            "components": ["api", "worker", "storage"],
            "has_queue": True,
            "has_docker": True,
        },
        "timeout": 600,
        "needs_architect": True,
    },
]

ALL_TASKS = L5_TASKS + L6_TASKS + L7_TASKS + L8_TASKS


# ===== VALIDATION HELPERS =====

def validate_task_result(workspace: Path, task: dict) -> dict:
    """
    Validate task completion based on task requirements.

    Args:
        workspace: Path to workspace directory
        task: Task definition dict

    Returns:
        Validation result dict with success, details, and metrics
    """
    result = {
        "success": False,
        "details": {},
        "files_created": [],
        "tests_passed": False,
        "code_quality": False,
    }

    # Get all created files
    try:
        files = []
        for item in workspace.rglob("*"):
            if item.is_file() and not item.name.startswith('.'):
                rel_path = item.relative_to(workspace)
                files.append(str(rel_path))
        result["files_created"] = sorted(files)
    except Exception as e:
        result["error"] = f"Failed to list files: {e}"
        return result

    # Check for required classes/functions using semantic validator
    if "validation" in task:
        validation = task["validation"]

        # Use SemanticValidator
        validator = SemanticValidator(workspace)

        if "classes" in validation or "functions" in validation:
            required = {
                "classes": validation.get("classes", []),
                "functions": validation.get("functions", []),
            }

            symbol_result = validator.has_required_symbols(required)
            result["details"]["symbols"] = symbol_result

            # Check if all required symbols found
            all_found = (
                len(symbol_result["missing"]["classes"]) == 0 and
                len(symbol_result["missing"]["functions"]) == 0
            )
            result["details"]["symbols_found"] = all_found

        # Check for setup.py
        if validation.get("has_setup"):
            has_setup = any("setup.py" in f for f in files)
            result["details"]["has_setup"] = has_setup

        # Check for README
        if validation.get("has_readme"):
            has_readme = any("README" in f.upper() for f in files)
            result["details"]["has_readme"] = has_readme

        # Check for CLI entry point
        if validation.get("has_cli"):
            has_cli = (
                any("setup.py" in f for f in files) or
                any("__main__" in f for f in files)
            )
            result["details"]["has_cli"] = has_cli

        # Check for minimum modules
        if "min_modules" in validation:
            py_files = [f for f in files if f.endswith('.py')]
            result["details"]["modules_count"] = len(py_files)
            result["details"]["has_min_modules"] = len(py_files) >= validation["min_modules"]

        # Check for services (L8)
        if "services" in validation:
            found_services = []
            for service in validation["services"]:
                if any(service in f for f in files):
                    found_services.append(service)
            result["details"]["services_found"] = found_services
            result["details"]["has_all_services"] = len(found_services) == len(validation["services"])

        # Check for Docker
        if validation.get("has_docker"):
            has_docker = any("docker" in f.lower() for f in files)
            result["details"]["has_docker"] = has_docker

    # Run tests if test files exist
    test_files = [f for f in files if "test_" in f and f.endswith('.py')]
    if test_files:
        try:
            # Run pytest in workspace
            proc = subprocess.run(
                ["pytest", "-q", "--tb=short"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )
            result["tests_passed"] = proc.returncode == 0
            result["details"]["test_output"] = proc.stdout + proc.stderr
        except Exception as e:
            result["details"]["test_error"] = str(e)

    # Check code quality with ruff (if Python files exist)
    py_files = [f for f in files if f.endswith('.py')]
    if py_files:
        try:
            proc = subprocess.run(
                ["ruff", "check", ".", "--quiet"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result["code_quality"] = proc.returncode == 0
            if proc.stdout:
                result["details"]["ruff_output"] = proc.stdout
        except Exception as e:
            result["details"]["ruff_error"] = str(e)

    # Determine overall success
    # For L5-L7: need symbols, tests, and code quality
    # For L8: need components and architecture
    if task["level"] in ["L5", "L6", "L7"]:
        result["success"] = (
            result["details"].get("symbols_found", False) and
            (result["tests_passed"] or len(test_files) == 0) and
            result["code_quality"]
        )
    else:  # L8
        result["success"] = (
            len(result["files_created"]) > 0 and
            result["details"].get("has_all_services", True) and
            (result["tests_passed"] or len(test_files) == 0)
        )

    return result


def run_orchestrator_once(task: dict, use_architect: bool = False) -> dict:
    """
    Run orchestrator in --once mode for a single task.

    Args:
        task: Task definition dict
        use_architect: Whether to use architect for planning

    Returns:
        Result dict with status, duration, workspace, etc.
    """
    # Build command
    cmd = [sys.executable, "orchestrator_main.py", "--once"]

    # For L8 tasks with architect, add architect hint
    if use_architect and task.get("needs_architect"):
        goal = f"[Use architect for planning] {task['goal']}"
    else:
        goal = task["goal"]

    cmd.append(goal)

    # Run orchestrator
    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=task.get("timeout", 300),
            cwd=Path.cwd(),
        )
        duration = time.time() - start_time

        # Find the workspace that was created
        # Orchestrator creates workspace in .agent_workspace/ with slug from goal
        workspace = find_workspace_for_task(task)

        return {
            "success": proc.returncode == 0,
            "duration": duration,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "workspace": workspace,
        }
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        workspace = find_workspace_for_task(task)
        return {
            "success": False,
            "duration": duration,
            "error": "timeout",
            "timeout": task.get("timeout", 300),
            "workspace": workspace,
        }
    except Exception as e:
        duration = time.time() - start_time
        workspace = find_workspace_for_task(task)
        return {
            "success": False,
            "duration": duration,
            "error": str(e),
            "workspace": workspace,
        }


def find_workspace_for_task(task: dict) -> Path:
    """
    Find the workspace directory created by orchestrator for a task.

    Args:
        task: Task definition dict

    Returns:
        Path to workspace directory (may not exist if task failed early)
    """
    import re

    # Create slug from goal (matches workspace_manager.py logic)
    goal = task["goal"]
    slug = re.sub(r'[^a-z0-9]+', '-', goal.lower())
    slug = slug.strip('-')[:60]

    workspace_dir = Path.cwd() / ".agent_workspace"

    # Try exact match
    exact_path = workspace_dir / slug
    if exact_path.exists():
        return exact_path

    # Try finding workspace by partial match (orchestrator may truncate differently)
    if workspace_dir.exists():
        # Get all workspaces and find best match by modification time
        workspaces = []
        for item in workspace_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if slug is in workspace name
                if slug[:30] in item.name:
                    workspaces.append((item.stat().st_mtime, item))

        if workspaces:
            # Return most recent matching workspace
            workspaces.sort(reverse=True)
            return workspaces[0][1]

    # No workspace found - return expected path anyway
    return exact_path


# ===== PYTEST TEST CASES =====

@pytest.mark.parametrize("task", L5_TASKS, ids=[t["id"] for t in L5_TASKS])
def test_l5_task(task):
    """Test L5 single-file utility tasks."""
    # Run orchestrator
    run_result = run_orchestrator_once(task)

    # Get workspace from run result
    workspace = run_result.get("workspace")
    if not workspace:
        pytest.fail(f"No workspace found for task {task['id']}")

    # Validate result
    validation_result = validate_task_result(workspace, task)

    # Save results for later analysis
    save_test_result(task, run_result, validation_result)

    # Assert success
    assert validation_result["success"], (
        f"Task {task['id']} failed validation:\n"
        f"Workspace: {workspace}\n"
        f"Files: {validation_result['files_created']}\n"
        f"Details: {validation_result['details']}\n"
        f"Duration: {run_result['duration']:.1f}s"
    )


@pytest.mark.parametrize("task", L6_TASKS, ids=[t["id"] for t in L6_TASKS])
def test_l6_task(task):
    """Test L6 multi-file module tasks."""
    # Run orchestrator
    run_result = run_orchestrator_once(task)

    # Get workspace from run result
    workspace = run_result.get("workspace")
    if not workspace:
        pytest.fail(f"No workspace found for task {task['id']}")

    # Validate result
    validation_result = validate_task_result(workspace, task)

    # Save results
    save_test_result(task, run_result, validation_result)

    # Assert success
    assert validation_result["success"], (
        f"Task {task['id']} failed validation:\n"
        f"Workspace: {workspace}\n"
        f"Files: {validation_result['files_created']}\n"
        f"Details: {validation_result['details']}\n"
        f"Duration: {run_result['duration']:.1f}s"
    )


@pytest.mark.parametrize("task", L7_TASKS, ids=[t["id"] for t in L7_TASKS])
def test_l7_task(task):
    """Test L7 complete package tasks."""
    # Run orchestrator
    run_result = run_orchestrator_once(task)

    # Get workspace from run result
    workspace = run_result.get("workspace")
    if not workspace:
        pytest.fail(f"No workspace found for task {task['id']}")

    # Validate result
    validation_result = validate_task_result(workspace, task)

    # Save results
    save_test_result(task, run_result, validation_result)

    # Assert success
    assert validation_result["success"], (
        f"Task {task['id']} failed validation:\n"
        f"Workspace: {workspace}\n"
        f"Files: {validation_result['files_created']}\n"
        f"Details: {validation_result['details']}\n"
        f"Duration: {run_result['duration']:.1f}s"
    )


@pytest.mark.parametrize("task", L8_TASKS, ids=[t["id"] for t in L8_TASKS])
def test_l8_task_with_architect(task):
    """Test L8 full project tasks with architect."""
    # Run orchestrator with architect
    run_result = run_orchestrator_once(task, use_architect=True)

    # Get workspace from run result
    workspace = run_result.get("workspace")
    if not workspace:
        pytest.fail(f"No workspace found for task {task['id']}")

    # Validate result
    validation_result = validate_task_result(workspace, task)

    # Save results
    save_test_result(task, run_result, validation_result)

    # Assert success
    assert validation_result["success"], (
        f"Task {task['id']} failed validation:\n"
        f"Workspace: {workspace}\n"
        f"Files: {validation_result['files_created']}\n"
        f"Details: {validation_result['details']}\n"
        f"Duration: {run_result['duration']:.1f}s"
    )


@pytest.mark.parametrize("task", L8_TASKS, ids=[t["id"] for t in L8_TASKS])
def test_l8_task_without_architect(task):
    """Test L8 full project tasks without architect."""
    # Run orchestrator without architect
    run_result = run_orchestrator_once(task, use_architect=False)

    # Get workspace from run result
    workspace = run_result.get("workspace")
    if not workspace:
        pytest.fail(f"No workspace found for task {task['id']}")

    # Validate result
    validation_result = validate_task_result(workspace, task)

    # Save results
    save_test_result(task, run_result, validation_result)

    # Note: We don't assert here - just collect data for comparison
    # This test is for benchmarking architect vs no-architect


# ===== RESULT TRACKING =====

def save_test_result(task: dict, run_result: dict, validation_result: dict):
    """Save test result to JSON file for later analysis."""
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)

    result = {
        "task_id": task["id"],
        "level": task["level"],
        "name": task["name"],
        "description": task["description"],
        "timestamp": datetime.now().isoformat(),
        "run_result": {
            "success": run_result.get("success", False),
            "duration": run_result.get("duration", 0),
            "error": run_result.get("error"),
        },
        "validation_result": {
            "success": validation_result["success"],
            "files_created": validation_result["files_created"],
            "tests_passed": validation_result["tests_passed"],
            "code_quality": validation_result["code_quality"],
            "details": validation_result["details"],
        },
    }

    # Append to results file
    results_file = results_dir / "project_eval_results.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")
