"""
Manual Evaluation Helper

Creates sample implementations for evaluation tasks to:
1. Test the evaluation framework validation logic
2. Demonstrate what correct implementations look like
3. Verify task definitions are valid
"""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_suite import EvaluationSuite


def create_simple_function(workspace: Path):
    """Create solution for 'simple_function' task."""
    greet_py = workspace / "greet.py"
    greet_py.write_text("""def greet(name):
    return f'Hello, {name}!'
""")


def create_simple_math(workspace: Path):
    """Create solution for 'simple_math' task."""
    math_utils = workspace / "math_utils.py"
    math_utils.write_text("""def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""")


def create_list_operations(workspace: Path):
    """Create solution for 'list_operations' task."""
    list_utils = workspace / "list_utils.py"
    list_utils.write_text("""def get_max(lst):
    return max(lst)

def get_min(lst):
    return min(lst)

def get_average(lst):
    return sum(lst) / len(lst)

def remove_duplicates(lst):
    seen = []
    for item in lst:
        if item not in seen:
            seen.append(item)
    return seen
""")


def create_class_definition(workspace: Path):
    """Create solution for 'class_definition' task."""
    person_py = workspace / "person.py"
    person_py.write_text("""class Person:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

    def __str__(self):
        return f"Person(name={self.name}, age={self.age}, email={self.email})"

    def is_adult(self):
        return self.age >= 18
""")


def create_multi_file_package(workspace: Path):
    """Create solution for 'multi_file_package' task."""
    shapes_dir = workspace / "shapes"
    shapes_dir.mkdir(exist_ok=True)

    # __init__.py
    (shapes_dir / "__init__.py").write_text("""from .circle import Circle
from .rectangle import Rectangle

__all__ = ['Circle', 'Rectangle']
""")

    # circle.py
    (shapes_dir / "circle.py").write_text("""import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2
""")

    # rectangle.py
    (shapes_dir / "rectangle.py").write_text("""class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height
""")


def test_task_validation(task_name: str):
    """Test validation for a specific task."""
    print(f"\n{'='*70}")
    print(f"Testing validation for: {task_name}")
    print('='*70)

    suite = EvaluationSuite()
    task = next((t for t in suite.tasks if t.name == task_name), None)

    if not task:
        print(f"❌ Task '{task_name}' not found")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create a task executor to get the actual workspace directory
        from task_executor_agent import TaskExecutorAgent
        executor = TaskExecutorAgent(
            workspace=workspace,
            goal=task.goal,
            max_rounds=1
        )
        actual_workspace = executor.workspace_manager.workspace_dir

        # Create the implementation in the ACTUAL workspace
        if task_name == "simple_function":
            create_simple_function(actual_workspace)
        elif task_name == "simple_math":
            create_simple_math(actual_workspace)
        elif task_name == "list_operations":
            create_list_operations(actual_workspace)
        elif task_name == "class_definition":
            create_class_definition(actual_workspace)
        elif task_name == "multi_file_package":
            create_multi_file_package(actual_workspace)
        else:
            print(f"⚠️  No manual implementation for '{task_name}'")
            return

        # Run validation
        result = suite.run_task(task, workspace)

        # Print results
        if result.success:
            print("\n✅ PASS - Task validation successful!")
            print(f"   Duration: {result.duration:.2f}s")
            print(f"   Files created: {', '.join(result.files_created)}")
        else:
            print("\n❌ FAIL - Task validation failed")
            print(f"   Failure category: {result.failure_category}")
            print(f"   Validation output:\n{result.validation_output}")


if __name__ == "__main__":
    # Test a few basic tasks
    print("🧪 Manual Evaluation Helper - Testing Task Validations")

    tasks_to_test = [
        "simple_function",
        "simple_math",
        "list_operations",
        "class_definition",
        "multi_file_package"
    ]

    for task_name in tasks_to_test:
        try:
            test_task_validation(task_name)
        except Exception as e:
            print(f"\n❌ Error testing {task_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("✅ Manual validation testing complete!")
    print("="*70)
