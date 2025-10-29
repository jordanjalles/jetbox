"""
Simple Evaluation Test - Direct file creation and validation

Tests the validation logic by creating files directly and running validation commands.
"""
from pathlib import Path
import tempfile
import subprocess

# Test 1: Simple function
def test_simple_function():
    print("\n=== Test 1: Simple Function ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create implementation
        greet_py = workspace / "greet.py"
        greet_py.write_text("""def greet(name):
    return f'Hello, {name}!'
""")

        # Validate
        import os
        os.chdir(workspace)
        result = subprocess.run(
            ["python", "-c", "from greet import greet; assert greet('World') == 'Hello, World!'"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✅ PASS - Simple function works")
            return True
        else:
            print(f"❌ FAIL - {result.stderr}")
            return False


# Test 2: Simple math
def test_simple_math():
    print("\n=== Test 2: Simple Math ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create implementation
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

        # Validate
        import os
        os.chdir(workspace)
        result = subprocess.run(
            ["python", "-c", "from math_utils import add, divide; assert add(2,3)==5; assert divide(10,2)==5"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✅ PASS - Math functions work")
            return True
        else:
            print(f"❌ FAIL - {result.stderr}")
            return False


# Test 3: Multi-file package
def test_multi_file_package():
    print("\n=== Test 3: Multi-file Package ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create implementation
        shapes_dir = workspace / "shapes"
        shapes_dir.mkdir()

        (shapes_dir / "__init__.py").write_text("""from .circle import Circle
from .rectangle import Rectangle

__all__ = ['Circle', 'Rectangle']
""")

        (shapes_dir / "circle.py").write_text("""import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2
""")

        (shapes_dir / "rectangle.py").write_text("""class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height
""")

        # Validate
        import os
        os.chdir(workspace)
        result = subprocess.run(
            ["python", "-c", "from shapes import Circle, Rectangle; import math; c=Circle(5); assert abs(c.area()-math.pi*25)<0.1"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✅ PASS - Multi-file package works")
            return True
        else:
            print(f"❌ FAIL - {result.stderr}")
            return False


if __name__ == "__main__":
    print("🧪 Simple Evaluation Tests")
    print("=" * 60)

    results = []
    results.append(("simple_function", test_simple_function()))
    results.append(("simple_math", test_simple_math()))
    results.append(("multi_file_package", test_multi_file_package()))

    print("\n" + "=" * 60)
    print("Summary:")
    passed = sum(1 for _, success in results if success)
    print(f"  {passed}/{len(results)} tests passed")

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    if passed == len(results):
        print("\n✅ All validation tests passed!")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed")
