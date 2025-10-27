# Wrapper to expose calculator functions from submodule
import importlib.util
import os

# Path to the submodule calculator.py
submodule_path = os.path.join(os.getcwd(), ".agent_workspace", "create-calculator-py-with-add-subtract-multiply-fu", "calculator.py")

spec = importlib.util.spec_from_file_location("sub_calculator", submodule_path)
sub_calculator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub_calculator)

# Re-export functions
add = sub_calculator.add
subtract = sub_calculator.subtract
multiply = sub_calculator.multiply

__all__ = ["add", "subtract", "multiply"]
