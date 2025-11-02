"""MathX package providing basic arithmetic operations.

This package exposes four functions: add, subtract, multiply, and divide.
Each operation is implemented in its own module for clarity and modularity.
"""

from .add import add
from .subtract import subtract
from .multiply import multiply
from .divide import divide

__all__ = ["add", "subtract", "multiply", "divide"]
