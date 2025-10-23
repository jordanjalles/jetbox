"""Advanced arithmetic operations for the tests package.

This file mirrors :mod:`mathx.advanced` but is placed inside the
``tests/mathx`` package.
"""

from .base import MathOperation


class Multiply(MathOperation):
    def compute(self):
        return self.a * self.b


class Divide(MathOperation):
    def compute(self):
        if self.b == 0:
            raise ZeroDivisionError("division by zero")
        return self.a / self.b


# Public helper functions

def multiply(a, b):
    return Multiply(a, b)()


def divide(a, b):
    return Divide(a, b)()

# End of tests/mathx/advanced.py
