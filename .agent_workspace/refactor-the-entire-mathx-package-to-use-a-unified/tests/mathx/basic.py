"""Basic arithmetic operations for the tests package.

This file mirrors :mod:`mathx.basic` but is placed inside the
``tests/mathx`` package so that the test suite can import the
operations directly from ``mathx`` when it runs from the tests
directory.
"""

from .base import MathOperation


class Add(MathOperation):
    def compute(self):
        return self.a + self.b


class Subtract(MathOperation):
    def compute(self):
        return self.a - self.b


# Public helper functions

def add(a, b):
    return Add(a, b)()


def subtract(a, b):
    return Subtract(a, b)()

# End of tests/mathx/basic.py
