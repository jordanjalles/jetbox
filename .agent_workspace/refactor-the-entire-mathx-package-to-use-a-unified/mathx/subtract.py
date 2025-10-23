"""Subtraction operation.

Defines :class:`Subtract` inheriting from :class:`mathx.base.MathOperation`.
Provides a helper function ``subtract``.
"""

from .base import MathOperation


class Subtract(MathOperation):
    def compute(self):
        return self.a - self.b


def subtract(a, b):
    return Subtract(a, b)()

# End of mathx/subtract.py
