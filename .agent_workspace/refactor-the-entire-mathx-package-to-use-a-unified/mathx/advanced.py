"""Advanced arithmetic operations.

The :class:`Multiply` and :class:`Divide` classes are defined here.
They follow the same pattern as the basic operations.
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
    """Return the product of *a* and *b*.

    This function is a thin wrapper around :class:`Multiply`.
    """
    return Multiply(a, b)()


def divide(a, b):
    """Return the quotient of *a* divided by *b*.

    This function is a thin wrapper around :class:`Divide`.
    """
    return Divide(a, b)()

# End of mathx/advanced.py
