"""Basic arithmetic operations.

Each operation is implemented as a subclass of :class:`MathOperation`.
The module also exposes convenience functions that instantiate the
corresponding class and return the result.  This keeps the public API
identical to the original implementation.
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
    """Return the sum of *a* and *b*.

    This function is a thin wrapper around :class:`Add`.
    """
    return Add(a, b)()


def subtract(a, b):
    """Return the difference of *a* and *b*.

    This function is a thin wrapper around :class:`Subtract`.
    """
    return Subtract(a, b)()

# End of mathx/basic.py
