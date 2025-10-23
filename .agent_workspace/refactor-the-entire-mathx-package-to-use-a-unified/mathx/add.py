"""Addition operation.

This module defines the :class:`Add` class which inherits from
:class:`mathx.base.MathOperation`.  It implements the ``compute``
method to return the sum of the two operands.  A convenience function
``add`` is also provided to keep the public API identical to the
original implementation.
"""

from .base import MathOperation


class Add(MathOperation):
    def compute(self):
        return self.a + self.b


# Public helper function

def add(a, b):
    """Return the sum of *a* and *b*.

    This function is a thin wrapper around :class:`Add`.
    """
    return Add(a, b)()

# End of mathx/add.py
