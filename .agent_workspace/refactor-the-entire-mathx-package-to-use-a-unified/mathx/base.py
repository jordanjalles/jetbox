"""Base module for mathx operations.

This module defines the :class:`MathOperation` base class that all
operations in the :mod:`mathx` package inherit from.  The base class
provides a simple interface that stores the operands and defines a
``compute`` method that must be implemented by subclasses.  The
``__call__`` method forwards to ``compute`` so that an instance can be
used as a callable.
"""

from __future__ import annotations


class MathOperation:
    """Base class for all math operations.

    Subclasses must implement :meth:`compute` which performs the actual
    calculation.  The constructor stores the operands ``a`` and ``b``.
    ``__call__`` simply forwards to :meth:`compute` so that an instance
    can be used as a function.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute(self):  # pragma: no cover - abstract method
        """Return the result of the operation.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement compute()")

    def __call__(self):
        return self.compute()

# End of mathx/base.py
