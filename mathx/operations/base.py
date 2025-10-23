"""Base class for all math operations.

The :class:`MathOperation` class implements the ``__call__`` protocol
so that subclasses can be used as simple callables.  Subclasses must
implement :meth:`compute` which performs the actual calculation.
"""

from __future__ import annotations


class MathOperation:
    """Base class for arithmetic operations.

    Subclasses should implement :meth:`compute` which receives the
    arguments passed to the instance when called.
    """

    def compute(self, *args, **kwargs):  # pragma: no cover - to be overridden
        """Perform the operation.

        Subclasses must override this method.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Delegate to :meth:`compute`.

        This allows an instance to be used like a function.
        """
        return self.compute(*args, **kwargs)

# End of file
