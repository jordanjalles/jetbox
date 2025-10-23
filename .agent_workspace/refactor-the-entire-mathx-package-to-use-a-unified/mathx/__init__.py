"""MathX package – public API.

This top-level package re-exports the arithmetic functions from the
refactored implementation located in
``refactor-the-entire-mathx-package-to-use-a-unified/mathx``.

The tests import ``add``, ``subtract``, ``multiply`` and ``divide``
directly from ``mathx``.
"""

from .basic import add, subtract
from .advanced import multiply, divide

__all__ = ["add", "subtract", "multiply", "divide"]
