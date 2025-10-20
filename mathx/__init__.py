"""mathx package providing basic arithmetic functions.

This module defines two simple functions:

- :func:`add(a, b)` – returns the sum of ``a`` and ``b``.
- :func:`multiply(a, b)` – returns the product of ``a`` and ``b``.

The functions are intentionally straightforward to keep the example
minimal and to allow the tests to focus on the package structure.
"""

from __future__ import annotations

__all__ = ["add", "multiply"]


def add(a: float | int, b: float | int) -> float | int:
    """Return the sum of *a* and *b*.

    Parameters
    ----------
    a, b:
        Numbers to add.  The function accepts both ``int`` and ``float``
        types and returns the same type as the inputs.

    Returns
    -------
    int | float
        The sum of the two arguments.
    """
    return a + b


def multiply(a: float | int, b: float | int) -> float | int:
    """Return the product of *a* and *b*.

    Parameters
    ----------
    a, b:
        Numbers to multiply.  The function accepts both ``int`` and ``float``
        types and returns the same type as the inputs.

    Returns
    -------
    int | float
        The product of the two arguments.
    """
    return a * b
