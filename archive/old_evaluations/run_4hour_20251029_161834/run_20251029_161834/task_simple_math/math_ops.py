"""Simple math operations module.

Provides four basic arithmetic functions: add, subtract, multiply, and divide.
Each function accepts two numeric arguments and returns the result.
"""

from __future__ import annotations


def add(a: float | int, b: float | int) -> float:
    """Return the sum of *a* and *b*.

    Parameters
    ----------
    a, b : int | float
        Operands.

    Returns
    -------
    float
        The result of ``a + b``.
    """
    return a + b


def subtract(a: float | int, b: float | int) -> float:
    """Return the difference ``a - b``.

    Parameters
    ----------
    a, b : int | float
        Operands.

    Returns
    -------
    float
        The result of ``a - b``.
    """
    return a - b


def multiply(a: float | int, b: float | int) -> float:
    """Return the product ``a * b``.

    Parameters
    ----------
    a, b : int | float
        Operands.

    Returns
    -------
    float
        The result of ``a * b``.
    """
    return a * b


def divide(a: float | int, b: float | int) -> float:
    """Return the quotient ``a / b``.

    Parameters
    ----------
    a, b : int | float
        Operands. ``b`` must not be zero.

    Returns
    -------
    float
        The result of ``a / b``.

    Raises
    ------
    ZeroDivisionError
        If ``b`` is zero.
    """
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b

__all__ = ["add", "subtract", "multiply", "divide"]
