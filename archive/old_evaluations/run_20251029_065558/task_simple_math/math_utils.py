"""Utility math functions.

Provides simple arithmetic operations with basic error handling.
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
    return float(a) + float(b)


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
    return float(a) - float(b)


def multiply(a: float | int, b: float | int) -> float:
    """Return the product of *a* and *b*.

    Parameters
    ----------
    a, b : int | float
        Operands.

    Returns
    -------
    float
        The result of ``a * b``.
    """
    return float(a) * float(b)


def divide(a: float | int, b: float | int) -> float:
    """Return the quotient ``a / b``.

    Parameters
    ----------
    a, b : int | float
        Operands.

    Returns
    -------
    float
        The result of ``a / b``.

    Raises
    ------
    ZeroDivisionError
        If *b* is zero.
    """
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return float(a) / float(b)

__all__ = ["add", "subtract", "multiply", "divide"]
