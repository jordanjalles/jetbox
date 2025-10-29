"""Utility functions for number checks.

Functions:
- is_even(n): Return True if n is an even integer.
- is_odd(n): Return True if n is an odd integer.
- is_positive(n): Return True if n is greater than zero.
- is_negative(n): Return True if n is less than zero.

All functions accept any numeric type that supports modulo and comparison.
"""

def is_even(n):
    """Return True if n is an even integer.

    Parameters
    ----------
    n : int or numeric
        The number to check.

    Returns
    -------
    bool
        True if n is even, False otherwise.
    """
    try:
        return n % 2 == 0
    except Exception:
        return False


def is_odd(n):
    """Return True if n is an odd integer.

    Parameters
    ----------
    n : int or numeric
        The number to check.

    Returns
    -------
    bool
        True if n is odd, False otherwise.
    """
    try:
        return n % 2 != 0
    except Exception:
        return False


def is_positive(n):
    """Return True if n is greater than zero.

    Parameters
    ----------
    n : numeric
        The number to check.

    Returns
    -------
    bool
        True if n > 0, False otherwise.
    """
    try:
        return n > 0
    except Exception:
        return False


def is_negative(n):
    """Return True if n is less than zero.

    Parameters
    ----------
    n : numeric
        The number to check.

    Returns
    -------
    bool
        True if n < 0, False otherwise.
    """
    try:
        return n < 0
    except Exception:
        return False
