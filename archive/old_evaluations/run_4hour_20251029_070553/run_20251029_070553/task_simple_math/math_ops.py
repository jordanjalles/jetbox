"""Simple math operations module.

Provides four basic arithmetic functions: add, subtract, multiply, divide.
"""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def subtract(a, b):
    """Return the difference of a and b (a - b)."""
    return a - b


def multiply(a, b):
    """Return the product of a and b."""
    return a * b


def divide(a, b):
    """Return the quotient of a divided by b.

    Raises a ZeroDivisionError if b is zero.
    """
    return a / b
