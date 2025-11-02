"""Divide two numbers.

Parameters
----------
x, y : int or float
    Numerator and denominator.

Returns
-------
float
    The quotient x / y.

Raises
------
ZeroDivisionError
    If y is zero.
"""

def divide(x, y):
    if y == 0:
        raise ZeroDivisionError("division by zero")
    return x / y
