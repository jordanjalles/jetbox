"""Divide two numbers.

Parameters
----------
x, y : int or float
    Numbers to divide.

Returns
-------
float
    The quotient of x divided by y.

Raises
------
ZeroDivisionError
    If y is zero.
"""

def divide(x, y):
    if y == 0:
        raise ZeroDivisionError("division by zero")
    return x / y
