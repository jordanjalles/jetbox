"""Divide two numbers.

Parameters
----------
    a : int or float
        Dividend.
    b : int or float
        Divisor.

Returns
-------
    float
        Quotient a / b.

Raises
------
    ZeroDivisionError
        If b is zero.
"""

def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b
