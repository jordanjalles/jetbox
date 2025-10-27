# Advanced math operations
import math

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b

def square_root(x):
    """Return the square root of x.

    Raises ValueError if x is negative.
    """
    if x < 0:
        raise ValueError("math domain error: negative input for square_root")
    return math.sqrt(x)
