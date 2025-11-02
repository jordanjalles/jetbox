"""Test suite for mathx package.

The tests cover basic arithmetic operations and edge cases such as
division by zero.
"""

import pytest
from mathx import add, subtract, multiply, divide

# Add tests

def test_add_integers():
    assert add(1, 2) == 3

def test_add_floats():
    assert add(1.5, 2.5) == 4.0

# Subtract tests

def test_subtract_integers():
    assert subtract(5, 3) == 2

def test_subtract_floats():
    assert subtract(5.5, 2.2) == 3.3

# Multiply tests

def test_multiply_integers():
    assert multiply(3, 4) == 12

def test_multiply_floats():
    assert multiply(2.5, 4) == 10.0

# Divide tests

def test_divide_integers():
    assert divide(10, 2) == 5

def test_divide_floats():
    assert divide(7.5, 2.5) == 3.0

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
