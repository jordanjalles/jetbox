"""Tests for the mathx package.

Each operation is tested with a few simple cases.
"""

import pytest
from mathx import add, subtract, multiply, divide

@pytest.mark.parametrize("x,y,expected", [
    (1, 2, 3),
    (-1, 5, 4),
    (0, 0, 0),
])
def test_add(x, y, expected):
    assert add(x, y) == expected

@pytest.mark.parametrize("x,y,expected", [
    (5, 3, 2),
    (0, 5, -5),
    (-2, -2, 0),
])
def test_subtract(x, y, expected):
    assert subtract(x, y) == expected

@pytest.mark.parametrize("x,y,expected", [
    (2, 3, 6),
    (-1, 5, -5),
    (0, 10, 0),
])
def test_multiply(x, y, expected):
    assert multiply(x, y) == expected

@pytest.mark.parametrize("x,y,expected", [
    (6, 3, 2),
    (5, 2, 2.5),
    (-4, 2, -2),
])
def test_divide(x, y, expected):
    assert divide(x, y) == expected

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
