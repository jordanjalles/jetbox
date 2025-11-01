import pytest
from mathx import add, subtract, multiply, divide

# Test data
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (-1, 5, 4),
    (0, 0, 0),
])
def test_add(a, b, expected):
    assert add(a, b) == expected

@pytest.mark.parametrize("a,b,expected", [
    (5, 3, 2),
    (0, 5, -5),
    (-2, -3, 1),
])
def test_subtract(a, b, expected):
    assert subtract(a, b) == expected

@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 6),
    (-1, 5, -5),
    (0, 10, 0),
])
def test_multiply(a, b, expected):
    assert multiply(a, b) == expected

@pytest.mark.parametrize("a,b,expected", [
    (6, 3, 2),
    (5, 2, 2.5),
    (-4, -2, 2),
])
def test_divide(a, b, expected):
    assert divide(a, b) == expected

# Test division by zero
def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
