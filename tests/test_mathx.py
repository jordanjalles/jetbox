from mathx import add, subtract, multiply, divide, square_root

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2

def test_multiply():
    assert multiply(3, 4) == 12

def test_divide():
    assert divide(10, 2) == 5

def test_square_root():
    assert square_root(9) == 3
    assert square_root(0) == 0
    import math
    assert math.isclose(square_root(2), math.sqrt(2))
    import pytest
    with pytest.raises(ValueError):
        square_root(-1)
