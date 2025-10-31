import pytest
from calculator import Calculator

@pytest.fixture
def calc():
    return Calculator()

def test_add_positive(calc):
    assert calc.add(2, 3) == 5

def test_add_negative(calc):
    assert calc.add(-1, -1) == -2
