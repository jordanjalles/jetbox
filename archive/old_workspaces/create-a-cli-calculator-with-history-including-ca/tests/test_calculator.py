import pytest

from calculator import Calculator


@pytest.fixture
def calc():
    return Calculator()


def test_simple_calculation(calc):
    assert calc.calculate("2+3") == 5
    assert calc.calculate("10/2") == 5.0


def test_history(calc):
    calc.calculate("1+1")
    calc.calculate("2*3")
    history = calc.history
    assert len(history) == 2
    assert history[0] == ("1+1", 2)
    assert history[1] == ("2*3", 6)


def test_clear_history(calc):
    calc.calculate("4-2")
    assert len(calc.history) == 1
    calc.clear_history()
    assert calc.history == []


def test_invalid_expression(calc):
    with pytest.raises(ValueError):
        calc.calculate("import os")
    with pytest.raises(ValueError):
        calc.calculate("__import__('os')")
    with pytest.raises(ValueError):
        calc.calculate("2 + unknown_var")


def test_division_by_zero(calc):
    with pytest.raises(ValueError):
        calc.calculate("1/0")
