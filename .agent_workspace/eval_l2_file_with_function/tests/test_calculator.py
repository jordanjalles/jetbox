import pytest
from calculator import add

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
        (2.5, 3.5, 6.0),
    ],
)
def test_add(a, b, expected):
    assert add(a, b) == expected
