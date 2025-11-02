import pytest
from strutils.capitalize import capitalize
from strutils.reverse import reverse
from strutils.count_words import count_words

@pytest.mark.parametrize("input,expected", [
    ("hello", "Hello"),
    ("hELLO", "Hello"),
    ("", ""),
    ("a", "A"),
])
def test_capitalize(input, expected):
    assert capitalize(input) == expected

@pytest.mark.parametrize("input,expected", [
    ("hello", "olleh"),
    ("", ""),
    ("a", "a"),
    ("123", "321"),
])
def test_reverse(input, expected):
    assert reverse(input) == expected

@pytest.mark.parametrize("input,expected", [
    ("hello world", 2),
    ("", 0),
    ("one", 1),
    ("  multiple   spaces  ", 2),
])
def test_count_words(input, expected):
    assert count_words(input) == expected
