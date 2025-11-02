"""Tests for strutils package.

Each function is tested with a variety of inputs to ensure correct
behaviour.
"""

import pytest

from strutils import capitalize, reverse, count_words

# Capitalize tests
@pytest.mark.parametrize(
    "input,expected",
    [
        ("hello", "Hello"),
        ("Hello", "Hello"),
        ("h", "H"),
        ("", ""),
        ("1abc", "1abc"),
    ],
)
def test_capitalize(input, expected):
    assert capitalize(input) == expected

# Reverse tests
@pytest.mark.parametrize(
    "input,expected",
    [
        ("hello", "olleh"),
        ("a", "a"),
        ("", ""),
        ("12345", "54321"),
    ],
)
def test_reverse(input, expected):
    assert reverse(input) == expected

# Count words tests
@pytest.mark.parametrize(
    "input,expected",
    [
        ("hello world", 2),
        ("  leading and trailing  ", 3),
        ("single", 1),
        ("", 0),
        ("   ", 0),
    ],
)
def test_count_words(input, expected):
    assert count_words(input) == expected
