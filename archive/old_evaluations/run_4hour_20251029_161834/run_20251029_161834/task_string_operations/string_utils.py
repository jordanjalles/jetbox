"""Utility functions for string manipulation.

Functions
---------
- uppercase(s): Return the string in all uppercase.
- lowercase(s): Return the string in all lowercase.
- reverse_string(s): Return the string reversed.
- count_vowels(s): Return the number of vowels (a, e, i, o, u) in the string.
"""

VOWELS = set("aeiouAEIOU")


def uppercase(s: str) -> str:
    """Return the string in all uppercase."""
    return s.upper()


def lowercase(s: str) -> str:
    """Return the string in all lowercase."""
    return s.lower()


def reverse_string(s: str) -> str:
    """Return the string reversed."""
    return s[::-1]


def count_vowels(s: str) -> int:
    """Return the number of vowels in the string."""
    return sum(1 for char in s if char in VOWELS)

__all__ = ["uppercase", "lowercase", "reverse_string", "count_vowels"]
