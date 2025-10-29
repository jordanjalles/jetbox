"""Utility functions for list operations.

Functions:
- get_first(lst): Return the first element of a list.
- get_last(lst): Return the last element of a list.
- reverse_list(lst): Return a new list that is the reverse of the input.
"""

from typing import List, TypeVar

T = TypeVar("T")


def get_first(lst: List[T]) -> T:
    """Return the first element of a list.

    Raises:
        IndexError: If the list is empty.
    """
    if not lst:
        raise IndexError("get_first() called on empty list")
    return lst[0]


def get_last(lst: List[T]) -> T:
    """Return the last element of a list.

    Raises:
        IndexError: If the list is empty.
    """
    if not lst:
        raise IndexError("get_last() called on empty list")
    return lst[-1]


def reverse_list(lst: List[T]) -> List[T]:
    """Return a new list that is the reverse of the input list.

    The original list is not modified.
    """
    return lst[::-1]
