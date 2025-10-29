"""Utility functions for list operations.

Functions
---------
- get_first(lst): Return the first element of a list or None if the list is empty.
- get_last(lst): Return the last element of a list or None if the list is empty.
- reverse_list(lst): Return a new list that is the reverse of the input list.
"""

from typing import List, Optional, TypeVar

T = TypeVar("T")


def get_first(lst: List[T]) -> Optional[T]:
    """Return the first element of *lst* or ``None`` if *lst* is empty.

    Parameters
    ----------
    lst : list
        The list to inspect.

    Returns
    -------
    element or None
        The first element or ``None`` if the list is empty.
    """
    return lst[0] if lst else None


def get_last(lst: List[T]) -> Optional[T]:
    """Return the last element of *lst* or ``None`` if *lst* is empty.

    Parameters
    ----------
    lst : list
        The list to inspect.

    Returns
    -------
    element or None
        The last element or ``None`` if the list is empty.
    """
    return lst[-1] if lst else None


def reverse_list(lst: List[T]) -> List[T]:
    """Return a new list that is the reverse of *lst*.

    Parameters
    ----------
    lst : list
        The list to reverse.

    Returns
    -------
    list
        A new list containing the elements of *lst* in reverse order.
    """
    return list(reversed(lst))

__all__ = ["get_first", "get_last", "reverse_list"]
