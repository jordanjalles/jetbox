"""Utility functions for list operations.

Functions:
- get_max(lst): Return the maximum value in a list.
- get_min(lst): Return the minimum value in a list.
- get_average(lst): Return the average of numeric values in a list.
- remove_duplicates(lst): Return a new list with duplicates removed, preserving order.
"""

from typing import List, Iterable, TypeVar

T = TypeVar("T")


def get_max(lst: Iterable[T]) -> T:
    """Return the maximum element of *lst*.

    Raises
    ------
    ValueError
        If *lst* is empty.
    """
    try:
        return max(lst)
    except ValueError as e:
        raise ValueError("get_max() arg is an empty sequence") from e


def get_min(lst: Iterable[T]) -> T:
    """Return the minimum element of *lst*.

    Raises
    ------
    ValueError
        If *lst* is empty.
    """
    try:
        return min(lst)
    except ValueError as e:
        raise ValueError("get_min() arg is an empty sequence") from e


def get_average(lst: Iterable[float]) -> float:
    """Return the arithmetic mean of *lst*.

    Raises
    ------
    ValueError
        If *lst* is empty.
    """
    values = list(lst)
    if not values:
        raise ValueError("get_average() arg is an empty sequence")
    return sum(values) / len(values)


def remove_duplicates(lst: Iterable[T]) -> List[T]:
    """Return a new list with duplicates removed, preserving order.

    Parameters
    ----------
    lst : Iterable[T]
        Input iterable.

    Returns
    -------
    List[T]
        List with duplicates removed.
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

__all__ = ["get_max", "get_min", "get_average", "remove_duplicates"]
