"""Sorting algorithms module.

This module provides a simple implementation of the bubble sort algorithm.
"""

from typing import List, TypeVar

T = TypeVar("T")


def bubble_sort(lst: List[T]) -> List[T]:
    """Return a new list containing the elements of *lst* sorted in ascending order.

    The function implements the classic bubble sort algorithm. It makes a copy of
    the input list so that the original list is not modified.

    Parameters
    ----------
    lst : list
        The list to sort.

    Returns
    -------
    list
        A new list with the elements sorted in ascending order.
    """
    # Make a copy so we don't modify the original list
    arr = list(lst)
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                # Swap if the element found is greater than the next element
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# If this module is run directly, demonstrate the function
if __name__ == "__main__":
    sample = [64, 34, 25, 12, 22, 11, 90]
    print("Original list:", sample)
    print("Sorted list:", bubble_sort(sample))
