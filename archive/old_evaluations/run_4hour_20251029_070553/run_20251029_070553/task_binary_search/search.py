def binary_search(lst, target):
    """Perform binary search on a sorted list.

    Parameters
    ----------
    lst : list
        Sorted list of comparable elements.
    target : any
        Element to search for.

    Returns
    -------
    int
        Index of target in lst if found, otherwise -1.
    """
    left, right = 0, len(lst) - 1
    while left <= right:
        mid = (left + right) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
