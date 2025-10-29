"""sorting.py

This module provides three classic sorting algorithms:

* :func:`bubble_sort` – a simple comparison‑based algorithm that repeatedly
  swaps adjacent elements that are out of order.
* :func:`quick_sort` – a divide‑and‑conquer algorithm that selects a pivot
  and partitions the list into elements less than and greater than the pivot.
* :func:`merge_sort` – another divide‑and‑conquer algorithm that splits the
  list into halves, recursively sorts each half, and then merges the sorted
  halves.

All functions return a new sorted list and leave the input list unchanged.
"""

from __future__ import annotations
from typing import List, TypeVar

T = TypeVar("T")


+def bubble_sort(lst: List[T]) -> List[T]:
+    """Return a new list containing the elements of *lst* sorted using
+    the bubble‑sort algorithm.
+
+    Bubble sort repeatedly steps through the list, compares adjacent
+    elements and swaps them if they are in the wrong order.  After the
+    first pass the largest element is guaranteed to be at the end of the
+    list; after the second pass the second‑largest element is in its
+    final position, and so on.
+
+    The algorithm has a worst‑case and average time complexity of
+    :math:`O(n^2)` and a best‑case complexity of :math:`O(n)` when the
+    input list is already sorted.  It uses :math:`O(1)` additional
+    memory.
+
+    Parameters
+    ----------
+    lst:
+        The list to sort.
+
+    Returns
+    -------
+    List[T]
+        A new list containing the sorted elements.
+    """
+    n = len(lst)
+    # Work on a copy so the original list is not modified.
+    arr = list(lst)
+    for i in range(n):
+        # After each outer loop iteration, the last i elements are in
+        # their final position.
+        swapped = False
+        for j in range(0, n - i - 1):
+            if arr[j] > arr[j + 1]:
+                arr[j], arr[j + 1] = arr[j + 1], arr[j]
+                swapped = True
+        if not swapped:
+            # No swaps means the list is already sorted.
+            break
+    return arr
+
+
+def quick_sort(lst: List[T]) -> List[T]:
+    """Return a new list containing the elements of *lst* sorted using
+    the quick‑sort algorithm.
+
+    Quick‑sort is a divide‑and‑conquer algorithm.  It selects a pivot
+    element (here we use the last element for simplicity), partitions
+    the remaining elements into those less than the pivot and those
+    greater than or equal to the pivot, and then recursively sorts the
+    partitions.  The algorithm has an average time complexity of
+    :math:`O(n\log n)` and a worst‑case complexity of :math:`O(n^2)`
+    (when the pivot choices are poor).  It uses :math:`O(log n)`
+    additional memory on average due to recursion.
+
+    Parameters
+    ----------
+    lst:
+        The list to sort.
+
+    Returns
+    -------
+    List[T]
+        A new list containing the sorted elements.
+    """
+    if len(lst) <= 1:
+        return list(lst)
+
+    pivot = lst[-1]
+    less = [x for x in lst[:-1] if x < pivot]
+    greater_equal = [x for x in lst[:-1] if x >= pivot]
+    return quick_sort(less) + [pivot] + quick_sort(greater_equal)
+
+
+def merge_sort(lst: List[T]) -> List[T]:
+    """Return a new list containing the elements of *lst* sorted using
+    the merge‑sort algorithm.
+
+    Merge‑sort is a stable divide‑and‑conquer algorithm.  It splits the
+    list into two halves, recursively sorts each half, and then merges
+    the two sorted halves into a single sorted list.  The algorithm has
+    a guaranteed time complexity of :math:`O(n\log n)` and uses
+    :math:`O(n)` additional memory for the temporary lists during the
+    merge step.
+
+    Parameters
+    ----------
+    lst:
+        The list to sort.
+
+    Returns
+    -------
+    List[T]
+        A new list containing the sorted elements.
+    """
+    if len(lst) <= 1:
+        return list(lst)
+
+    mid = len(lst) // 2
+    left = merge_sort(lst[:mid])
+    right = merge_sort(lst[mid:])
+
+    # Merge the two sorted halves.
+    merged: List[T] = []
+    i = j = 0
+    while i < len(left) and j < len(right):
+        if left[i] <= right[j]:
+            merged.append(left[i])
+            i += 1
+        else:
+            merged.append(right[j])
+            j += 1
+    # Append any remaining elements.
+    merged.extend(left[i:])
+    merged.extend(right[j:])
+    return merged
+
*** End of File ***