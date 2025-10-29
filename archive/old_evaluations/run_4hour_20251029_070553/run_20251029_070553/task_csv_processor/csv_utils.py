"""Utility functions for CSV file operations.

This module provides simple helpers to read, write, and filter CSV data.

Functions
---------
read_csv(path)
    Read a CSV file and return a list of rows (each row is a list of strings).

write_csv(path, rows)
    Write a list of rows to a CSV file.

filter_rows(rows, condition)
    Return a new list containing only rows for which the condition(row) is True.

The functions use the built‑in csv module and assume UTF‑8 encoding.
"""

import csv
from typing import List, Callable, Iterable


def read_csv(path: str) -> List[List[str]]:
    """Read a CSV file and return its rows.

    Parameters
    ----------
    path: str
        Path to the CSV file.

    Returns
    -------
    List[List[str]]
        A list where each element is a row represented as a list of strings.
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [row for row in reader]


def write_csv(path: str, rows: Iterable[Iterable[str]]) -> None:
    """Write rows to a CSV file.

    Parameters
    ----------
    path: str
        Destination file path.
    rows: Iterable[Iterable[str]]
        Iterable of rows, each row being an iterable of string values.
    """
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def filter_rows(rows: Iterable[Iterable[str]], condition: Callable[[Iterable[str]], bool]) -> List[List[str]]:
    """Filter rows based on a condition.

    Parameters
    ----------
    rows: Iterable[Iterable[str]]
        Input rows.
    condition: Callable[[Iterable[str]], bool]
        A function that receives a row and returns True if the row should be kept.

    Returns
    -------
    List[List[str]]
        List of rows that satisfy the condition.
    """
    return [list(row) for row in rows if condition(row)]


__all__ = [
    'read_csv',
    'write_csv',
    'filter_rows',
]
