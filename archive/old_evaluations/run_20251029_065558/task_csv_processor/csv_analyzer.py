"""
CSV Analyzer Module
===================

This module provides a small set of helper functions for working with CSV files.

Functions
---------
* :func:`load_csv` – Load a CSV file into a list of dictionaries.
* :func:`get_column_stats` – Return basic statistics for a numeric column.
* :func:`filter_rows` – Return a subset of rows that satisfy a condition.
* :func:`export_csv` – Write a list of dictionaries back to a CSV file.

Missing values are represented as ``None``.  The functions are tolerant of
different delimiters and quote characters.
"""

from __future__ import annotations

import csv
import statistics
from typing import Any, Callable, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------
Row = Dict[str, Any]
Data = List[Row]
Condition = Callable[[Row], bool]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(filename: str, *, delimiter: str = ",", quotechar: str = '"') -> Data:
    """Load a CSV file.

    Parameters
    ----------
    filename:
        Path to the CSV file.
    delimiter:
        Field delimiter.  Defaults to comma.
    quotechar:
        Quote character.  Defaults to double quote.

    Returns
    -------
    list[dict]
        Each row is represented as a dictionary mapping column names to
        values.  Empty fields are converted to ``None``.
    """
    data: Data = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            cleaned: Row = {k: (v if v != "" else None) for k, v in row.items()}
            data.append(cleaned)
    return data


def get_column_stats(data: Data, column: str) -> Dict[str, Optional[float]]:
    """Return statistics for a numeric column.

    Missing values (``None``) are ignored.  If the column contains no numeric
    values, all statistics are returned as ``None``.
    """
    values: List[float] = []
    for row in data:
        val = row.get(column)
        if val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        values.append(num)

    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None}

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def filter_rows(data: Data, condition: Condition) -> Data:
    """Return rows that satisfy *condition*.

    Parameters
    ----------
    data:
        List of rows.
    condition:
        A callable that receives a row and returns ``True`` if the row should
        be kept.
    """
    return [row for row in data if condition(row)]


def export_csv(data: Data, filename: str, *, delimiter: str = ",", quotechar: str = '"') -> None:
    """Export *data* to a CSV file.

    Parameters
    ----------
    data:
        List of rows to write.
    filename:
        Destination file path.
    delimiter:
        Field delimiter.
    quotechar:
        Quote character.
    """
    if not data:
        raise ValueError("Cannot export empty data set")

    fieldnames = list(data[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, quotechar=quotechar, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            # Convert None back to empty string for CSV output
            cleaned = {k: (v if v is not None else "") for k, v in row.items()}
            writer.writerow(cleaned)

# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------
"""
