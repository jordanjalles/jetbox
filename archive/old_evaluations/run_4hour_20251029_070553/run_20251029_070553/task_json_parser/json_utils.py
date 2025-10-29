"""Utility functions for JSON operations.

This module provides simple helpers for loading, saving, and retrieving values
from JSON data structures.
"""

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Any:
    """Load JSON data from a file.

    Parameters
    ----------
    path: str | Path
        Path to the JSON file.

    Returns
    -------
    Any
        The parsed JSON data.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    """Save data as JSON to a file.

    Parameters
    ----------
    path: str | Path
        Destination file path.
    data: Any
        Data to serialize.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_value(data: Dict[str, Any], key: str) -> Any:
    """Retrieve a value from a dictionary, raising KeyError if missing.

    Parameters
    ----------
    data: dict
        Dictionary to search.
    key: str
        Key to retrieve.

    Returns
    -------
    Any
        The value associated with ``key``.
    """
    if key not in data:
        raise KeyError(f"Key '{key}' not found in data")
    return data[key]
