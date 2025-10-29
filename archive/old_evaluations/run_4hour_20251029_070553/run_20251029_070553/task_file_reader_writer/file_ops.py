"""Utility functions for file operations.

This module provides two simple helper functions:

* :func:`write_file` – writes a string to a file, creating any missing
  parent directories.
* :func:`read_file` – reads the entire contents of a file and returns it
  as a string.

Both functions use UTF‑8 encoding and raise the underlying exception if
the operation fails.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["write_file", "read_file"]


def write_file(path: str | os.PathLike[str], content: str) -> None:
    """Write *content* to *path*, creating parent directories if needed.

    Parameters
    ----------
    path:
        The file path to write to.
    content:
        The string content to write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def read_file(path: str | os.PathLike[str]) -> str:
    """Return the entire contents of *path* as a string.

    Parameters
    ----------
    path:
        The file path to read.
    """
    p = Path(path)
    return p.read_text(encoding="utf-8")
