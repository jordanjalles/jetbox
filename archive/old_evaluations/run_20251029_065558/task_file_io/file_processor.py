"""Utility functions for file processing.

This module provides simple helpers to write lines to a file, read lines from a file,
and count the number of words in a file.  All functions are intentionally
minimal and use only the Python standard library.

The functions perform basic error handling:

* ``write_lines`` will create the file if it does not exist and overwrite any
  existing content.
* ``read_lines`` will raise a ``FileNotFoundError`` with a clear message if the
  file does not exist.
* ``count_words`` will also raise ``FileNotFoundError`` if the file is missing
  and otherwise returns the total word count.

The module is designed to be straightforward to use in scripts or tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

__all__ = ["write_lines", "read_lines", "count_words"]


def write_lines(filename: str | Path, lines: Iterable[str]) -> None:
    """Write an iterable of lines to *filename*.

    Parameters
    ----------
    filename:
        Path to the file to write.  The file is created if it does not exist.
    lines:
        Iterable of strings.  Each string is written as a separate line.

    The function ensures that a newline is appended to each line.  Existing
    content is overwritten.
    """
    path = Path(filename)
    # Ensure parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def read_lines(filename: str | Path) -> List[str]:
    """Return a list of lines from *filename*.

    Parameters
    ----------
    filename:
        Path to the file to read.

    Returns
    -------
    list[str]
        List of lines without trailing newlines.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def count_words(filename: str | Path) -> int:
    """Count the total number of words in *filename*.

    Words are defined as sequences separated by whitespace.

    Parameters
    ----------
    filename:
        Path to the file to analyze.

    Returns
    -------
    int
        Total word count.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    lines = read_lines(filename)
    return sum(len(line.split()) for line in lines)


# If run as a script, demonstrate basic usage.
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python file_processor.py <file>")
        sys.exit(1)
    file = sys.argv[1]
    try:
        print(f"Word count for {file}: {count_words(file)}")
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)
