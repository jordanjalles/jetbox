"""Utility functions for working with text files.

This module provides simple helpers for common text file operations.
"""

from pathlib import Path
from typing import Iterable
import shutil


def count_lines(file_path: Path | str) -> int:
    """Return the number of lines in *file_path*.

    Parameters
    ----------
    file_path:
        Path to the file to count.

    Returns
    -------
    int
        Number of lines in the file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("rb") as f:
        return sum(1 for _ in f)


def concat_files(source_files: Iterable[Path | str], destination: Path | str) -> None:
    """Concatenate *source_files* into *destination*.

    Parameters
    ----------
    source_files:
        Iterable of paths to source files.
    destination:
        Path to the destination file.
    """
    dest_path = Path(destination)
    with dest_path.open("wb") as out_f:
        for src in source_files:
            src_path = Path(src)
            if not src_path.is_file():
                raise FileNotFoundError(f"Source file not found: {src_path}")
            with src_path.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f)

# End of file
