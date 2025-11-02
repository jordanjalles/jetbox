"""Read a file and return its contents as a string.

The function uses UTF-8 encoding by default and raises a FileNotFoundError
if the file does not exist.
"""

from pathlib import Path


def read_file(path: str) -> str:
    """Return the contents of *path* as a string.

    Parameters
    ----------
    path: str
        Path to the file to read.

    Returns
    -------
    str
        File contents.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")
