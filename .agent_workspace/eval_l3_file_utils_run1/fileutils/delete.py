"""Delete *path* if it exists.

The function raises FileNotFoundError if the file does not exist.
"""

from pathlib import Path


def delete_file(path: str) -> None:
    """Delete the file at *path*.

    Parameters
    ----------
    path: str
        Path to delete.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    p.unlink()
