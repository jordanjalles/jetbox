"""Append *content* to *path*, creating the file if it does not exist.

The function creates parent directories if they do not exist.
"""

from pathlib import Path


def append_file(path: str, content: str) -> None:
    """Append *content* to *path*.

    Parameters
    ----------
    path: str
        Destination file path.
    content: str
        Text to append.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(p.read_text(encoding="utf-8") + content, encoding="utf-8")
