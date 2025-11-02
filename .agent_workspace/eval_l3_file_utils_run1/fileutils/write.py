"""Write *content* to *path*, overwriting any existing file.

The function creates parent directories if they do not exist.
"""

from pathlib import Path


def write_file(path: str, content: str) -> None:
    """Write *content* to *path*, overwriting if it exists.

    Parameters
    ----------
    path: str
        Destination file path.
    content: str
        Text to write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
