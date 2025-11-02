"""File utilities package providing read, write, append, and delete operations.

The package exposes four functions:

- read_file(path: str) -> str
- write_file(path: str, content: str) -> None
- append_file(path: str, content: str) -> None
- delete_file(path: str) -> None

Each function is implemented in its own module for clarity and testability.
"""

from .read import read_file
from .write import write_file
from .append import append_file
from .delete import delete_file

__all__ = [
    "read_file",
    "write_file",
    "append_file",
    "delete_file",
]
