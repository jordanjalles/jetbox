"""Read a text file.

Parameters
----------
path : str
    Path to the file.
encoding : str, optional
    Text encoding (default: utf-8).
max_size : int, optional
    Maximum bytes to read (default: 1_000_000).

Returns
-------
str
    File contents.
"""

def read_file(path: str, encoding: str = "utf-8", max_size: int = 1_000_000) -> str:
    with open(path, "r", encoding=encoding) as f:
        data = f.read(max_size)
    return data
