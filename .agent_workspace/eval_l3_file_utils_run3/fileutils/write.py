"""Write a text file.

Parameters
----------
path : str
    Path to the file.
content : str
    Content to write.
encoding : str, optional
    Text encoding (default: utf-8).
overwrite : bool, optional
    If False and file exists, raise FileExistsError.
"""

def write_file(path: str, content: str, encoding: str = "utf-8", overwrite: bool = True) -> None:
    mode = "w" if overwrite else "x"
    with open(path, mode, encoding=encoding) as f:
        f.write(content)
