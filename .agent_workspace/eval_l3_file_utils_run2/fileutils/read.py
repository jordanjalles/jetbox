def read_file(path: str, encoding: str = 'utf-8') -> str:
    """Read a text file and return its contents.

    Parameters
    ----------
    path: str
        Path to the file to read.
    encoding: str, optional
        Text encoding to use. Defaults to 'utf-8'.

    Returns
    -------
    str
        The file contents.
    """
    with open(path, 'r', encoding=encoding) as f:
        return f.read()
