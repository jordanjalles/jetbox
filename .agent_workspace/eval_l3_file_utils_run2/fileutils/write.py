def write_file(path: str, content: str, encoding: str = 'utf-8') -> None:
    """Write content to a file, overwriting if it exists.

    Parameters
    ----------
    path: str
        Path to the file to write.
    content: str
        Text to write.
    encoding: str, optional
        Text encoding to use. Defaults to 'utf-8'.
    """
    with open(path, 'w', encoding=encoding) as f:
        f.write(content)
