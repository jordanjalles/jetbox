def append_file(path: str, content: str, encoding: str = 'utf-8') -> None:
    """Append content to a file.

    Parameters
    ----------
    path: str
        Path to the file to append to.
    content: str
        Text to append.
    encoding: str, optional
        Text encoding to use. Defaults to 'utf-8'.
    """
    with open(path, 'a', encoding=encoding) as f:
        f.write(content)
