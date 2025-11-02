def delete_file(path: str) -> None:
    """Delete a file.

    Parameters
    ----------
    path: str
        Path to the file to delete.
    """
    import os
    os.remove(path)
