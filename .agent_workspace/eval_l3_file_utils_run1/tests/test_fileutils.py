"""Tests for the fileutils package.

The tests create a temporary directory using the tempfile module and
exercise read, write, append, and delete operations.
"""

import os
import tempfile
import shutil

from fileutils import read_file, write_file, append_file, delete_file


def test_write_and_read(tmp_path):
    file_path = tmp_path / "test.txt"
    write_file(str(file_path), "Hello")
    assert read_file(str(file_path)) == "Hello"


def test_append(tmp_path):
    file_path = tmp_path / "test.txt"
    write_file(str(file_path), "Hello")
    append_file(str(file_path), ", World!")
    assert read_file(str(file_path)) == "Hello, World!"


def test_delete(tmp_path):
    file_path = tmp_path / "test.txt"
    write_file(str(file_path), "Hello")
    delete_file(str(file_path))
    try:
        read_file(str(file_path))
        assert False, "File should have been deleted"
    except FileNotFoundError:
        pass

# Run tests if executed as a script
if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
