import os
import tempfile
import pytest
from fileutils import read_file, write_file, append_file, delete_file

# Helper to create temp file path

def _temp_path(name='test.txt'):
    return os.path.join(tempfile.gettempdir(), name)


def test_write_and_read(tmp_path):
    path = tmp_path / 'write_read.txt'
    write_file(str(path), 'hello world')
    assert read_file(str(path)) == 'hello world'


def test_append(tmp_path):
    path = tmp_path / 'append.txt'
    write_file(str(path), 'first')
    append_file(str(path), '\nsecond')
    assert read_file(str(path)) == 'first\nsecond'


def test_delete(tmp_path):
    path = tmp_path / 'delete.txt'
    write_file(str(path), 'to be deleted')
    delete_file(str(path))
    assert not os.path.exists(str(path))
