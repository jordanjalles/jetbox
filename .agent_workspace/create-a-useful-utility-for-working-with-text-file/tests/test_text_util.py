import os
import tempfile
import shutil
from pathlib import Path

import pytest

from text_util import count_lines, concat_files


def test_count_lines(tmp_path: Path):
    # Create a temporary file with known number of lines
    file_path = tmp_path / "sample.txt"
    lines = ["first line", "second line", "third line"]
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert count_lines(file_path) == len(lines)


def test_concat_files(tmp_path: Path):
    # Create two source files
    src1 = tmp_path / "a.txt"
    src2 = tmp_path / "b.txt"
    src1.write_text("hello\nworld\n", encoding="utf-8")
    src2.write_text("foo\nbar\n", encoding="utf-8")

    out = tmp_path / "out.txt"
    concat_files([src1, src2], out)

    expected = "hello\nworld\nfoo\nbar\n"
    assert out.read_text(encoding="utf-8") == expected


def test_concat_files_nonexistent(tmp_path: Path):
    # Ensure FileNotFoundError is raised for missing file
    src = tmp_path / "missing.txt"
    out = tmp_path / "out.txt"
    with pytest.raises(FileNotFoundError):
        concat_files([src], out)
