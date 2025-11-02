"""Test suite for :mod:`csv_parser`.

The tests cover:
* Header detection – both when a header is present and when it is absent.
* Type inference – integers, floats, and strings.
* The parser's return types (list of dicts vs list of lists).
"""

import pathlib
import tempfile
import textwrap

import pytest

from csv_parser import parse_csv

# Helper to create a temporary CSV file

def _make_csv(content: str) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    tmp.write(content)
    tmp.close()
    return pathlib.Path(tmp.name)


class TestHeaderDetection:
    def test_header_present(self):
        csv = _make_csv("""name,age,score\nAlice,30,85.5\nBob,25,90\n""")
        data, types = parse_csv(csv)
        assert isinstance(data[0], dict)
        assert set(data[0].keys()) == {"name", "age", "score"}
        assert types["age"] == "int"
        assert types["score"] == "float"

    def test_no_header(self):
        csv = _make_csv("""Alice,30,85.5\nBob,25,90\n""")
        data, types = parse_csv(csv)
        assert isinstance(data[0], list)
        assert types[0] == "str"
        assert types[1] == "int"
        assert types[2] == "float"

    def test_empty_file(self):
        csv = _make_csv("")
        data, types = parse_csv(csv)
        assert data == []
        assert types == {}


class TestTypeInference:
    def test_infer_int(self):
        csv = _make_csv("""1,2,3\n4,5,6\n""")
        _, types = parse_csv(csv)
        assert all(t == "int" for t in types.values())

    def test_infer_float(self):
        csv = _make_csv("""1.1,2.2,3.3\n4.4,5.5,6.6\n""")
        _, types = parse_csv(csv)
        assert all(t == "float" for t in types.values())

    def test_mixed(self):
        csv = _make_csv("""1,2.5,hello\n3,4.5,world\n""")
        _, types = parse_csv(csv)
        assert types[0] == "int"
        assert types[1] == "float"
        assert types[2] == "str"


# Clean up temporary files after tests

@pytest.fixture(autouse=True)
def cleanup(tmp_path_factory):
    yield
    for f in tmp_path_factory.getbasetemp().glob("*csv"):
        f.unlink()
