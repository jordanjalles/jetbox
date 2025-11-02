import csv_parser
import os
import tempfile
import json
import pytest

# Helper to create temp csv

def create_csv(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def test_parse_with_header():
    csv_content = "name,age,active\nAlice,30,true\nBob,25,false\n"
    path = create_csv(csv_content)
    try:
        result = csv_parser.parse_csv(path, has_header=True)
        assert result == [
            {"name": "Alice", "age": 30, "active": True},
            {"name": "Bob", "age": 25, "active": False},
        ]
    finally:
        os.unlink(path)


def test_parse_without_header():
    csv_content = "Alice,30,true\nBob,25,false\n"
    path = create_csv(csv_content)
    try:
        result = csv_parser.parse_csv(path, has_header=False)
        assert result == [
            ["Alice", 30, True],
            ["Bob", 25, False],
        ]
    finally:
        os.unlink(path)


def test_empty_file():
    path = create_csv("")
    try:
        result = csv_parser.parse_csv(path, has_header=True)
        assert result == []
    finally:
        os.unlink(path)


def test_infer_types():
    csv_content = "num,float,bool,str\n1,2.5,True,hello\n"
    path = create_csv(csv_content)
    try:
        result = csv_parser.parse_csv(path, has_header=True)
        assert result == [{"num": 1, "float": 2.5, "bool": True, "str": "hello"}]
    finally:
        os.unlink(path)

if __name__ == "__main__":
    pytest.main([__file__])
