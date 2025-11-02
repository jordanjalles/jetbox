import pytest
from csv_parser import parse_csv

# Helper to create temporary CSV content

def write_temp_csv(tmp_path, content: str):
    file_path = tmp_path / "test.csv"
    file_path.write_text(content)
    return str(file_path)

# Test header detection

def test_header_detection(tmp_path):
    content = "name,age,score\nAlice,30,85.5\nBob,25,90"
    file_path = write_temp_csv(tmp_path, content)
    data, types, header = parse_csv(file_path)
    assert header == ["name", "age", "score"]
    assert len(data) == 2
    assert data[0]["name"] == "Alice"
    assert types == [str, int, float]

# Test no header

def test_no_header(tmp_path):
    content = "Alice,30,85.5\nBob,25,90"
    file_path = write_temp_csv(tmp_path, content)
    data, types, header = parse_csv(file_path, has_header=False)
    assert header is None
    assert len(data) == 2
    assert data[0] == ["Alice", "30", "85.5"]
    assert types == [str, int, float]

# Test boolean inference

def test_bool_inference(tmp_path):
    content = "flag\ntrue\nfalse\n1\n0"
    file_path = write_temp_csv(tmp_path, content)
    data, types, header = parse_csv(file_path)
    assert header == ["flag"]
    assert types == [bool]
    assert data[0]["flag"] == "true"

# Test mixed empty values

def test_empty_values(tmp_path):
    content = "id,value\n1,\n2,3.5\n3,4"
    file_path = write_temp_csv(tmp_path, content)
    data, types, header = parse_csv(file_path)
    assert header == ["id", "value"]
    assert types == [int, float]
    assert data[1]["value"] == "3.5"
    assert data[0]["value"] == ""

# Run tests via pytest
if __name__ == "__main__":
    pytest.main(["-q", "-x", __file__])
