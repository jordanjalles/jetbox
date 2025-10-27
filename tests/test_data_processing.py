import json
import csv
import xml.etree.ElementTree as ET
import os
from pathlib import Path

import pytest

from data_processing.reader import read_file
from data_processing.converter import convert

# Helper to create sample files

def create_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def create_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def create_xml(path, data):
    root = ET.Element("root")
    for record in data:
        item = ET.SubElement(root, "item")
        for k, v in record.items():
            child = ET.SubElement(item, k)
            child.text = str(v)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


@pytest.fixture
def sample_data():
    return [
        {"name": "Alice", "age": "30", "city": "NY"},
        {"name": "Bob", "age": "25", "city": "LA"},
    ]


def test_read_csv(tmp_path, sample_data):
    csv_path = tmp_path / "sample.csv"
    create_csv(csv_path, sample_data)
    result = read_file(csv_path)
    assert result == sample_data


def test_read_json(tmp_path, sample_data):
    json_path = tmp_path / "sample.json"
    create_json(json_path, sample_data)
    result = read_file(json_path)
    assert result == sample_data


def test_read_xml(tmp_path, sample_data):
    xml_path = tmp_path / "sample.xml"
    create_xml(xml_path, sample_data)
    result = read_file(xml_path)
    # XML parsing returns strings, same as sample_data
    assert result == sample_data


def test_convert_csv_to_json(tmp_path, sample_data):
    csv_path = tmp_path / "sample.csv"
    json_path = tmp_path / "output.json"
    create_csv(csv_path, sample_data)
    convert(csv_path, json_path, "json")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == sample_data


def test_convert_json_to_xml(tmp_path, sample_data):
    json_path = tmp_path / "sample.json"
    xml_path = tmp_path / "output.xml"
    create_json(json_path, sample_data)
    convert(json_path, xml_path, "xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []
    for item in root:
        record = {child.tag: child.text for child in item}
        records.append(record)
    assert records == sample_data


def test_convert_xml_to_csv(tmp_path, sample_data):
    xml_path = tmp_path / "sample.xml"
    csv_path = tmp_path / "output.csv"
    create_xml(xml_path, sample_data)
    convert(xml_path, csv_path, "csv")
    result = read_file(csv_path)
    assert result == sample_data

