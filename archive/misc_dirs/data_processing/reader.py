import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any


def read_file(file_path: str | Path) -> List[Dict[str, Any]]:
    """Read a CSV, JSON, or XML file and return a list of dictionaries.

    Parameters
    ----------
    file_path: str or Path
        Path to the file.

    Returns
    -------
    List[Dict[str, Any]]
        List of records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = path.suffix.lower()
    if ext == ".csv":
        return _read_csv(path)
    elif ext == ".json":
        return _read_json(path)
    elif ext == ".xml":
        return _read_xml(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("JSON root must be a list or dict")


def _read_xml(path: Path) -> List[Dict[str, Any]]:
    tree = ET.parse(path)
    root = tree.getroot()
    records = []
    for item in root:
        record = {}
        for child in item:
            record[child.tag] = child.text
        records.append(record)
    return records

