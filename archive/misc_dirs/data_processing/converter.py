import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

from .reader import read_file


def convert(input_path: str | Path, output_path: str | Path, output_format: str) -> None:
    """Convert a file from its current format to another format.

    Parameters
    ----------
    input_path: str or Path
        Path to the input file.
    output_path: str or Path
        Path where the converted file will be written.
    output_format: str
        Desired output format: "csv", "json", or "xml".
    """
    data = read_file(input_path)
    out_path = Path(output_path)
    fmt = output_format.lower()
    if fmt == "csv":
        _write_csv(out_path, data)
    elif fmt == "json":
        _write_json(out_path, data)
    elif fmt == "xml":
        _write_xml(out_path, data)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _write_csv(path: Path, data: List[Dict[str, Any]]) -> None:
    if not data:
        raise ValueError("No data to write")
    fieldnames = list(data[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def _write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_xml(path: Path, data: List[Dict[str, Any]]) -> None:
    root = ET.Element("root")
    for record in data:
        item = ET.SubElement(root, "item")
        for k, v in record.items():
            child = ET.SubElement(item, k)
            child.text = str(v)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)

