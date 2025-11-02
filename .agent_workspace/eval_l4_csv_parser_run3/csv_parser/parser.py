import csv
from typing import List, Dict, Any, Iterable, Union

# Type inference helpers

def _infer_type(value: str) -> Any:
    """Infer the type of a CSV cell value.
    Tries int, float, bool, else returns string.
    """
    if value == "":
        return None
    # bool
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    # int
    try:
        return int(value)
    except ValueError:
        pass
    # float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _infer_row(row: List[str]) -> List[Any]:
    return [_infer_type(v) for v in row]


def parse_csv(file_path: str, has_header: bool = True) -> List[Union[Dict[str, Any], List[Any]]]:
    """Parse a CSV file.

    Parameters
    ----------
    file_path: str
        Path to the CSV file.
    has_header: bool, optional
        If True, first row is treated as header and returned as dicts.
        If False, rows are returned as lists.

    Returns
    -------
    List[Union[Dict[str, Any], List[Any]]]
        Parsed rows.
    """
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return []
    if has_header:
        header = rows[0]
        data_rows = rows[1:]
        parsed = []
        for row in data_rows:
            parsed.append({h: _infer_type(v) for h, v in zip(header, row)})
        return parsed
    else:
        return [_infer_row(row) for row in rows]

__all__ = ["parse_csv"]
