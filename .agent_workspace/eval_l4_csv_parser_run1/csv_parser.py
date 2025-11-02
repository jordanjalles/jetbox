"""
CSV Parser with header detection and type inference.

Functions:
- parse_csv(file_path, has_header=None)
    Reads a CSV file and returns a list of rows. If has_header is None, the parser
    will try to detect whether the first row is a header by checking if any
    element contains alphabetic characters.

- infer_column_types(rows)
    Given a list of rows (list of lists), infer the type of each column.
    Returns a list of types: int, float, bool, str.

The parser returns a tuple (data, types, header) where:
- data: list of rows (dicts if header, else lists)
- types: list of inferred types per column
- header: list of column names if header detected, else None
"""

import csv
from typing import List, Tuple, Any, Optional


def _is_header(row: List[str]) -> bool:
    """Heuristic: if any cell contains alphabetic characters, treat as header."""
    for cell in row:
        if any(c.isalpha() for c in cell):
            return True
    return False


def _infer_type(values: List[str]) -> type:
    """Infer type for a column based on its values.
    Order of precedence: int > float > bool > str.
    """
    # Try int
    try:
        for v in values:
            if v == "":
                continue
            int(v)
        return int
    except ValueError:
        pass
    # Try float
    try:
        for v in values:
            if v == "":
                continue
            float(v)
        return float
    except ValueError:
        pass
    # Try bool
    bool_vals = {"true", "false", "1", "0"}
    if all(v.lower() in bool_vals or v == "" for v in values):
        return bool
    return str


def infer_column_types(rows: List[List[str]]) -> List[type]:
    """Infer types for each column given rows (list of lists)."""
    if not rows:
        return []
    num_cols = len(rows[0])
    cols = [[] for _ in range(num_cols)]
    for row in rows:
        for i, val in enumerate(row):
            cols[i].append(val)
    return [_infer_type(col) for col in cols]


def parse_csv(file_path: str, has_header: Optional[bool] = None) -> Tuple[List[Any], List[type], Optional[List[str]]]:
    """Parse a CSV file.

    Returns a tuple (data, types, header). If header is None, data is a list of lists.
    If header is present, data is a list of dicts mapping header to value.
    """
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], [], None
    header = None
    if has_header is None:
        header = rows[0] if _is_header(rows[0]) else None
    elif has_header:
        header = rows[0]
    if header:
        data_rows = rows[1:]
    else:
        data_rows = rows
    types = infer_column_types(data_rows)
    if header:
        data = [dict(zip(header, row)) for row in data_rows]
    else:
        data = data_rows
    return data, types, header

# Example usage (uncomment to test manually)
# if __name__ == "__main__":
#     data, types, header = parse_csv("sample.csv")
#     print("Header:", header)
#     print("Types:", types)
#     print("Data:", data)
