"""Simple JSON schema validator.

This module provides a minimal JSON schema validation function that supports
basic type checking for the following JSON types:

- string
- number
- boolean
- null
- object
- array

The schema is represented as a Python dictionary where keys correspond to
JSON object keys and values are either a type string or a nested schema
for objects. For arrays the schema should be a dictionary with a single
key ``"items"`` whose value is the type or nested schema for the array
elements.

The :func:`validate_json` function returns ``True`` if the data matches
the schema and ``False`` otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

# Mapping from schema type string to Python type
_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
    "object": dict,
    "array": list,
}


def _check_type(value: Any, expected_type: str) -> bool:
    """Return True if *value* matches the expected JSON type.

    Parameters
    ----------
    value:
        The value to check.
    expected_type:
        One of the supported type strings.
    """
    py_type = _TYPE_MAP.get(expected_type)
    if py_type is None:
        raise ValueError(f"Unsupported schema type: {expected_type}")
    return isinstance(value, py_type)


def _validate_object(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate a JSON object against a schema.

    All keys defined in the schema must be present in *data* and their
    values must match the corresponding type or nested schema.
    """
    for key, subschema in schema.items():
        if key not in data:
            return False
        if isinstance(subschema, dict):
            # Determine if this is an array schema or nested object
            if "items" in subschema:
                # Array schema
                if not _check_type(data[key], "array"):
                    return False
                item_schema = subschema["items"]
                for item in data[key]:
                    if isinstance(item_schema, dict):
                        if not _validate_object(item, item_schema):
                            return False
                    else:
                        if not _check_type(item, item_schema):
                            return False
            else:
                # Nested object
                if not _check_type(data[key], "object"):
                    return False
                if not _validate_object(data[key], subschema):
                    return False
        else:
            # Primitive type
            if not _check_type(data[key], subschema):
                return False
    return True


def validate_json(data: Any, schema: Any) -> bool:
    """Validate *data* against *schema*.

    Parameters
    ----------
    data:
        The JSON data to validate (typically a dict, list, etc.).
    schema:
        The schema definition. For objects it should be a dict mapping
        keys to type strings or nested schemas. For arrays it should be
        a dict with a single key ``"items"``.

    Returns
    -------
    bool
        ``True`` if *data* conforms to *schema*, ``False`` otherwise.
    """
    if isinstance(schema, dict):
        if "items" in schema:
            # Array schema
            if not _check_type(data, "array"):
                return False
            item_schema = schema["items"]
            for item in data:
                if isinstance(item_schema, dict):
                    if not _validate_object(item, item_schema):
                        return False
                else:
                    if not _check_type(item, item_schema):
                        return False
            return True
        else:
            # Object schema
            if not _check_type(data, "object"):
                return False
            return _validate_object(data, schema)
    else:
        # Primitive schema
        return _check_type(data, schema)


__all__ = ["validate_json"]
