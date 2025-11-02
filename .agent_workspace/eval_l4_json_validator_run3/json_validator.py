"""Simple JSON schema validator.

This module provides a very small subset of JSON Schema validation.
It supports the following schema constructs:

- ``type``: ``object``, ``array``, ``string``, ``number``, ``boolean``
- ``properties``: mapping of property names to subschemas (only for objects)
- ``required``: list of required property names (only for objects)
- ``items``: subschema for array items (only for arrays)

The :func:`validate` function returns ``True`` if the data conforms to the
schema, otherwise it raises :class:`ValueError` with a descriptive message.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

# Type alias for schema representation
Schema = Dict[str, Any]


def _type_check(value: Any, expected_type: str) -> bool:
    """Return ``True`` if *value* matches *expected_type*.

    Supported types are ``string``, ``number``, ``boolean``, ``object`` and
    ``array``.
    """
    type_map = {
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    if expected_type not in type_map:
        raise ValueError(f"Unsupported type '{expected_type}' in schema")
    return isinstance(value, type_map[expected_type])


def _validate_object(data: Any, schema: Schema) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"Expected object, got {type(data).__name__}")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    # Check required properties
    for prop in required:
        if prop not in data:
            raise ValueError(f"Missing required property '{prop}'")
    # Validate each property present in data
    for key, value in data.items():
        if key in properties:
            validate(value, properties[key])
        else:
            # Unknown properties are allowed by default
            pass


def _validate_array(data: Any, schema: Schema) -> None:
    if not isinstance(data, list):
        raise ValueError(f"Expected array, got {type(data).__name__}")
    items_schema = schema.get("items")
    if items_schema is None:
        # No item schema specified, accept any items
        return
    for idx, item in enumerate(data):
        try:
            validate(item, items_schema)
        except ValueError as exc:
            raise ValueError(f"Array item at index {idx} invalid: {exc}") from exc


def validate(data: Any, schema: Schema) -> bool:
    """Validate *data* against *schema*.

    Parameters
    ----------
    data:
        The JSON data to validate.
    schema:
        The schema definition.

    Returns
    -------
    bool
        ``True`` if validation succeeds.

    Raises
    ------
    ValueError
        If validation fails.
    """
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary")
    if "type" not in schema:
        raise ValueError("Schema missing 'type' field")
    schema_type = schema["type"]
    if not _type_check(data, schema_type):
        raise ValueError(f"Expected type '{schema_type}', got {type(data).__name__}")
    if schema_type == "object":
        _validate_object(data, schema)
    elif schema_type == "array":
        _validate_array(data, schema)
    # For primitive types, nothing else to check
    return True

__all__ = ["validate"]
