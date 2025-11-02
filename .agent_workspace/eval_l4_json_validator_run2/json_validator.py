"""Simple JSON schema validator.

This module provides a minimal JSON schema validator that supports a subset of
the JSON Schema specification. It is intentionally lightweight and does not
require external dependencies.

Supported schema features:
- ``type``: ``object``, ``array``, ``string``, ``number``, ``boolean``, ``null``
- ``properties``: mapping of property names to subschemas (for objects)
- ``required``: list of required property names (for objects)
- ``items``: subschema for array items (for arrays)

The :func:`validate` function raises :class:`ValueError` if the data does not
conform to the schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union


class ValidationError(ValueError):
    """Raised when validation fails."""


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate_type(value: Any, expected_type: str) -> None:
    actual_type = _type_name(value)
    if actual_type != expected_type:
        raise ValidationError(f"Expected type '{expected_type}', got '{actual_type}'")


def _validate_object(value: Dict[str, Any], schema: Dict[str, Any]) -> None:
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required properties
    for prop in required:
        if prop not in value:
            raise ValidationError(f"Missing required property '{prop}'")

    # Validate each property present in the object
    for key, val in value.items():
        if key in properties:
            _validate(val, properties[key])
        else:
            # Unknown properties are allowed by default
            pass


def _validate_array(value: List[Any], schema: Dict[str, Any]) -> None:
    items_schema = schema.get("items")
    if items_schema is None:
        # No items schema means any items are allowed
        return
    for idx, item in enumerate(value):
        try:
            _validate(item, items_schema)
        except ValidationError as e:
            raise ValidationError(f"Item at index {idx} invalid: {e}") from e


def _validate(value: Any, schema: Dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        raise ValidationError("Schema must be a dict")
    expected_type = schema.get("type")
    if expected_type is None:
        raise ValidationError("Schema missing 'type' field")

    _validate_type(value, expected_type)

    if expected_type == "object":
        _validate_object(value, schema)
    elif expected_type == "array":
        _validate_array(value, schema)
    # Primitive types have no further validation


def validate(data: Any, schema: Dict[str, Any]) -> None:
    """Validate *data* against *schema*.

    Raises :class:`ValidationError` if validation fails.
    """
    _validate(data, schema)


__all__ = ["validate", "ValidationError"]
