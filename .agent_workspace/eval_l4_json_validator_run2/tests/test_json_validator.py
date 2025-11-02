"""Test suite for json_validator.

The tests cover the supported subset of the JSON Schema specification:
- type checking
- object properties and required fields
- array items validation

Each test uses the :func:`validate` function from :mod:`json_validator` and
expects a :class:`ValidationError` when the data does not match the schema.
"""

import pytest

from json_validator import validate, ValidationError

# ---------- Type validation ----------

def test_type_string():
    schema = {"type": "string"}
    validate("hello", schema)


def test_type_number():
    schema = {"type": "number"}
    validate(42, schema)
    validate(3.14, schema)


def test_type_boolean():
    schema = {"type": "boolean"}
    validate(True, schema)
    validate(False, schema)


def test_type_null():
    schema = {"type": "null"}
    validate(None, schema)


def test_type_object():
    schema = {"type": "object"}
    validate({}, schema)
    validate({"a": 1}, schema)


def test_type_array():
    schema = {"type": "array"}
    validate([], schema)
    validate([1, 2, 3], schema)

# ---------- Object validation ----------

def test_object_properties_and_required():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
        "required": ["name"],
    }
    # Valid data
    validate({"name": "Alice", "age": 30}, schema)
    validate({"name": "Bob"}, schema)
    # Missing required
    with pytest.raises(ValidationError):
        validate({"age": 25}, schema)
    # Wrong type
    with pytest.raises(ValidationError):
        validate({"name": 123}, schema)

# ---------- Array validation ----------

def test_array_items():
    schema = {
        "type": "array",
        "items": {"type": "number"},
    }
    validate([1, 2, 3], schema)
    with pytest.raises(ValidationError):
        validate([1, "two", 3], schema)

# ---------- Nested schemas ----------

def test_nested_object_and_array():
    schema = {
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "number"},
                        "name": {"type": "string"},
                    },
                    "required": ["id", "name"],
                },
            },
        },
        "required": ["users"],
    }
    data = {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
    }
    validate(data, schema)
    # Invalid nested data
    bad_data = {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": "two", "name": "Bob"},
        ]
    }
    with pytest.raises(ValidationError):
        validate(bad_data, schema)

# ---------- Unknown properties allowed ----------

def test_unknown_properties_allowed():
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    validate({"x": 1, "y": 2}, schema)  # Should pass

# ---------- Missing type field in schema ----------

def test_missing_type_in_schema():
    schema = {"properties": {"x": {"type": "number"}}}
    with pytest.raises(ValidationError):
        validate({"x": 1}, schema)

# ---------- Schema not a dict ----------

def test_schema_not_dict():
    with pytest.raises(ValidationError):
        validate(1, "not a dict")
