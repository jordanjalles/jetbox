import json
import pytest
from json_validator import validate

# Sample schemas and data
SCHEMA_OBJECT = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age"],
}

SCHEMA_ARRAY = {
    "type": "array",
    "items": {"type": "number"},
}

SCHEMA_PRIMITIVE = {"type": "string"}

# Test cases

def test_valid_object():
    data = {"name": "Alice", "age": 30, "tags": ["admin", "user"]}
    assert validate(data, SCHEMA_OBJECT)


def test_missing_required():
    data = {"name": "Bob"}
    with pytest.raises(ValueError, match="Missing required property 'age'"):
        validate(data, SCHEMA_OBJECT)


def test_invalid_type():
    data = {"name": "Carol", "age": "thirty"}
    with pytest.raises(ValueError, match="Expected type 'number'"):
        validate(data, SCHEMA_OBJECT)


def test_valid_array():
    data = [1, 2, 3]
    assert validate(data, SCHEMA_ARRAY)


def test_array_item_invalid():
    data = [1, "two", 3]
    with pytest.raises(ValueError, match="Array item at index 1 invalid"):
        validate(data, SCHEMA_ARRAY)


def test_primitive_valid():
    data = "hello"
    assert validate(data, SCHEMA_PRIMITIVE)


def test_primitive_invalid():
    data = 123
    with pytest.raises(ValueError, match="Expected type 'string'"):
        validate(data, SCHEMA_PRIMITIVE)
