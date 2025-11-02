import json
import unittest
from json_validator import validate_json

class TestJsonValidator(unittest.TestCase):
    def test_simple_string(self):
        schema = "string"
        self.assertTrue(validate_json("hello", schema))
        self.assertFalse(validate_json(123, schema))

    def test_simple_number(self):
        schema = "number"
        self.assertTrue(validate_json(42, schema))
        self.assertTrue(validate_json(3.14, schema))
        self.assertFalse(validate_json("not a number", schema))

    def test_object_schema(self):
        schema = {
            "name": "string",
            "age": "number",
            "active": "boolean",
            "address": {
                "street": "string",
                "city": "string",
                "zip": "number"
            }
        }
        data = {
            "name": "Alice",
            "age": 30,
            "active": True,
            "address": {
                "street": "123 Main St",
                "city": "Wonderland",
                "zip": 12345
            }
        }
        self.assertTrue(validate_json(data, schema))
        # Missing key
        bad_data = data.copy()
        bad_data.pop("age")
        self.assertFalse(validate_json(bad_data, schema))

    def test_array_schema(self):
        schema = {"items": "number"}
        data = [1, 2, 3]
        self.assertTrue(validate_json(data, schema))
        bad_data = [1, "two", 3]
        self.assertFalse(validate_json(bad_data, schema))

    def test_nested_array_of_objects(self):
        schema = {"items": {"id": "number", "name": "string"}}
        data = [{"id": 1, "name": "Item1"}, {"id": 2, "name": "Item2"}]
        self.assertTrue(validate_json(data, schema))
        bad_data = [{"id": 1, "name": "Item1"}, {"id": "two", "name": "Item2"}]
        self.assertFalse(validate_json(bad_data, schema))

if __name__ == "__main__":
    unittest.main()
