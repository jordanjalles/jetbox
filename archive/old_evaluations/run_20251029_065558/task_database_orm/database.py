"""
Simple ORM using SQLite.

Features:
- Base Model class with automatic table creation.
- Database connection manager (singleton).
- CRUD methods: save(), find(), delete().
- Example User model.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple

# Global connection manager
class Database:
    """Singleton database connection manager."""
    _instance: Optional["Database"] = None
    _connection: Optional[sqlite3.Connection] = None

    def __new__(cls, db_path: str = ":memory:"):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._db_path = db_path
            cls._instance._connection = sqlite3.connect(db_path)
            cls._instance._connection.row_factory = sqlite3.Row
        return cls._instance

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cur = self._connection.cursor()
        cur.execute(sql, params)
        self._connection.commit()
        return cur

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
            Database._instance = None

# Base model
T = TypeVar("T", bound="Model")

class ModelMeta(type):
    """Metaclass to automatically create tables based on class attributes."""

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name == "Model":
            return cls
        # Collect fields: attributes that are not callable and not private
        fields: Dict[str, str] = {}
        for key, value in namespace.items():
            if key.startswith("_") or callable(value):
                continue
            # Simple type mapping
            if isinstance(value, int):
                fields[key] = "INTEGER"
            elif isinstance(value, float):
                fields[key] = "REAL"
            elif isinstance(value, bool):
                fields[key] = "INTEGER"
            else:
                fields[key] = "TEXT"
        cls._fields = fields
        cls._table = name.lower()
        # Create table if not exists
        db = Database()
        columns = [f"{name} {type_}" for name, type_ in fields.items()]
        columns.append("id INTEGER PRIMARY KEY AUTOINCREMENT")
        sql = f"CREATE TABLE IF NOT EXISTS {cls._table} ({', '.join(columns)})"
        db.execute(sql)
        return cls

class Model(metaclass=ModelMeta):
    """Base class for all models."""

    def __init__(self, **kwargs):
        for field in self._fields:
            setattr(self, field, kwargs.get(field))
        self.id: Optional[int] = kwargs.get("id")

    def save(self) -> None:
        db = Database()
        if getattr(self, "id", None) is None:
            # Insert
            fields = [f for f in self._fields if getattr(self, f) is not None]
            placeholders = ",".join(["?" for _ in fields])
            values = tuple(getattr(self, f) for f in fields)
            sql = f"INSERT INTO {self._table} ({', '.join(fields)}) VALUES ({placeholders})"
            cur = db.execute(sql, values)
            self.id = cur.lastrowid
        else:
            # Update
            fields = [f for f in self._fields if getattr(self, f) is not None]
            assignments = ",".join([f"{f} = ?" for f in fields])
            values = tuple(getattr(self, f) for f in fields)
            sql = f"UPDATE {self._table} SET {assignments} WHERE id = ?"
            db.execute(sql, values + (self.id,))

    @classmethod
    def find(cls: Type[T], **kwargs) -> List[T]:
        db = Database()
        conditions = []
        values: Tuple[Any, ...] = ()
        for key, value in kwargs.items():
            if key not in cls._fields:
                continue
            conditions.append(f"{key} = ?")
            values += (value,)
        where = " AND ".join(conditions) if conditions else "1"
        sql = f"SELECT * FROM {cls._table} WHERE {where}"
        cur = db.execute(sql, values)
        rows = cur.fetchall()
        results: List[T] = []
        for row in rows:
            data = {field: row[field] for field in cls._fields}
            data["id"] = row["id"]
            results.append(cls(**data))
        return results

    def delete(self) -> None:
        if getattr(self, "id", None) is None:
            return
        db = Database()
        sql = f"DELETE FROM {self._table} WHERE id = ?"
        db.execute(sql, (self.id,))
        self.id = None

# Example User model
class User(Model):
    name = ""
    email = ""
    age = 0

# Usage example (uncomment to test)
# if __name__ == "__main__":
#     db = Database("example.db")
#     u = User(name="Alice", email="alice@example.com", age=30)
#     u.save()
#     print("Saved user with id", u.id)
#     users = User.find(name="Alice")
#     print("Found users:", users)
#     u.delete()
#     db.close()
