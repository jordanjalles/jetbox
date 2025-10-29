# db.py
"""
A minimal SQLite wrapper providing basic CRUD operations.

Usage:
    from db import Database

    db = Database("example.db")
    db.create_table(
        "users",
        {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "name": "TEXT NOT NULL",
            "email": "TEXT UNIQUE NOT NULL",
        },
    )
    db.insert("users", {"name": "Alice", "email": "alice@example.com"})
    rows = db.query("SELECT * FROM users WHERE name = ?", ("Alice",))
    db.update("users", {"email": "alice@newdomain.com"}, "name = ?", ("Alice",))
    db.delete("users", "name = ?", ("Alice",))
"""

import sqlite3
from typing import Any, Dict, Iterable, List, Tuple, Union


class Database:
    """
    Simple SQLite database wrapper.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file. If the file does not exist,
        it will be created automatically.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connect()

    def _connect(self) -> None:
        """Establish a new database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self.cursor = self.conn.cursor()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    # ------------------------------------------------------------------
    # Table creation
    # ------------------------------------------------------------------
    def create_table(
        self,
        table_name: str,
        columns: Dict[str, str],
        if_not_exists: bool = True,
    ) -> None:
        """
        Create a table with the given columns.

        Parameters
        ----------
        table_name : str
            Name of the table to create.
        columns : dict
            Mapping of column names to SQLite type definitions.
            Example: {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}.
        if_not_exists : bool, optional
            Whether to add IF NOT EXISTS to the statement.
        """
        clause = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
        stmt = f"CREATE TABLE {'IF NOT EXISTS' if if_not_exists else ''} {table_name} ({clause})"
        self.cursor.execute(stmt)
        self.conn.commit()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def insert(self, table: str, values: Dict[str, Any]) -> int:
        """
        Insert a row into the specified table.

        Parameters
        ----------
        table : str
            Target table.
        values : dict
            Mapping of column names to values.

        Returns
        -------
        int
            The rowid of the inserted row.
        """
        cols = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)
        stmt = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        self.cursor.execute(stmt, tuple(values.values()))
        self.conn.commit()
        return self.cursor.lastrowid

    def query(
        self,
        sql: str,
        params: Union[Tuple[Any, ...], List[Any]] = (),
        fetch_all: bool = True,
    ) -> List[sqlite3.Row]:
        """
        Execute a SELECT query.

        Parameters
        ----------
        sql : str
            The SQL query string.
        params : tuple or list, optional
            Parameters to bind to the query.
        fetch_all : bool, optional
            If True, return all rows; otherwise return a single row.

        Returns
        -------
        list[sqlite3.Row] or sqlite3.Row
            Query result(s).
        """
        self.cursor.execute(sql, params)
        if fetch_all:
            return self.cursor.fetchall()
        return self.cursor.fetchone()

    def update(
        self,
        table: str,
        values: Dict[str, Any],
        where_clause: str,
        where_params: Union[Tuple[Any, ...], List[Any]] = (),
    ) -> int:
        """
        Update rows in a table.

        Parameters
        ----------
        table : str
            Target table.
        values : dict
            Mapping of columns to new values.
        where_clause : str
            WHERE clause without the 'WHERE' keyword.
        where_params : tuple or list, optional
            Parameters for the WHERE clause.

        Returns
        -------
        int
            Number of rows affected.
        """
        set_clause = ", ".join(f"{col} = ?" for col in values)
        stmt = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = tuple(values.values()) + tuple(where_params)
        self.cursor.execute(stmt, params)
        self.conn.commit()
        return self.cursor.rowcount

    def delete(
        self,
        table: str,
        where_clause: str,
        where_params: Union[Tuple[Any, ...], List[Any]] = (),
    ) -> int:
        """
        Delete rows from a table.

        Parameters
        ----------
        table : str
            Target table.
        where_clause : str
            WHERE clause without the 'WHERE' keyword.
        where_params : tuple or list, optional
            Parameters for the WHERE clause.

        Returns
        -------
        int
            Number of rows deleted.
        """
        stmt = f"DELETE FROM {table} WHERE {where_clause}"
        self.cursor.execute(stmt, where_params)
        self.conn.commit()
        return self.cursor.rowcount

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
