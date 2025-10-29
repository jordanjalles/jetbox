"""
A simple command‑line argument parser.

Features
--------
- Flags (boolean switches) e.g. --verbose
- Options with values e.g. --output file.txt
- Positional arguments (remaining tokens)
- Basic error handling for missing values
"""

from __future__ import annotations

import sys
import argparse
from typing import Dict, List, Tuple, Union


class ArgumentError(Exception):
    """Raised when the command line is malformed."""
    pass


class Parser:
    """
    A minimal CLI argument parser.

    Usage
    -----
    >>> parser = Parser()
    >>> parser.add_flag("--verbose", help="Enable verbose output")
    >>> parser.add_option("--output", help="Output file", required=True)
    >>> parser.add_positional("input", help="Input file")
    >>> args = parser.parse(["--verbose", "--output", "out.txt", "in.txt"])
    >>> args.verbose
    True
    >>> args.output
    'out.txt'
    >>> args.input
    'in.txt'
    """

    def __init__(self, argv: List[str] | None = None):
        """
        Parameters
        ----------
        argv : list[str] | None
            List of command‑line tokens. If None, ``sys.argv[1:]`` is used.
        """
        self._flags: Dict[str, bool] = {}
        self._options: Dict[str, Tuple[str, bool]] = {}  # name -> (value, required)
        self._positional_names: List[str] = []

        self._argv = argv if argv is not None else sys.argv[1:]

    # ------------------------------------------------------------------
    # Argument registration helpers
    # ------------------------------------------------------------------
    def add_flag(self, name: str, help: str | None = None) -> None:
        """Register a boolean flag."""
        if not name.startswith("--"):
            raise ValueError("Flags must start with '--'")
        self._flags[name] = False

    def add_option(self, name: str, help: str | None = None, required: bool = False) -> None:
        """Register an option that expects a value."""
        if not name.startswith("--"):
            raise ValueError("Options must start with '--'")
        self._options[name] = (None, required)

    def add_positional(self, name: str, help: str | None = None) -> None:
        """Register a positional argument."""
        self._positional_names.append(name)

    # ------------------------------------------------------------------
    # Parsing logic
    # ------------------------------------------------------------------
    def parse(self, argv: List[str] | None = None) -> argparse.Namespace:
        """
        Parse the provided command line.

        Returns
        -------
        argparse.Namespace
            An object with attributes for each registered argument.
        """
        if argv is None:
            argv = self._argv

        # Internal state
        parsed_flags: Dict[str, bool] = {k: False for k in self._flags}
        parsed_options: Dict[str, Union[str, None]] = {k: None for k in self._options}
        positional_values: List[str] = []

        it = iter(argv)
        for token in it:
            if token in self._flags:
                parsed_flags[token] = True
            elif token in self._options:
                # Consume next token as value
                try:
                    value = next(it)
                except StopIteration:
                    raise ArgumentError(f"Option {token} requires a value")
                parsed_options[token] = value
            elif token.startswith("-"):
                raise ArgumentError(f"Unknown option or flag: {token}")
            else:
                positional_values.append(token)

        # Validate required options
        for name, (_, required) in self._options.items():
            if required and parsed_options[name] is None:
                raise ArgumentError(f"Missing required option: {name}")

        # Validate positional count
        if len(positional_values) != len(self._positional_names):
            raise ArgumentError(
                f"Expected {len(self._positional_names)} positional arguments, "
                f"got {len(positional_values)}"
            )

        # Build namespace
        ns = argparse.Namespace()
        for name, value in parsed_flags.items():
            attr = name.lstrip("-").replace("-", "_")
            setattr(ns, attr, value)
        for name, value in parsed_options.items():
            attr = name.lstrip("-").replace("-", "_")
            setattr(ns, attr, value)
        for name, value in zip(self._positional_names, positional_values):
            setattr(ns, name, value)

        return ns


# ----------------------------------------------------------------------
# Example usage (uncomment to test manually)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = Parser()
    parser.add_flag("--verbose")
    parser.add_option("--output", required=True)
    parser.add_positional("input")

    try:
        args = parser.parse()
        print("Parsed arguments:", args)
    except ArgumentError as e:
        print("Error:", e)
        sys.exit(1)
