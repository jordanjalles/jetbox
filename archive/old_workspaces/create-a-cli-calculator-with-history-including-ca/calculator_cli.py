#!/usr/bin/env python3
"""Entry point for the calculator CLI.

This script imports the :class:`calculator.Calculator` class and runs the
interactive loop defined in :mod:`calculator.cli`.
"""

from calculator.cli import main

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
