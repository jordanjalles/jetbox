"""Temperature conversion utilities.

This module provides two simple functions for converting temperatures
between Celsius and Fahrenheit.

Functions
---------
celsius_to_fahrenheit(c)
    Convert a temperature from Celsius to Fahrenheit.

fahrenheit_to_celsius(f)
    Convert a temperature from Fahrenheit to Celsius.

Both functions accept a numeric value (int or float) and return a float.
"""

from __future__ import annotations


def celsius_to_fahrenheit(c: float | int) -> float:
    """Convert Celsius to Fahrenheit.

    Formula: F = C * 9/5 + 32
    """
    return c * 9.0 / 5.0 + 32.0


def fahrenheit_to_celsius(f: float | int) -> float:
    """Convert Fahrenheit to Celsius.

    Formula: C = (F - 32) * 5/9
    """
    return (f - 32.0) * 5.0 / 9.0

# If run as a script, demonstrate usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python temp_converter.py <celsius|fahrenheit> <value>")
        sys.exit(1)
    mode, val = sys.argv[1], float(sys.argv[2])
    if mode.lower() == "celsius":
        print(f"{val}°C = {celsius_to_fahrenheit(val):.2f}°F")
    elif mode.lower() == "fahrenheit":
        print(f"{val}°F = {fahrenheit_to_celsius(val):.2f}°C")
    else:
        print("Unknown mode. Use 'celsius' or 'fahrenheit'.")
        sys.exit(1)
