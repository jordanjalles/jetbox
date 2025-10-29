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

    Parameters
    ----------
    c : float | int
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Temperature in degrees Fahrenheit.
    """
    return c * 9 / 5 + 32


def fahrenheit_to_celsius(f: float | int) -> float:
    """Convert Fahrenheit to Celsius.

    Parameters
    ----------
    f : float | int
        Temperature in degrees Fahrenheit.

    Returns
    -------
    float
        Temperature in degrees Celsius.
    """
    return (f - 32) * 5 / 9
