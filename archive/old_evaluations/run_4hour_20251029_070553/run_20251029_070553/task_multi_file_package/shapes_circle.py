"""Circle geometry functions.

Functions
---------
area(radius)
    Return the area of a circle with the given radius.

circumference(radius)
    Return the circumference of a circle with the given radius.

Both functions use the value of pi from the math module.
"""

from math import pi


def area(radius: float) -> float:
    """Return the area of a circle.

    Parameters
    ----------
    radius : float
        Radius of the circle.

    Returns
    -------
    float
        The area of the circle.
    """
    return pi * radius ** 2


def circumference(radius: float) -> float:
    """Return the circumference of a circle.

    Parameters
    ----------
    radius : float
        Radius of the circle.

    Returns
    -------
    float
        The circumference of the circle.
    """
    return 2 * pi * radius

__all__ = ["area", "circumference"]
