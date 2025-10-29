class Circle:
    """Represents a circle with a given radius.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    """

    def __init__(self, radius: float):
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        self.radius = radius

    def area(self) -> float:
        """Return the area of the circle.

        Formula: π * r^2
        """
        import math
        return math.pi * self.radius ** 2

    def __repr__(self) -> str:
        return f"Circle(radius={self.radius})"
