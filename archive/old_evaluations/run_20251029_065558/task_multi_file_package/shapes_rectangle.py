class Rectangle:
    """Represent a rectangle with a given width and height.

    Parameters
    ----------
    width : float
        Width of the rectangle.
    height : float
        Height of the rectangle.
    """

    def __init__(self, width: float, height: float):
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        self.width = width
        self.height = height

    def area(self) -> float:
        """Return the area of the rectangle.

        Formula: width * height
        """
        return self.width * self.height

    def __repr__(self) -> str:
        return f"Rectangle(width={self.width}, height={self.height})"
