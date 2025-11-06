"""
Fractal generation utilities.

This module provides a small, self‑contained library for rendering
Mandelbrot and Julia sets.  The implementation is intentionally
light‑weight – it relies only on :mod:`numpy` and :mod:`PIL` – and is
designed to be easy to extend with additional fractal types or colour
schemes.

The public API consists of three functions:

* :func:`generate_mandelbrot` – renders a Mandelbrot set.
* :func:`generate_julia` – renders a Julia set.
* :func:`generate_fractal` – dispatcher that selects the appropriate
  renderer based on a ``mode`` string.

All functions return a PNG image as a ``bytes`` object which can be
written to disk or sent directly to a client.

The code is heavily commented to aid maintainability and to serve as a
reference for future extensions (e.g. smooth colouring, GPU
acceleration, multi‑threaded rendering).
"""

from __future__ import annotations

from dataclasses import dataclass

import io
import numpy as np
from PIL import Image

__all__ = [
    "FractalParams",
    "generate_mandelbrot",
    "generate_julia",
    "generate_fractal",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FractalParams:
    """Container for parameters used to generate a fractal image.

    Attributes
    ----------
    width, height : int
        Dimensions of the output image in pixels.
    x, y : float
        Centre of the view in the complex plane.
    zoom : float
        Zoom factor – a value > 1 zooms in, < 1 zooms out.
    max_iter : int
        Maximum number of iterations for the escape‑time algorithm.
    """

    width: int
    height: int
    x: float
    y: float
    zoom: float
    max_iter: int

# ---------------------------------------------------------------------------
# Core algorithm – escape‑time calculation
# ---------------------------------------------------------------------------

def _mandelbrot_escape_time(c: np.ndarray, max_iter: int) -> np.ndarray:
    """Compute the escape time for each point in ``c``.

    Parameters
    ----------
    c : np.ndarray
        2‑D array of complex numbers representing points in the
        complex plane.
    max_iter : int
        Maximum iterations to perform.

    Returns
    -------
    np.ndarray
        Integer array of the same shape as ``c`` containing the
        iteration count at which the magnitude of the sequence exceeded
        2.  Points that never escape are marked with ``max_iter``.
    """
    # ``z`` holds the current value of the iteration.  ``mask`` keeps
    # track of points that are still inside the radius‑2 circle.
    z = np.zeros_like(c, dtype=np.complex128)
    mask = np.full(c.shape, True, dtype=bool)
    escape = np.full(c.shape, max_iter, dtype=np.int32)

    # Iterate until either all points have escaped or we reach the
    # maximum number of iterations.  The loop is written in a way that
    # allows NumPy to operate on the entire array at once, which is
    # considerably faster than a Python ``for`` loop over individual
    # points.
    i = 0
    while i < max_iter and mask.any():
        z[mask] = z[mask] * z[mask] + c[mask]
        mask = np.abs(z) <= 2
        escape[mask] = i
        i += 1
    return escape

# ---------------------------------------------------------------------------
# Public API – image generation
# ---------------------------------------------------------------------------

def _create_image(escape: np.ndarray) -> bytes:
    """Convert an escape‑time array into a PNG image.

    The function maps the iteration counts to a simple grayscale
    palette.  The helper is shared between the Mandelbrot and Julia
    renderers.
    """
    norm = escape / escape.max()
    img_array = (norm * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode="L")
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        return output.getvalue()


def generate_mandelbrot(
    width: int,
    height: int,
    x: float = 0.0,
    y: float = 0.0,
    zoom: float = 1.0,
    max_iter: int = 200,
) -> bytes:
    """Generate a PNG image of the Mandelbrot set.

    Parameters
    ----------
    width, height : int
        Dimensions of the output image.
    x, y : float, optional
        Centre of the view in the complex plane.
    zoom : float, optional
        Zoom factor – a value > 1 zooms in.
    max_iter : int, optional
        Maximum iterations for the escape‑time algorithm.

    Returns
    -------
    bytes
        PNG image data.
    """
    # Build a grid of complex numbers.  The real axis spans ``x ± 1.5/zoom``
    # and the imaginary axis spans ``y ± 1.0/zoom`` – these ranges are
    # chosen to give a roughly square view of the classic Mandelbrot set.
    re = np.linspace(x - 1.5 / zoom, x + 1.5 / zoom, width)
    im = np.linspace(y - 1.0 / zoom, y + 1.0 / zoom, height)
    c = re[np.newaxis, :] + 1j * im[:, np.newaxis]

    escape = _mandelbrot_escape_time(c, max_iter)
    return _create_image(escape)


def generate_julia(
    width: int,
    height: int,
    x: float = 0.0,
    y: float = 0.0,
    zoom: float = 1.0,
    max_iter: int = 200,
    c: complex = 0.355 + 0.355j,
) -> bytes:
    """Generate a PNG image of a Julia set.

    Parameters are identical to :func:`generate_mandelbrot` except for the
    additional ``c`` parameter which defines the constant used in the
    iteration ``z = z**2 + c``.
    """
    re = np.linspace(x - 1.5 / zoom, x + 1.5 / zoom, width)
    im = np.linspace(y - 1.0 / zoom, y + 1.0 / zoom, height)
    z = re[np.newaxis, :] + 1j * im[:, np.newaxis]

    escape = _mandelbrot_escape_time(z, max_iter)
    return _create_image(escape)


def generate_fractal(
    mode: str,
    width: int,
    height: int,
    x: float = 0.0,
    y: float = 0.0,
    zoom: float = 1.0,
    max_iter: int = 200,
    c: complex | None = None,
) -> bytes:
    """Dispatch to the appropriate fractal renderer.

    Parameters
    ----------
    mode : str
        ``"mandelbrot"`` or ``"julia"``.
    c : complex, optional
        Constant used for Julia set rendering.  Ignored for Mandelbrot.
    """
    mode = mode.lower()
    if mode == "julia":
        if c is None:
            c = 0.355 + 0.355j
        return generate_julia(width, height, x, y, zoom, max_iter, c)
    if mode == "mandelbrot":
        return generate_mandelbrot(width, height, x, y, zoom, max_iter)
    raise ValueError(f"Unsupported fractal mode: {mode!r}")

# ---------------------------------------------------------------------------
# Main guard for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate sample images and write them to disk.
    mandelbrot = generate_mandelbrot(800, 600, zoom=1.0, max_iter=200)
    with open("mandelbrot.png", "wb") as f:
        f.write(mandelbrot)
    print("Sample Mandelbrot image written to mandelbrot.png")

    julia = generate_julia(800, 600, zoom=1.0, max_iter=200, c=0.355 + 0.355j)
    with open("julia.png", "wb") as f:
        f.write(julia)
    print("Sample Julia image written to julia.png")
