"""Procedural noise-driven landscape generator.

This module generates a 2‑D height map using a simple Perlin‑like noise
algorithm and renders it as a PNG image.  The implementation is
self‑contained and only depends on the standard library, NumPy and Pillow.

The main public function is :func:`generate_landscape` which returns a
``PIL.Image`` instance.  The module can also be executed as a script to
create a sample image.

Example
-------
>>> from procedural_art.landscape import generate_landscape
>>> img = generate_landscape(512, 512)
>>> img.save("landscape.png")
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _fade(t: float) -> float:
    """Perlin fade function.

    Smooths the interpolation curve.
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def _grad(hash_val: int, x: float, y: float) -> float:
    """Gradient function used in 2‑D Perlin noise.

    The hash value is used to pick one of 8 possible gradient directions.
    """
    h = hash_val & 7
    u = x if h < 4 else y
    v = y if h < 4 else x
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


# ---------------------------------------------------------------------------
# Core noise implementation
# ---------------------------------------------------------------------------

def perlin_noise(x: float, y: float, perm: np.ndarray) -> float:
    """Return a Perlin noise value for coordinates (x, y).

    Parameters
    ----------
    x, y : float
        Coordinates in noise space.
    perm : np.ndarray
        Permutation table used for hashing.
    """
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255

    xf = x - math.floor(x)
    yf = y - math.floor(y)

    u = _fade(xf)
    v = _fade(yf)

    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1, yf), u)
    x2 = _lerp(_grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1), u)

    return _lerp(x1, x2, v)


# ---------------------------------------------------------------------------
# Height map generation
# ---------------------------------------------------------------------------
@dataclass
class NoiseParams:
    scale: float = 100.0
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    seed: int | None = None


def generate_height_map(
    width: int,
    height: int,
    params: NoiseParams,
) -> np.ndarray:
    """Generate a 2‑D height map using fractal noise.

    The output is a NumPy array of shape (height, width) with values in
    the range [0, 1].
    """
    if params.seed is None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = params.seed
    rng = np.random.default_rng(seed)
    perm = np.arange(256, dtype=int)
    rng.shuffle(perm)
    perm = np.stack([perm, perm]).flatten()

    lin_x = np.linspace(0, width / params.scale, width, endpoint=False)
    lin_y = np.linspace(0, height / params.scale, height, endpoint=False)
    grid_x, grid_y = np.meshgrid(lin_x, lin_y)

    noise = np.zeros((height, width), dtype=float)
    amplitude = 1.0
    frequency = 1.0
    max_amp = 0.0

    for _ in range(params.octaves):
        noise += amplitude * np.vectorize(perlin_noise)(grid_x * frequency, grid_y * frequency, perm)
        max_amp += amplitude
        amplitude *= params.persistence
        frequency *= params.lacunarity

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    return noise


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def height_to_color(height: float) -> Tuple[int, int, int]:
    """Map a normalized height value to an RGB color.

    The mapping is a simple gradient from deep blue (water) to white
    (snow).  The function is intentionally lightweight.
    """
    if height < 0.3:
        # Water
        return (0, 0, int(128 + 127 * height / 0.3))
    elif height < 0.4:
        # Shore
        return (0, int(128 + 127 * (height - 0.3) / 0.1), 0)
    elif height < 0.6:
        # Grass
        return (int(0 + 255 * (height - 0.4) / 0.2), 255, 0)
    elif height < 0.8:
        # Mountain
        return (int(255 * (height - 0.6) / 0.2), 255, 0)
    else:
        # Snow
        val = int(255 * (height - 0.8) / 0.2)
        return (val, val, val)


def render_height_map(height_map: np.ndarray) -> Image.Image:
    """Render a height map as a Pillow image.

    The image is RGB and uses a simple terrain color palette.
    """
    h, w = height_map.shape
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            pixels[x, y] = height_to_color(height_map[y, x])
    return img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_landscape(
    width: int = 512,
    height: int = 512,
    params: NoiseParams | None = None,
) -> Image.Image:
    """Generate a procedural landscape image.

    Parameters
    ----------
    width, height : int
        Dimensions of the output image.
    params : NoiseParams, optional
        Noise parameters.  If omitted, defaults are used.
    """
    if params is None:
        params = NoiseParams()
    height_map = generate_height_map(width, height, params)
    return render_height_map(height_map)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a procedural landscape image.")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--scale", type=float, default=100.0, help="Noise scale")
    parser.add_argument("--octaves", type=int, default=6, help="Number of octaves")
    parser.add_argument("--persistence", type=float, default=0.5, help="Persistence")
    parser.add_argument("--lacunarity", type=float, default=2.0, help="Lacunarity")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="landscape.png", help="Output file")

    args = parser.parse_args()

    params = NoiseParams(
        scale=args.scale,
        octaves=args.octaves,
        persistence=args.persistence,
        lacunarity=args.lacunarity,
        seed=args.seed,
    )
    img = generate_landscape(args.width, args.height, params)
    img.save(args.output)
    print(f"Saved landscape to {args.output}")
