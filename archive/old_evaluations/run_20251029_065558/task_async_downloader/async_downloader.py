"""Async downloader module.

This module provides two public async functions:

* ``download_file(url, filename)`` – download a single file from ``url`` and
  store it in ``filename``.
* ``download_multiple(urls)`` – concurrently download a list of URLs.

Both functions use :mod:`aiohttp` for asynchronous HTTP requests and provide
basic progress output to the console. Errors are caught and logged but do not
crash the program – the functions return a dictionary mapping each URL to a
boolean indicating success.

The implementation is intentionally lightweight and does not depend on any
external progress‑bar libraries.  It can be used as a drop‑in helper in
scripts or larger projects.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Dict, Iterable, List

import aiohttp

__all__ = ["download_file", "download_multiple"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

async def _download_one(session: aiohttp.ClientSession, url: str, dest: Path) -> bool:
    """Download a single URL to *dest*.

    Parameters
    ----------
    session:
        An active :class:`aiohttp.ClientSession`.
    url:
        The URL to download.
    dest:
        Path where the file should be written.

    Returns
    -------
    bool
        ``True`` if the download succeeded, ``False`` otherwise.
    """
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 32):
                    f.write(chunk)
            print(f"✅  Downloaded {url} → {dest}")
            return True
    except Exception as exc:  # pragma: no cover – error handling path
        print(f"❌  Failed to download {url}: {exc}")
        return False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def download_file(url: str, filename: str | os.PathLike[str]) -> bool:
    """Download *url* to *filename*.

    This is a thin wrapper around :func:`_download_one` that creates a single
    :class:`aiohttp.ClientSession`.
    """
    async with aiohttp.ClientSession() as session:
        return await _download_one(session, url, Path(filename))

async def download_multiple(urls: Iterable[str]) -> Dict[str, bool]:
    """Download all *urls* concurrently.

    Parameters
    ----------
    urls:
        Iterable of URLs to download.

    Returns
    -------
    dict
        Mapping from URL to a boolean indicating success.
    """
    results: Dict[str, bool] = {}
    async with aiohttp.ClientSession() as session:
        tasks: List[asyncio.Task] = []
        for url in urls:
            # Derive a filename from the URL – simple basename extraction.
            dest = Path(url.split("?")[0].split("#")[0].split("/")[-1])
            if not dest:
                dest = Path("downloaded_file")
            task = asyncio.create_task(_download_one(session, url, dest))
            tasks.append(task)
        # Gather results preserving order
        for url, task in zip(urls, tasks):
            results[url] = await task
    return results

# ---------------------------------------------------------------------------
# CLI helper – optional
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover – manual testing helper
    import sys

    if len(sys.argv) < 2:
        print("Usage: python async_downloader.py <url1> [<url2> ...]", file=sys.stderr)
        sys.exit(1)

    urls = sys.argv[1:]
    asyncio.run(download_multiple(urls))

"""
End of module.
"""
