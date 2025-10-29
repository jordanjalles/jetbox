"""Async downloader module.

Provides two public async functions:

* ``download_file(url, path)`` – download a single URL and write the
  response body to ``path``.
* ``download_multiple(urls)`` – concurrently download a list of URLs.

The implementation uses :mod:`aiohttp` for HTTP requests and
:mod:`asyncio` for concurrency.
"""

import asyncio
from pathlib import Path
from typing import Iterable, List

import aiohttp

__all__ = ["download_file", "download_multiple"]


async def download_file(url: str, path: str | Path) -> None:
    """Download *url* and write the response body to *path*.

    Parameters
    ----------
    url:
        The URL to download.
    path:
        Destination file path.  Parent directories are created if
        necessary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
            path.write_bytes(data)


async def download_multiple(urls: Iterable[str]) -> List[Path]:
    """Download all *urls* concurrently.

    Returns a list of :class:`~pathlib.Path` objects pointing to the
    downloaded files.  Filenames are derived from the URL path; if the
    URL ends with a slash a ``index.html`` file is used.
    """
    tasks = []
    for url in urls:
        # Derive a safe filename from the URL
        name = url.split("?")[0].split("#")[0].rstrip("/")
        if not name:
            name = "index.html"
        else:
            name = name.split("/")[-1]
        path = Path(name)
        tasks.append(download_file(url, path))

    await asyncio.gather(*tasks)
    return [Path(name) for name in [url.split("?")[0].split("#")[0].rstrip("/") or "index.html" for url in urls]]
