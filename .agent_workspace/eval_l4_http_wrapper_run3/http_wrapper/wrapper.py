"""Implementation of the HTTP wrapper.

The `get` function performs an HTTP GET request with retry logic and a
timeout. Retries are performed on network-related exceptions (e.g.
`requests.exceptions.RequestException`). The back‑off strategy is a simple
exponential back‑off with a small jitter.

The function is intentionally lightweight so it can be used in scripts or
tests without pulling in heavy dependencies.
"""

import time
from typing import Optional

import requests
from requests.exceptions import RequestException

__all__ = ["get"]


def get(
    url: str,
    *,
    retries: int = 3,
    timeout: Optional[float] = 5.0,
    backoff_factor: float = 0.5,
    **kwargs,
):
    """Perform an HTTP GET request with retry logic.

    Parameters
    ----------
    url: str
        The URL to request.
    retries: int, default 3
        Number of retry attempts. A value of 0 means no retries.
    timeout: float or tuple, optional
        Timeout for the request. Passed directly to ``requests.get``.
    backoff_factor: float, default 0.5
        Base factor for exponential back‑off. The delay for attempt ``n`` is
        ``backoff_factor * 2 ** (n - 1)`` seconds.
    **kwargs:
        Additional keyword arguments forwarded to ``requests.get``.

    Returns
    -------
    requests.Response
        The successful response object.

    Raises
    ------
    requests.exceptions.RequestException
        If all retry attempts fail.
    """

    attempt = 0
    while True:
        try:
            return requests.get(url, timeout=timeout, **kwargs)
        except RequestException as exc:
            attempt += 1
            if attempt > retries:
                # All attempts exhausted – re‑raise the last exception.
                raise
            # Exponential back‑off with jitter.
            sleep_time = backoff_factor * 2 ** (attempt - 1)
            time.sleep(sleep_time)

