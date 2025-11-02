"""Client module implementing HTTP methods with retry logic.

The module exposes simple functions: get, post, put, delete.
Each function accepts a URL and optional kwargs that are forwarded to
`requests.request`.  Retries are performed using a simple exponential
backoff strategy.  The default configuration retries 3 times with a
base delay of 0.1 seconds.

The implementation is intentionally lightweight and uses only the
standard `requests` library.
"""

import time
from typing import Any, Dict, Optional

import requests

# Default retry configuration
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.1  # seconds


def _request_with_retry(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Internal helper that performs a request with retry logic.

    Parameters
    ----------
    method: str
        HTTP method name (e.g., 'GET', 'POST').
    url: str
        Target URL.
    **kwargs:
        Arguments forwarded to ``requests.request``.

    Returns
    -------
    requests.Response
        The successful response object.

    Raises
    ------
    requests.RequestException
        If all retry attempts fail or a non‑retryable error occurs.
    """
    retries = kwargs.pop("retries", DEFAULT_RETRIES)
    backoff = kwargs.pop("backoff", DEFAULT_BACKOFF)

    if not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a non‑negative integer")
    if not isinstance(backoff, (int, float)) or backoff < 0:
        raise ValueError("backoff must be a non‑negative number")

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.request(method, url, **kwargs)
            # Retry on server errors (5xx)
            if 500 <= response.status_code < 600:
                raise requests.HTTPError(f"Server error: {response.status_code}")
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(backoff * (2 ** (attempt - 1)))
    # All attempts failed – re‑raise the last exception
    raise last_exc  # type: ignore[return-value]


def get(url: str, **kwargs: Any) -> requests.Response:
    """Perform a GET request with retry logic."""
    return _request_with_retry("GET", url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    """Perform a POST request with retry logic."""
    return _request_with_retry("POST", url, **kwargs)


def put(url: str, **kwargs: Any) -> requests.Response:
    """Perform a PUT request with retry logic."""
    return _request_with_retry("PUT", url, **kwargs)


def delete(url: str, **kwargs: Any) -> requests.Response:
    """Perform a DELETE request with retry logic."""
    return _request_with_retry("DELETE", url, **kwargs)

__all__ = ["get", "post", "put", "delete"]
