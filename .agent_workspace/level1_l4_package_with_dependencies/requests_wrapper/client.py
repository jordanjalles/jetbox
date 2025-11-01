"""Client module implementing retry logic.

The :func:`request` function is a thin wrapper around
:func:`requests.request`. It accepts the same arguments and returns the
:class:`requests.Response` object.

Retry behaviour:
* Retries on :class:`requests.exceptions.RequestException` (network errors).
* Retries on 5xx status codes.
* Uses exponential backoff with a small jitter.
* Maximum number of attempts is configurable via ``max_retries``.

The implementation is intentionally simple and suitable for unit tests.
"""

import time
import random
from typing import Any, Dict, Optional

import requests
from requests.exceptions import RequestException

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5


def _should_retry(status_code: int) -> bool:
    """Return True if the status code should trigger a retry.

    We retry on any 5xx status code.
    """
    return 500 <= status_code < 600


def request(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    json: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    **kwargs: Any,
) -> requests.Response:
    """Send an HTTP request with retry logic.

    Parameters
    ----------
    method: str
        HTTP method (GET, POST, etc.).
    url: str
        Target URL.
    params, data, json, headers, timeout, **kwargs
        Forwarded to :func:`requests.request`.
    max_retries: int, default 3
        Maximum number of attempts (including the first try).
    backoff_factor: float, default 0.5
        Base backoff multiplier. The actual sleep time is
        ``backoff_factor * (2 ** attempt)`` plus a small jitter.

    Returns
    -------
    requests.Response
        The successful response.

    Raises
    ------
    RequestException
        If all retry attempts fail.
    """

    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.request(
                method,
                url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                timeout=timeout,
                **kwargs,
            )
            if not _should_retry(response.status_code):
                return response
            # 5xx status code: treat as retryable
        except RequestException:
            # Network error: treat as retryable
            pass
        # If we reach here, we need to retry
        attempt += 1
        if attempt == max_retries:
            raise RequestException(
                f"Request failed after {max_retries} attempts", response=None
            )
        sleep_time = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
        time.sleep(sleep_time)

    # Should not reach here
    raise RequestException("Unexpected error in retry logic")

# End of client.py
