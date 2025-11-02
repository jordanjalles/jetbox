import requests
import time
from typing import Any, Dict, Optional

class HttpWrapper:
    """Simple HTTP wrapper with retry and timeout logic.

    Parameters
    ----------
    retry: int
        Number of attempts to retry on failure. Default is 3.
    timeout: int | float
        Timeout in seconds for each request. Default is 5.
    """

    def __init__(self, retry: int = 3, timeout: int | float = 5):
        self.retry = retry
        self.timeout = timeout

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Internal helper that performs a single request.

        Raises
        ------
        requests.RequestException
            Propagated from :func:`requests.request`.
        """
        return requests.request(method, url, timeout=self.timeout, **kwargs)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Perform an HTTP request with retry logic.

        Parameters
        ----------
        method: str
            HTTP method (e.g., 'GET', 'POST').
        url: str
            Target URL.
        **kwargs
            Additional keyword arguments forwarded to :func:`requests.request`.

        Returns
        -------
        requests.Response
            The successful response.

        Raises
        ------
        requests.RequestException
            If all retry attempts fail.
        """
        attempts = 0
        while attempts < self.retry:
            try:
                return self._request(method, url, **kwargs)
            except requests.RequestException as exc:
                attempts += 1
                if attempts >= self.retry:
                    raise
                # Exponential back‑off: wait 0.5 * 2**attempt seconds
                time.sleep(0.5 * (2 ** attempts))
        # Should never reach here
        raise RuntimeError("Unreachable code in HttpWrapper.request")
