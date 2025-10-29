"""API client module.

This module provides a simple JSONClient class that wraps the
`requests` library for making HTTP GET and POST requests.  The
client automatically serialises data to JSON for POST requests and
parses JSON responses.  Basic network error handling is included.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests
from requests import Response


class JSONClient:
    """Simple HTTP client for JSON APIs.

    The client uses the :mod:`requests` library under the hood.  It
    provides two convenience methods:

    * :meth:`get` – perform a GET request and return the parsed JSON
      response.
    * :meth:`post` – perform a POST request with a JSON body and return
      the parsed JSON response.

    Both methods return the parsed JSON data (a ``dict`` or ``list``)
    or ``None`` if the response body is empty.

    Network errors (connection errors, timeouts, etc.) are caught and
    re‑raised as :class:`JSONClientError` with a helpful message.
    """

    def __init__(self, timeout: float = 10.0, headers: Optional[Dict[str, str]] = None):
        """Create a new client.

        Parameters
        ----------
        timeout:
            Default timeout for requests in seconds.
        headers:
            Optional default headers to send with every request.
        """
        self.timeout = timeout
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a GET request.

        Parameters
        ----------
        url:
            Target URL.
        params:
            Optional query parameters.

        Returns
        -------
        Any
            Parsed JSON response or ``None``.
        """
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            return self.parse_response(response)
        except requests.RequestException as exc:
            raise JSONClientError(f"GET request failed: {exc}") from exc

    def post(self, url: str, data: Any) -> Any:
        """Send a POST request with JSON payload.

        Parameters
        ----------
        url:
            Target URL.
        data:
            Data to serialise to JSON.

        Returns
        -------
        Any
            Parsed JSON response or ``None``.
        """
        try:
            json_data = json.dumps(data)
            headers = {"Content-Type": "application/json"}
            response = self.session.post(url, data=json_data, headers=headers, timeout=self.timeout)
            return self.parse_response(response)
        except requests.RequestException as exc:
            raise JSONClientError(f"POST request failed: {exc}") from exc

    @staticmethod
    def parse_response(response: Response) -> Any:
        """Parse a :class:`requests.Response` as JSON.

        If the response body is empty, ``None`` is returned.
        """
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise JSONClientError("Response is not valid JSON") from exc


class JSONClientError(RuntimeError):
    """Custom exception for JSONClient errors."""

    pass


# If this module is run directly, demonstrate a simple GET request.
if __name__ == "__main__":
    client = JSONClient()
    try:
        result = client.get("https://httpbin.org/get", params={"foo": "bar"})
        print("GET result:", result)
    except JSONClientError as e:
        print("Error:", e)
