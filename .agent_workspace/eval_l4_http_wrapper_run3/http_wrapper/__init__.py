"""HTTP wrapper module with retry logic and timeout support.

This module provides a simple `get` function that performs an HTTP GET request
with configurable retry attempts and timeout. It uses the `requests` library
under the hood.

Example usage:

>>> from http_wrapper import get
>>> response = get("https://httpbin.org/get", retries=2, timeout=3)
>>> print(response.status_code)
200

The function returns a `requests.Response` object.
"""

from .wrapper import get

__all__ = ["get"]
