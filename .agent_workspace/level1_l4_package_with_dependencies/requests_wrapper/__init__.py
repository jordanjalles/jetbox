"""requests_wrapper package providing retry logic for HTTP requests.

This package exposes a single function `request` that wraps `requests.request` with
retry logic. It retries on network errors and 5xx HTTP status codes.
"""

from .client import request

__all__ = ["request"]
