"""Test suite for the http_wrapper module.

The tests use the public httpbin.org service to verify that the wrapper
performs GET requests correctly and that the retry logic works.
"""

import time
from unittest import mock

import pytest
import requests

from http_wrapper import get

# Helper to simulate a flaky endpoint that fails the first N times.
class FlakyResponse:
    def __init__(self, fail_times, status_code=200):
        self.fail_times = fail_times
        self.status_code = status_code
        self.attempt = 0

    def __call__(self, *args, **kwargs):
        self.attempt += 1
        if self.attempt <= self.fail_times:
            # Raise a RequestException to be caught by the wrapper
            raise requests.exceptions.RequestException("Simulated network failure")
        return mock.Mock(status_code=self.status_code)


def test_get_success(monkeypatch):
    """Verify that a simple GET request succeeds."""
    url = "https://httpbin.org/get"
    response = get(url, timeout=2)
    assert response.status_code == 200


def test_get_retry_success(monkeypatch):
    """Verify that the wrapper retries on failure and eventually succeeds."""
    flaky = FlakyResponse(fail_times=2)
    monkeypatch.setattr("requests.get", flaky)
    start = time.time()
    response = get("https://example.com", retries=3, timeout=1)
    elapsed = time.time() - start
    # Should have succeeded after 3 attempts (2 failures + 1 success)
    assert flaky.attempt == 3
    assert response.status_code == 200
    # Ensure some back‑off time was spent (backoff_factor=0.5 by default)
    assert elapsed >= 0.5 + 1.0


def test_get_retry_exhausted(monkeypatch):
    """Verify that an exception is raised when all retries fail."""
    flaky = FlakyResponse(fail_times=5)
    monkeypatch.setattr("requests.get", flaky)
    with pytest.raises(requests.exceptions.RequestException):
        get("https://example.com", retries=3, timeout=1)
    assert flaky.attempt == 4  # 3 retries + 1 initial attempt
