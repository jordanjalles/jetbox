"""Unit tests for the requests_wrapper package.

The tests use :mod:`responses` to mock HTTP responses without making real
network calls. They verify that:

* Successful responses are returned immediately.
* 5xx responses trigger retries up to ``max_retries``.
* Network errors (``RequestException``) trigger retries.
* After exhausting retries, an exception is raised.
"""

import pytest
import responses
from requests.exceptions import RequestException

from requests_wrapper import request

# Helper to count calls
class CallCounter:
    def __init__(self):
        self.count = 0
    def __call__(self, *args, **kwargs):
        self.count += 1


@responses.activate
def test_success_no_retry():
    responses.add(responses.GET, "https://example.com", status=200, json={"ok": True})
    resp = request("GET", "https://example.com")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@responses.activate
def test_retry_on_5xx_then_success():
    counter = CallCounter()
    # First two responses are 500, third is 200
    responses.add(responses.GET, "https://example.com", status=500, body="error", match=[responses.matchers.header_matcher({"X-Call": "1"})])
    responses.add(responses.GET, "https://example.com", status=500, body="error", match=[responses.matchers.header_matcher({"X-Call": "2"})])
    responses.add(responses.GET, "https://example.com", status=200, json={"ok": True}, match=[responses.matchers.header_matcher({"X-Call": "3"})])

    # Monkeypatch the request to add a header indicating call number
    original_request = request
    def wrapped(method, url, **kwargs):
        kwargs.setdefault("headers", {})["X-Call"] = str(counter.count + 1)
        return original_request(method, url, **kwargs)

    # Use wrapped function
    resp = wrapped("GET", "https://example.com")
    assert counter.count == 3
    assert resp.status_code == 200


@responses.activate
def test_retry_on_network_error_then_success():
    counter = CallCounter()
    # First two attempts raise RequestException, third succeeds
    def request_callback(request):
        if counter.count < 2:
            counter.count += 1
            raise RequestException("network error")
        return (200, {}, "ok")
    responses.add_callback(responses.GET, "https://example.com", callback=request_callback)
    resp = request("GET", "https://example.com")
    assert counter.count == 2
    assert resp.status_code == 200


@responses.activate
def test_exhaust_retries_raises():
    counter = CallCounter()
    def request_callback(request):
        counter.count += 1
        raise RequestException("network error")
    responses.add_callback(responses.GET, "https://example.com", callback=request_callback)
    with pytest.raises(RequestException):
        request("GET", "https://example.com", max_retries=2)
    assert counter.count == 2


# End of tests
