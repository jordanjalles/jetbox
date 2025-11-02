import json
import time

import pytest
import responses

from requests_wrapper.client import get, post, put, delete

@responses.activate
def test_get_success():
    url = "https://example.com/success"
    responses.add(responses.GET, url, json={"ok": True}, status=200)
    resp = get(url)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

@responses.activate
def test_get_retry_success():
    url = "https://example.com/retry"
    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, json={"ok": True}, status=200)
    start = time.time()
    resp = get(url, retries=3, backoff=0.01)
    elapsed = time.time() - start
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert elapsed >= 0.03

@responses.activate
def test_post_retry_failure():
    url = "https://example.com/fail"
    responses.add(responses.POST, url, status=500)
    responses.add(responses.POST, url, status=500)
    responses.add(responses.POST, url, status=500)
    with pytest.raises(Exception):
        post(url, retries=3, backoff=0.01)

@responses.activate
def test_put_and_delete():
    url = "https://example.com/resource"
    responses.add(responses.PUT, url, status=204)
    responses.add(responses.DELETE, url, status=204)
    resp_put = put(url)
    assert resp_put.status_code == 204
    resp_del = delete(url)
    assert resp_del.status_code == 204

@responses.activate
def test_custom_headers():
    url = "https://example.com/headers"
    def request_callback(request):
        assert request.headers["X-Custom"] == "value"
        return (200, {}, json.dumps({"ok": True}))
    responses.add_callback(responses.GET, url, callback=request_callback)
    resp = get(url, headers={"X-Custom": "value"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

@responses.activate
def test_retry_parameter_exhausted():
    url = "https://example.com/retry_param"
    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, json={"ok": True}, status=200)
    with pytest.raises(Exception):
        get(url, retries=2, backoff=0.01)

@responses.activate
def test_default_retry_config():
    url = "https://example.com/default_retry"
    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, status=500)
    responses.add(responses.GET, url, json={"ok": True}, status=200)
    resp = get(url)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

@responses.activate
def test_non_200_returned():
    url = "https://example.com/non200"
    responses.add(responses.GET, url, status=404)
    responses.add(responses.GET, url, status=404)
    responses.add(responses.GET, url, status=404)
    resp = get(url, retries=3, backoff=0.01)
    assert resp.status_code == 404

@responses.activate
def test_import():
    import importlib
    mod = importlib.import_module("requests_wrapper.client")
    assert hasattr(mod, "get")
    assert hasattr(mod, "post")
    assert hasattr(mod, "put")
    assert hasattr(mod, "delete")
