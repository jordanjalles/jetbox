import pytest
from http_wrapper.wrapper import HttpWrapper
import requests
from unittest import mock

# Helper to simulate requests.request
@mock.patch('http_wrapper.wrapper.requests.request')
def test_success_once(mock_req):
    mock_req.return_value = mock.Mock(status_code=200)
    wrapper = HttpWrapper(retry=3, timeout=1)
    resp = wrapper.request('GET', 'http://example.com')
    assert resp.status_code == 200
    mock_req.assert_called_once()

@mock.patch('http_wrapper.wrapper.requests.request')
def test_retry_on_exception(mock_req):
    # First two attempts raise, third succeeds
    mock_req.side_effect = [requests.RequestException("fail"),
                            requests.RequestException("fail"),
                            mock.Mock(status_code=200)]
    wrapper = HttpWrapper(retry=3, timeout=1)
    resp = wrapper.request('GET', 'http://example.com')
    assert resp.status_code == 200
    assert mock_req.call_count == 3

@mock.patch('http_wrapper.wrapper.requests.request')
def test_retry_exhausted(mock_req):
    mock_req.side_effect = requests.RequestException("fail")
    wrapper = HttpWrapper(retry=2, timeout=1)
    with pytest.raises(requests.RequestException):
        wrapper.request('GET', 'http://example.com')
    assert mock_req.call_count == 2
