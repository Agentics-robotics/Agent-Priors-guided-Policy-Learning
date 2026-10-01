import json

import pytest

from appl.api import ResponsesClient, client


def test_configuration_and_missing_credentials_are_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    transport = client({}, tmp_path)
    assert transport.configuration()["base_url"] == "https://api.openai.com/v1"
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        transport.respond({"model": transport.model, "reasoning": {"effort": "xhigh"}})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("url", ["https://name:secret@example.org/v1", "https://example.org/v1?token=secret", "http://example.org/v1"])
def test_unsafe_provider_configuration_is_rejected(tmp_path, url):
    with pytest.raises(ValueError):
        ResponsesClient(tmp_path, base_url=url)


def test_transient_retry_preserves_request_and_redacts_credentials(tmp_path, monkeypatch):
    import appl.api as api
    credential = "unit-test-credential-never-publish"
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    statuses = iter([503, 200])
    requests = []

    class Connection:
        def __init__(self, *args, **kwargs):
            self.status = next(statuses)

        def request(self, method, endpoint, body, headers):
            requests.append((method, endpoint, body))
            assert headers["Authorization"] == "Bearer " + credential

        def getresponse(self):
            return self

        def read(self):
            return json.dumps({"model": "gpt-5.6-sol", "status": "completed", "output": [],
                               "echo": credential}).encode()

        def close(self):
            pass

    monkeypatch.setattr(api, "HTTPSConnection", Connection)
    transport = ResponsesClient(tmp_path, retry_delays=(0,))
    status, body = transport.respond({"model": transport.model, "reasoning": {"effort": "xhigh"}})
    assert status == 200 and body["echo"] == "[REDACTED]"
    assert len(requests) == 2 and requests[0] == requests[1]
    assert requests[0][1] == "/v1/responses"
    assert all(credential not in path.read_text() for path in tmp_path.rglob("*.json"))


def test_quota_failure_is_not_retried(tmp_path, monkeypatch):
    import appl.api as api
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-credential")
    calls = []

    class Connection:
        status = 429

        def __init__(self, *args, **kwargs):
            calls.append(1)

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return self

        def read(self):
            return b'{"error":{"code":"insufficient_quota"}}'

        def close(self):
            pass

    monkeypatch.setattr(api, "HTTPSConnection", Connection)
    transport = ResponsesClient(tmp_path, retry_delays=(0,))
    status, _ = transport.respond({"model": transport.model, "reasoning": {"effort": "xhigh"}})
    assert status == 429 and len(calls) == 1
