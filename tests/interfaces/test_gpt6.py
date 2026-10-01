"""Responses contract tests use an in-memory transport and never call the API."""
from copy import deepcopy
from dataclasses import asdict
import json
from unittest.mock import Mock

import pytest

from experiment_interfaces import gpt6
from experiment_interfaces.gpt6 import GenerateRequest, GPT6Client


def completed_response():
    return {
        "object": "response", "id": "resp_example", "model": "gpt-6-astra-2026-09-01",
        "status": "completed", "error": None, "incomplete_details": None,
        "reasoning": {"effort": "max"},
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                  "output_tokens_details": {"reasoning_tokens": 40}},
        "output": [
            {"type": "reasoning", "id": "rs_example", "summary": []},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "First "},
                         {"type": "output_text", "text": "second."}]},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "\nThird."}]},
        ],
    }


def transport(monkeypatch, payload, status=200, *, scheme="https"):
    response = Mock(status=status)
    response.read.return_value = json.dumps(payload).encode("utf-8")
    connection = Mock()
    connection.getresponse.return_value = response
    factory = Mock(return_value=connection)
    monkeypatch.setattr(gpt6, "HTTPSConnection" if scheme == "https" else "HTTPConnection", factory)
    return factory, connection


def test_generates_exact_explicit_multimodal_request_and_preserves_receipt(monkeypatch):
    payload = completed_response()
    factory, connection = transport(monkeypatch, payload)
    sink = Mock()
    client = GPT6Client(api_key="test-secret", response_sink=sink)
    content = [{"role": "user", "content": [
        {"type": "input_text", "text": "Supplied training evidence only."},
        {"type": "input_image", "image_url": "data:image/png;base64,c3VwcGxpZWQ=", "detail": "high"},
    ]}]
    request = GenerateRequest(content, "Keep the given extraction instructions exactly.")
    result = client.generate(request)

    factory.assert_called_once_with("api.openai.com", timeout=120)
    connection.request.assert_called_once()
    args, kwargs = connection.request.call_args
    assert args == ("POST", "/v1/responses")
    assert json.loads(kwargs["body"]) == {
        "model": "gpt-6-astra", "input": content, "instructions": request.instructions,
        "reasoning": {"effort": "max"}, "max_output_tokens": 4096, "store": False,
    }
    assert kwargs["headers"]["Authorization"] == "Bearer test-secret"
    connection.getresponse.assert_called_once()
    connection.close.assert_called_once()
    sink.assert_called_once_with(result.request, payload, 200)
    assert result.output_text == "First second.\nThird."
    assert result.response == payload
    assert result.response_id == "resp_example"
    assert result.model == payload["model"]
    assert result.usage == payload["usage"]
    assert result.status == "completed"
    assert result.http_status == 200
    assert "test-secret" not in repr(client)
    assert "test-secret" not in json.dumps(asdict(result))
    assert "Authorization" not in json.dumps(sink.call_args.args)


def test_request_body_is_independent_and_does_not_resolve_local_paths():
    supplied = [{"role": "user", "content": [{"type": "input_text", "text": "/missing/evidence.npz"}]}]
    request = GenerateRequest(supplied, "", model="explicit-model-id", reasoning_effort="max",
                              max_output_tokens=8192, timeout_seconds=30)
    body = request.to_body()
    assert body["input"] == supplied
    assert body["model"] == "explicit-model-id"
    assert "timeout_seconds" not in body
    body["input"][0]["content"][0]["text"] = "changed"
    assert supplied[0]["content"][0]["text"] == "/missing/evidence.npz"


def test_uses_only_process_environment_when_key_is_not_explicit(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    _, connection = transport(monkeypatch, completed_response())
    result = GPT6Client().generate(GenerateRequest("input", "instructions"))
    assert connection.request.call_args.kwargs["headers"]["Authorization"] == "Bearer environment-secret"
    assert "environment-secret" not in repr(result)
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(KeyError, match="OPENAI_API_KEY"):
        GPT6Client()


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_failure_is_one_request_and_preserves_raw_receipt(monkeypatch, status):
    payload = {"error": {"type": "request_error", "message": "request rejected"}}
    _, connection = transport(monkeypatch, payload, status)
    sink = Mock()
    request = GenerateRequest("input", "instructions")
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        GPT6Client("test-secret", sink).generate(request)
    connection.request.assert_called_once()
    connection.getresponse.assert_called_once()
    connection.close.assert_called_once()
    sink.assert_called_once_with(request.to_body(), payload, status)


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), OSError("connection failed")])
def test_transport_failure_propagates_once_without_retry(monkeypatch, failure):
    _, connection = transport(monkeypatch, completed_response())
    connection.getresponse.side_effect = failure
    sink = Mock()
    with pytest.raises(type(failure), match=str(failure)):
        GPT6Client("test-secret", sink).generate(GenerateRequest("input", "instructions"))
    connection.request.assert_called_once()
    connection.getresponse.assert_called_once()
    connection.close.assert_called_once()
    sink.assert_not_called()


@pytest.mark.parametrize("mutation,match", [
    ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "did not complete"),
    ({"status": "failed"}, "did not complete"),
    ({"error": {"code": "server_error"}}, "error or incomplete"),
    ({"object": "other"}, "response object"),
    ({"id": None}, "missing id"),
    ({"model": None}, "missing model"),
    ({"usage": None}, "token usage"),
    ({"usage": {"input_tokens": -1}}, "usage field"),
    ({"output": {}}, "must be a list"),
    ({"output": [None]}, "output item"),
    ({"output": []}, "no output text"),
    ({"output": [{"type": "message", "role": "assistant", "status": "incomplete", "content": []}]}, "not a completed"),
    ({"output": [{"type": "message", "role": "assistant", "status": "completed", "content": [
        {"type": "refusal", "refusal": "Cannot provide that."}]}]}, "refused"),
    ({"output": [{"type": "message", "role": "assistant", "status": "completed", "content": [
        {"type": "output_text", "text": 42}]}]}, "output_text block"),
])
def test_rejects_noncomplete_or_malformed_success_after_receipt(monkeypatch, mutation, match):
    payload = completed_response() | deepcopy(mutation)
    _, connection = transport(monkeypatch, payload)
    sink = Mock()
    request = GenerateRequest("input", "instructions")
    with pytest.raises(ValueError, match=match):
        GPT6Client("test-secret", sink).generate(request)
    connection.request.assert_called_once()
    sink.assert_called_once_with(request.to_body(), payload, 200)


def test_receipt_sink_cannot_change_validated_response(monkeypatch):
    payload = completed_response()
    transport(monkeypatch, payload)

    def sink(body, response, status):
        body.clear()
        response.clear()

    result = GPT6Client("test-secret", sink).generate(GenerateRequest("input", "instructions"))
    assert result.response == payload
    assert result.request["input"] == "input"


def test_invalid_json_propagates_without_retry(monkeypatch):
    _, connection = transport(monkeypatch, completed_response())
    connection.getresponse.return_value.read.return_value = b"not JSON"
    with pytest.raises(json.JSONDecodeError):
        GPT6Client("test-secret").generate(GenerateRequest("input", "instructions"))
    connection.request.assert_called_once()
    connection.close.assert_called_once()


@pytest.mark.parametrize("overrides", [
    {"input": ""}, {"input": []}, {"input": ["not-an-item"]},
    {"instructions": None}, {"model": ""}, {"reasoning_effort": "none"}, {"reasoning_effort": "ultra"},
    {"max_output_tokens": 15}, {"max_output_tokens": True},
    {"timeout_seconds": 0}, {"timeout_seconds": float("inf")}, {"timeout_seconds": True},
])
def test_rejects_invalid_request_contract(overrides):
    with pytest.raises(ValueError):
        GenerateRequest(**({"input": "input", "instructions": "instructions"} | overrides))


@pytest.mark.parametrize("base_url,scheme,host,port,path", [
    ("http://example.test:8123", "http", "example.test", 8123, "/responses"),
    ("http://example.test:8123/", "http", "example.test", 8123, "/responses"),
    ("http://example.test:8123/v1", "http", "example.test", 8123, "/v1/responses"),
    ("http://example.test:8123/provider/v1/", "http", "example.test", 8123, "/provider/v1/responses"),
    ("http://example.test:8123/provider//", "http", "example.test", 8123, "/provider//responses"),
    ("https://provider.example:8443/gateway", "https", "provider.example", 8443, "/gateway/responses"),
    ("https://provider.example/gateway", "https", "provider.example", None, "/gateway/responses"),
    ("http://[2001:db8::1234]:8123/base", "http", "2001:db8::1234", 8123, "/base/responses"),
])
def test_explicit_provider_scheme_port_and_full_path(monkeypatch, base_url, scheme, host, port, path):
    factory, connection = transport(monkeypatch, completed_response(), scheme=scheme)
    other = Mock(side_effect=AssertionError("A different protocol must not be attempted"))
    monkeypatch.setattr(gpt6, "HTTPConnection" if scheme == "https" else "HTTPSConnection", other)
    sink = Mock()
    headers = {"x-example-header": "example-value"}
    client = GPT6Client("provider-secret", sink, base_url=base_url, http_headers=headers)
    headers["x-example-header"] = "caller-mutated"
    result = client.generate(GenerateRequest("Explicit input", "Explicit instructions"))
    options = {"timeout": 120}
    if port is not None:
        options["port"] = port
    factory.assert_called_once_with(host, **options)
    connection.request.assert_called_once()
    args, kwargs = connection.request.call_args
    assert args == ("POST", path)
    assert kwargs["headers"]["x-example-header"] == "example-value"
    assert kwargs["headers"]["Authorization"] == "Bearer provider-secret"
    assert json.loads(kwargs["body"])["reasoning"] == {"effort": "max"}
    other.assert_not_called()
    connection.close.assert_called_once()
    serialized = json.dumps(asdict(result)) + json.dumps(sink.call_args.args) + repr(client)
    assert "provider-secret" not in serialized
    assert "example-value" not in serialized
    assert "x-example-header" not in serialized


def test_explicit_key_environment_has_no_other_credential_source(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.setenv("EXAMPLE_RESPONSES_KEY", "local-provider-secret")
    _, connection = transport(monkeypatch, completed_response(), scheme="http")
    client = GPT6Client(base_url="http://example.test:8123", api_key_env="EXAMPLE_RESPONSES_KEY")
    client.generate(GenerateRequest("input", "instructions"))
    assert connection.request.call_args.kwargs["headers"]["Authorization"] == "Bearer local-provider-secret"
    monkeypatch.delenv("EXAMPLE_RESPONSES_KEY")
    with pytest.raises(KeyError, match="EXAMPLE_RESPONSES_KEY"):
        GPT6Client(base_url="http://example.test:8123", api_key_env="EXAMPLE_RESPONSES_KEY")
    # Explicit credentials do not require the named environment variable.
    explicit = GPT6Client("explicit-secret", api_key_env="EXAMPLE_RESPONSES_KEY")
    assert "explicit-secret" not in repr(explicit)


@pytest.mark.parametrize("base_url", [
    "", "example.test:8123", "//example.test:8123", "ftp://example.test", "https:///v1",
    "http://user:secret@example.test:8123", "http://user@example.test", "http://@example.test",
    "http://example.test:8123?key=secret", "http://example.test?", "http://example.test#", "http://example.test/#fragment",
    " http://example.test", "http://example.test/with space", "http://local\nhost", "http://example.test:\t8123",
    "http://example.test:", "http://example.test:0", "http://example.test:65536", "http://example.test:invalid",
    "http://[2001:db8::1234", "http://example.test/\u00e9", None,
])
def test_invalid_provider_url_fails_before_any_network(monkeypatch, base_url):
    http, https = Mock(), Mock()
    monkeypatch.setattr(gpt6, "HTTPConnection", http)
    monkeypatch.setattr(gpt6, "HTTPSConnection", https)
    with pytest.raises(ValueError):
        GPT6Client("test-secret", base_url=base_url)
    http.assert_not_called()
    https.assert_not_called()


@pytest.mark.parametrize("headers", [
    {"Authorization": "other"}, {"authorization": "other"}, {"AUTHORIZATION": "other"},
    {"Content-Type": "text/plain"}, {"Accept": "text/plain"}, {"HOST": "elsewhere"},
    {"Content-Length": "1"}, {"Transfer-Encoding": "chunked"}, {"Connection": "upgrade"},
    {"Proxy-Authorization": "other"}, {"Upgrade": "websocket"}, {"TE": "trailers"},
    {"Trailer": "secret"}, {"Expect": "100-continue"}, {"Content-Encoding": "gzip"},
    {"X-Token": "one", "x-token": "two"}, {"Bad Header": "value"}, {"": "value"},
    {"X-Test\r\nHost": "value"}, {"X-Test": "value\r\nInjected: header"}, {"X-Test": "tab\tvalue"},
    {"X-Test": "non-Latin1 \u20ac"}, {"X-Test": 1}, {1: "value"}, [],
])
def test_invalid_or_reserved_custom_headers_fail_before_network(monkeypatch, headers):
    factory = Mock()
    monkeypatch.setattr(gpt6, "HTTPSConnection", factory)
    with pytest.raises(ValueError):
        GPT6Client("test-secret", http_headers=headers)
    factory.assert_not_called()


@pytest.mark.parametrize("reasoning", [None, {}, {"effort": "high"}, {"effort": "ultra"}])
def test_max_request_rejects_missing_or_changed_reasoning_after_receipt(monkeypatch, reasoning):
    payload = completed_response()
    payload["reasoning"] = reasoning
    _, connection = transport(monkeypatch, payload)
    sink = Mock()
    request = GenerateRequest("input", "instructions")
    with pytest.raises(ValueError, match="did not confirm the requested max"):
        GPT6Client("test-secret", sink).generate(request)
    connection.request.assert_called_once()
    sink.assert_called_once_with(request.to_body(), payload, 200)


def test_explicit_nonmax_request_preserves_previous_response_semantics(monkeypatch):
    payload = completed_response()
    del payload["reasoning"]
    _, connection = transport(monkeypatch, payload)
    result = GPT6Client("test-secret").generate(GenerateRequest("input", "instructions", reasoning_effort="high"))
    assert json.loads(connection.request.call_args.kwargs["body"])["reasoning"] == {"effort": "high"}
    assert result.response == payload


def test_local_provider_failure_does_not_try_official_endpoint(monkeypatch):
    factory, connection = transport(monkeypatch, completed_response(), scheme="http")
    failure = ConnectionRefusedError("local endpoint unavailable")
    connection.request.side_effect = failure
    official = Mock(side_effect=AssertionError("No provider fallback"))
    monkeypatch.setattr(gpt6, "HTTPSConnection", official)
    with pytest.raises(ConnectionRefusedError) as captured:
        GPT6Client("test-secret", base_url="http://example.test:8123").generate(GenerateRequest("input", "instructions"))
    assert captured.value is failure
    factory.assert_called_once_with("example.test", timeout=120, port=8123)
    connection.request.assert_called_once()
    connection.close.assert_called_once()
    official.assert_not_called()
