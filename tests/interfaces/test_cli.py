import json
from pathlib import Path

import pytest

from experiment_interfaces.cli import main
from experiment_interfaces import gpt6


def request_file(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"input": "Explicit training-only evidence", "instructions": "Existing contract"}))
    return path


def test_api_command_saves_explicit_request_and_complete_response(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")

    def generate(client, request):
        body = request.to_body()
        response = dict(id="resp_test", model="gpt-6-astra", status="completed",
                        usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
        client._response_sink(body, response, 200)
        return gpt6.GenerationResult(body, response, "proposal text", 200)

    monkeypatch.setattr(gpt6.GPT6Client, "generate", generate)
    output = tmp_path / "api_output"
    main(["gpt6", "--request", str(request_file(tmp_path)), "--output", str(output)])
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["request"]["input"] == "Explicit training-only evidence"
    assert json.loads((output / "result.json").read_text())["provenance"] == "openai_responses_api"
    assert "test-secret" not in "".join(path.read_text() for path in output.iterdir())
    with pytest.raises(FileExistsError):
        main(["gpt6", "--request", str(request_file(tmp_path)), "--output", str(output)])


def test_api_command_keeps_incomplete_receipt_without_success_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    error = ValueError("generation incomplete")

    def generate(client, request):
        client._response_sink(request.to_body(), {"status": "incomplete"}, 200)
        raise error

    monkeypatch.setattr(gpt6.GPT6Client, "generate", generate)
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError) as captured:
        main(["gpt6", "--request", str(request_file(tmp_path)), "--output", str(output)])
    assert captured.value is error
    assert json.loads((output / "receipt.json").read_text())["response"]["status"] == "incomplete"
    assert not (output / "result.json").exists()


def test_api_output_cannot_enter_published_namespace(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    with pytest.raises(ValueError, match="published"):
        main(["gpt6", "--request", str(request_file(tmp_path)), "--output", str(root / "policies" / "new_api")])


def test_api_command_uses_only_the_explicit_local_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXAMPLE_API_KEY", "local-test-secret")
    provider = tmp_path / "provider.json"
    provider.write_text(json.dumps(dict(base_url="https://provider.example/v1", api_key_env="EXAMPLE_API_KEY",
                                        http_headers={"x-example-header": "example-value"})))
    calls = []

    def generate(client, request):
        calls.append(client)
        assert client._api_key == "local-test-secret"
        body = request.to_body()
        assert body["model"] == "gpt-6-astra" and body["reasoning"]["effort"] == "max"
        response = dict(id="resp_local", model="gpt-6-astra", status="completed", reasoning={"effort": "max"},
                        usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
        client._response_sink(body, response, 200)
        return gpt6.GenerationResult(body, response, "接口检查通过。", 200)

    monkeypatch.setattr(gpt6.GPT6Client, "generate", generate)
    output = tmp_path / "local-output"
    main(["gpt6", "--request", str(request_file(tmp_path)), "--provider", str(provider), "--output", str(output)])
    assert len(calls) == 1
    result = json.loads((output / "result.json").read_text())
    assert result["requested_reasoning_effort"] == result["returned_reasoning"]["effort"] == "max"
    assert "local-test-secret" not in "".join(path.read_text() for path in output.iterdir())


def test_provider_file_does_not_accept_embedded_credentials(tmp_path):
    provider = tmp_path / "provider.json"
    provider.write_text(json.dumps({"api_key": "must-use-environment"}))
    with pytest.raises(ValueError, match="environment credentials"):
        main(["gpt6", "--request", str(request_file(tmp_path)), "--provider", str(provider),
              "--output", str(tmp_path / "out")])


