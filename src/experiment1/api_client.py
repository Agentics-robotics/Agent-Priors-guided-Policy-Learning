"""Stateless Responses function tools; no embedded credentials or automatic retries."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os

from experiment_interfaces.gpt6 import GPT6Client


@dataclass(frozen=True)
class APIConfig:
    provider: str = "OpenAI"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-6-astra"
    api_key_env: str = "OPENAI_API_KEY"
    mode: str = "responses"
    reasoning_effort: str = "max"
    timeout_seconds: float = 600
    max_output_tokens: int = 16384

    @classmethod
    def from_environment(cls):
        fields = {
            "provider": "PROVIDER", "base_url": "BASE_URL", "model": "MODEL",
            "api_key_env": "KEY_ENV_NAME", "mode": "MODE",
            "timeout_seconds": "TIMEOUT_SECONDS", "max_output_tokens": "MAX_OUTPUT_TOKENS",
        }
        values = {field: os.environ["EXPERIMENT1_API_" + key] for field, key in fields.items()
                  if "EXPERIMENT1_API_" + key in os.environ}
        for field, key in (("base_url", "OPENAI_BASE_URL"), ("model", "OPENAI_MODEL")):
            if field not in values and key in os.environ:
                values[field] = os.environ[key]
        for field, cast in (("timeout_seconds", float), ("max_output_tokens", int)):
            if field in values:
                values[field] = cast(values[field])
        return cls(**values)

    def public_record(self):
        return asdict(self)


class ToolResponsesClient(GPT6Client):
    provenance = "openai_responses_api"

    def __init__(self, config: APIConfig, response_sink=None):
        if config.mode != "responses" or config.reasoning_effort != "max":
            raise ValueError("Experiment 1 requires explicit Responses / max configuration")
        super().__init__(base_url=config.base_url, api_key_env=config.api_key_env,
                         response_sink=response_sink)
        self.config = config

    def body(self, history, instructions, tools):
        return dict(model=self.config.model, input=history, instructions=instructions,
                    reasoning={"effort": self.config.reasoning_effort},
                    tools=tools, tool_choice="auto", parallel_tool_calls=False,
                    include=["reasoning.encrypted_content"], store=False,
                    max_output_tokens=self.config.max_output_tokens)

    def respond(self, body):
        payload, status = self._send_body(body, self.config.timeout_seconds)
        self.validate_response(payload)
        return payload, status

    def validate_response(self, response):
        if response.get("object") != "response" or response.get("status") != "completed":
            raise ValueError("The design response did not complete; raw receipt retained")
        if response.get("error") is not None or response.get("incomplete_details") is not None:
            raise ValueError("Design response includes an API error")
        if response.get("model") != self.config.model or response.get("reasoning", {}).get("effort") != "max":
            raise ValueError("Returned design model / reasoning effort differs from the frozen request")
        if not isinstance(response.get("id"), str) or not response["id"]:
            raise ValueError("Missing provider response ID")
        usage = response.get("usage", {})
        if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens", "total_tokens")):
            raise ValueError("Missing API token accounting")
        if not isinstance(response.get("output"), list):
            raise ValueError("Invalid Responses output")
        for item in response["output"]:
            if item.get("type") == "function_call":
                if any(not isinstance(item.get(k), str) or not item[k] for k in ("call_id", "name", "arguments")):
                    raise ValueError("Malformed function call")
            elif item.get("type") == "message":
                if any(part.get("type") == "refusal" for part in item.get("content", [])):
                    raise ValueError("Design API refused the request")
            elif item.get("type") != "reasoning":
                raise ValueError("Only declared function tools, messages and reasoning are accepted")
