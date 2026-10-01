"""One explicit Responses API request; no evidence discovery or policy execution.

The caller supplies the complete input and design instructions. This module
does not read datasets, change prompts, retry requests, or substitute models.
Request/response artifacts contain no authorization headers or credentials.

API contract: https://developers.openai.com/api/reference/python/resources/responses/methods/create
Model: https://developers.openai.com/api/docs/models/gpt-6-astra
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
import json
import math
import os
import re
from typing import Callable
from urllib.parse import urlsplit


ResponseSink = Callable[[dict, dict, int], None]


@dataclass(frozen=True)
class GenerateRequest:
    """Explicit text or Responses input items, including supplied image content.

    Input items retain the API's native shape. Local paths are never read or
    converted into images, files, evidence, or extra instructions here.
    ``max_output_tokens`` includes both reasoning and visible output tokens.
    """

    input: str | list[dict]
    instructions: str
    model: str = "gpt-6-astra"
    reasoning_effort: str = "max"
    max_output_tokens: int = 4096
    timeout_seconds: float = 120

    def __post_init__(self):
        if not isinstance(self.input, (str, list)) or not self.input:
            raise ValueError("input must be nonempty text or a list of Responses input items")
        if isinstance(self.input, list) and any(not isinstance(item, dict) for item in self.input):
            raise ValueError("Responses input items must be dictionaries")
        if not isinstance(self.instructions, str):
            raise ValueError("instructions must be explicit text")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be an explicit nonempty model ID")
        if self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported GPT-6 Astra reasoning effort")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 16:
            raise ValueError("max_output_tokens must be an integer of at least 16")
        if (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be positive and finite")

    def to_body(self) -> dict:
        """Return the exact credential-free JSON body used for generation."""
        return {
            "model": self.model,
            "input": deepcopy(self.input),
            "instructions": self.instructions,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }


@dataclass(frozen=True)
class GenerationResult:
    """A completed textual response plus its original request and response JSON."""

    request: dict
    response: dict
    output_text: str
    http_status: int

    @property
    def response_id(self) -> str:
        return self.response["id"]

    @property
    def model(self) -> str:
        return self.response["model"]

    @property
    def usage(self) -> dict:
        return self.response["usage"]

    @property
    def status(self) -> str:
        return self.response["status"]


def _completed_text(response: dict) -> str:
    if response.get("object") != "response":
        raise ValueError("Expected a Responses API response object")
    if response.get("status") != "completed":
        raise ValueError("Responses API generation did not complete")
    if response.get("error") is not None or response.get("incomplete_details") is not None:
        raise ValueError("Responses API reported an error or incomplete generation")
    for field in ("id", "model"):
        if not isinstance(response.get(field), str) or not response[field]:
            raise ValueError(f"Responses API response is missing {field}")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Completed generation requires reported token usage")
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        if type(usage.get(field)) is not int or usage[field] < 0:
            raise ValueError(f"Invalid Responses API usage field: {field}")
    if not isinstance(response.get("output"), list):
        raise ValueError("Responses API output must be a list")
    texts = []
    for item in response["output"]:
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            raise ValueError("Malformed Responses API output item")
        if item["type"] != "message":
            continue
        if item.get("role") != "assistant" or item.get("status") != "completed":
            raise ValueError("Responses API output message is not a completed assistant message")
        if not isinstance(item.get("content"), list):
            raise ValueError("Responses API message content must be a list")
        for block in item["content"]:
            if not isinstance(block, dict):
                raise ValueError("Malformed Responses API message content")
            if block.get("type") == "refusal":
                raise ValueError("Responses API refused the generation request")
            if block.get("type") != "output_text" or not isinstance(block.get("text"), str):
                raise ValueError("Expected a Responses API output_text block")
            texts.append(block["text"])
    output_text = "".join(texts)
    if not output_text.strip():
        raise ValueError("Completed Responses API response has no output text")
    return output_text


def _provider_url(base_url: str) -> tuple[str, str, int | None, str]:
    if (not isinstance(base_url, str) or not base_url or not base_url.isascii()
            or any(ord(character) <= 32 or ord(character) == 127 for character in base_url)):
        raise ValueError("base_url must be an explicit ASCII HTTP or HTTPS URL without whitespace")
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base_url requires an explicit HTTP or HTTPS scheme and hostname")
    if parsed.username is not None or parsed.password is not None or "?" in base_url or "#" in base_url:
        raise ValueError("base_url cannot contain credentials, a query, or a fragment")
    port = parsed.port
    if parsed.netloc.endswith(":") or (port is not None and not 1 <= port <= 65535):
        raise ValueError("base_url has an invalid explicit port")
    path = parsed.path + ("" if parsed.path.endswith("/") else "/") + "responses"
    return parsed.scheme, parsed.hostname, port, path


def _provider_headers(http_headers: dict[str, str] | None) -> dict[str, str]:
    if http_headers is None:
        return {}
    if not isinstance(http_headers, dict):
        raise ValueError("http_headers must be a dictionary of explicit header strings")
    reserved = {
        "authorization", "content-type", "accept", "host", "content-length",
        "transfer-encoding", "connection", "proxy-connection", "proxy-authorization",
        "upgrade", "te", "trailer", "expect", "content-encoding", "keep-alive",
    }
    headers, seen = {}, set()
    for name, value in http_headers.items():
        if not isinstance(name, str) or re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name) is None:
            raise ValueError("Custom HTTP header names must be valid HTTP tokens")
        normalized = name.lower()
        if normalized in reserved or normalized in seen:
            raise ValueError("Custom HTTP headers cannot override protocol headers or duplicate names")
        if (not isinstance(value, str)
                or any(ord(character) < 32 or ord(character) == 127 or ord(character) > 255
                       for character in value)):
            raise ValueError("Custom HTTP header values must be single-line HTTP strings")
        headers[name] = value
        seen.add(normalized)
    return headers


class GPT6Client:
    """Explicit HTTP(S) Responses provider and receipt sink before validation.

    ``api_key=None`` selects only the named ``api_key_env`` process variable.
    No key files, OAuth, or other credential sources are consulted. The exact
    base URL selects HTTP or HTTPS and its full path prefixes ``/responses``.
    Custom headers cannot replace authorization or transport framing headers.
    ``response_sink`` receives the request body, decoded response object and
    HTTP status, including incomplete/refusal/error objects. It can save the
    raw receipt before a contract violation propagates to the caller.
    """

    def __init__(self, api_key: str | None = None, response_sink: ResponseSink | None = None,
                 *, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", http_headers: dict[str, str] | None = None):
        self._scheme, self._host, self._port, self._request_path = _provider_url(base_url)
        self._http_headers = _provider_headers(http_headers)
        if not isinstance(api_key_env, str) or not api_key_env or "=" in api_key_env or "\0" in api_key_env:
            raise ValueError("api_key_env must explicitly name one process environment variable")
        self._api_key = os.environ[api_key_env] if api_key is None else api_key
        if not isinstance(self._api_key, str) or not self._api_key.strip():
            raise ValueError("A nonempty OpenAI API key is required")
        if any(ord(character) <= 32 or ord(character) >= 127 for character in self._api_key):
            raise ValueError("Bearer credentials must be nonempty ASCII without whitespace")
        self._response_sink = response_sink

    def generate(self, request: GenerateRequest) -> GenerationResult:
        body = request.to_body()
        payload, http_status = self._send_body(body, request.timeout_seconds)
        output_text = _completed_text(payload)
        if request.reasoning_effort == "max":
            reasoning = payload.get("reasoning")
            if not isinstance(reasoning, dict) or reasoning.get("effort") != "max":
                raise ValueError("Responses provider did not confirm the requested max reasoning effort")
        return GenerationResult(body, payload, output_text, http_status)

    def _send_body(self, body: dict, timeout_seconds: float) -> tuple[dict, int]:
        """Shared transport for text and explicit function-tool Responses requests."""
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        connection_factory = HTTPSConnection if self._scheme == "https" else HTTPConnection
        connection_options = {"timeout": timeout_seconds}
        if self._port is not None:
            connection_options["port"] = self._port
        with closing(connection_factory(self._host, **connection_options)) as connection:
            connection.request("POST", self._request_path, body=encoded, headers={
                **self._http_headers,
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
            response = connection.getresponse()
            http_status = response.status
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise ValueError("Responses API response JSON must be an object")
        if self._response_sink is not None:
            self._response_sink(deepcopy(body), deepcopy(payload), http_status)
        if not 200 <= http_status < 300:
            raise RuntimeError(f"Responses API returned HTTP {http_status}")
        return payload, http_status
