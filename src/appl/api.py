"""Configurable Responses transport with bounded, auditable network retries.

Credentials are read at request time from an environment variable. They are never
included in configuration identities, request JSON, or transport receipts.
"""
from __future__ import annotations

import hashlib
from http.client import HTTPException, HTTPSConnection, HTTPConnection
import json
import os
from pathlib import Path
import ssl
import time
from urllib.parse import urlsplit
import uuid

from .journal import atomic, encode


RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
PERMANENT_CODES = frozenset({"insufficient_quota", "billing_hard_limit_reached", "credit_balance_exhausted"})
DEFAULT_DELAYS = (5, 10, 20, 40, 60, 60, 60)


def _redact(value, credential):
    if isinstance(value, str):
        return value.replace(credential, "[REDACTED]") if credential else value
    if isinstance(value, list):
        return [_redact(item, credential) for item in value]
    if isinstance(value, dict):
        return {str(_redact(key, credential)): _redact(item, credential) for key, item in value.items()}
    return value


class ResponsesClient:
    provenance = "responses_api"

    def __init__(self, journal_root, *, model="gpt-5.6-sol", reasoning_effort="xhigh",
                 base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY",
                 timeout_seconds=1800, retry_delays=DEFAULT_DELAYS, allow_insecure_http=False):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("base_url must be a credential-free API URL without query or fragment")
        if parsed.scheme != "https" and not allow_insecure_http:
            raise ValueError("HTTPS is required unless allow_insecure_http is explicitly enabled")
        if not model or not api_key_env or timeout_seconds <= 0:
            raise ValueError("A model, credential environment variable, and positive timeout are required")
        delays = tuple(float(delay) for delay in retry_delays)
        if len(delays) > 7 or any(delay < 0 or delay > 60 for delay in delays):
            raise ValueError("At most seven bounded retry delays of 0–60 seconds are supported")
        self.root = Path(journal_root)
        self.url = parsed
        self.model = model
        self.reasoning = reasoning_effort
        self.key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.delays = delays

    def configuration(self):
        return dict(base_url=self.url.geturl(), model=self.model, reasoning_effort=self.reasoning,
                    api_key_env=self.key_env, timeout_seconds=self.timeout_seconds,
                    retry_delays=list(self.delays), transport_retries=len(self.delays))

    def respond(self, request):
        if request.get("model") != self.model or request.get("reasoning") != {"effort": self.reasoning}:
            raise ValueError("Request model and reasoning must match the recorded configuration")
        credential = os.environ.get(self.key_env, "")
        if not credential:
            raise ValueError(f"Set {self.key_env} before starting an API-backed experiment")
        wire = encode(request).encode("utf-8")
        request_hash = hashlib.sha256(wire).hexdigest()
        logical_id = uuid.uuid4().hex
        request_path = self.root / "transport" / logical_id
        atomic(request_path / "request.json", _redact(request, credential))
        endpoint = self.url.path.rstrip("/")
        if not endpoint.endswith("/responses"):
            endpoint += "/responses"
        for index in range(len(self.delays) + 1):
            connection = (HTTPSConnection if self.url.scheme == "https" else HTTPConnection)(
                self.url.hostname, self.url.port, timeout=self.timeout_seconds)
            started = time.monotonic()
            status, body, error_kind = None, None, None
            try:
                connection.request("POST", endpoint, body=wire, headers={
                    "Content-Type": "application/json", "Authorization": "Bearer " + credential})
                response = connection.getresponse()
                status = response.status
                payload = response.read()
                try:
                    body = json.loads(payload)
                    if not isinstance(body, dict):
                        raise ValueError("Response must be an object")
                except (ValueError, UnicodeDecodeError):
                    body = {"error": {"code": "invalid_json_response"},
                            "payload_sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            except ssl.SSLCertVerificationError:
                error_kind = "SSLCertVerificationError"
                raise
            except (OSError, HTTPException) as error:
                error_kind = type(error).__name__
                body = {"error": {"code": "transport_error", "type": error_kind}}
            finally:
                connection.close()
                atomic(request_path / f"{index + 1:02d}.json", {
                    "attempt": index + 1, "request_sha256": request_hash, "http_status": status,
                    "elapsed_seconds": time.monotonic() - started,
                    "response": _redact(body, credential), "transport_error": error_kind,
                    "possible_provider_charge": error_kind is not None})
            error = body.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            transient = status is None or (status in RETRY_STATUSES and code not in PERMANENT_CODES)
            if transient and index < len(self.delays):
                time.sleep(self.delays[index])
                continue
            if status is None:
                raise TimeoutError("Transport attempts exhausted; request identity and receipts retained")
            if status == 200 and (body.get("error") or not str(body.get("model", "")).startswith(self.model)):
                raise ValueError("Successful response has an error or unexpected model; receipt retained")
            return status, _redact(body, credential)


def client(config, journal_root):
    """Build a client without reading credentials or making any network request."""
    api = config.get("api", config)
    return ResponsesClient(journal_root,
        model=os.environ.get("OPENAI_MODEL", api.get("model", "gpt-5.6-sol")),
        reasoning_effort=api.get("reasoning_effort", "xhigh"),
        base_url=os.environ.get("OPENAI_BASE_URL", api.get("base_url", "https://api.openai.com/v1")),
        api_key_env=api.get("api_key_env", "OPENAI_API_KEY"),
        timeout_seconds=api.get("timeout_seconds", 1800),
        retry_delays=api.get("retry_delays", DEFAULT_DELAYS),
        allow_insecure_http=api.get("allow_insecure_http", False))
