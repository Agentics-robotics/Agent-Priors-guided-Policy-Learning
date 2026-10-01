# Model API configuration

Offline result reproduction and policy inspection require no API key. New
construction-agent sessions and APPL runtime composition require a compatible
Responses API provider and access to the configured model.

Set `OPENAI_API_KEY` in your shell. `OPENAI_BASE_URL` defaults to
`https://api.openai.com/v1`; an optional `OPENAI_MODEL` override creates a
different recorded experiment identity. The paper used GPT-6 Astra with `max`
effort for Exp1 and GPT-5.6 Sol with `xhigh` effort for Exp2. Availability of those
models depends on the account/provider. No alternative model is selected silently.

The public transport has no embedded credentials, private routing headers, or
machine-specific provider addresses. HTTPS is the default. A separately configured
insecure endpoint is an explicit user choice. Transport retries preserve the
scientific request and write attempt receipts; quota exhaustion and scientific
failures are not retried as new experiments. Provider charges can remain uncertain
after a disconnected request.

The interface follows the official [authentication reference](https://developers.openai.com/api/reference/overview)
and [Responses function-calling guide](https://developers.openai.com/api/docs/guides/function-calling).
