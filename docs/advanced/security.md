# Security

## Environment Variables

Configuration supports values from environment variables using the `os.environ/VARIABLE_NAME` pattern:

```yaml
server:
  master_key: "os.environ/MASTER_KEY"
  model_prices_link: "os.environ/MODEL_PRICES_URL"

credentials:
  - name: "openai"
    type: "openai"
    api_key: "os.environ/OPENAI_API_KEY"
    base_url: "https://api.openai.com"

  - name: "vertex_ai"
    type: "vertex-ai"
    project_id: "os.environ/GCP_PROJECT_ID"
    location: "us-central1"
    credentials_json: "os.environ/VERTEX_CREDENTIALS"

  - name: "gemini_studio"
    type: "gemini"
    api_key: "os.environ/GEMINI_API_KEY"
    base_url: "https://generativelanguage.googleapis.com"

litellm_db:
  enabled: true
  database_url: "os.environ/LITELLM_DATABASE_URL"
```

Set the variables before starting the router:

```bash
export MASTER_KEY="sk-your-master-key"
export OPENAI_API_KEY="sk-proj-..."
export GCP_PROJECT_ID="my-project"
export GEMINI_API_KEY="AIza..."
export LITELLM_DATABASE_URL="postgresql://user:pass@localhost/litellm"

./auto_ai_router -config config.yaml
```

## Master Key Authentication

All API requests require the `Authorization` header with the master key:

```bash
curl -H "Authorization: Bearer sk-your-master-key" http://localhost:8080/v1/chat/completions ...
```

Health and metrics endpoints (`/health`, `/vhealth`, `/metrics`) do not require authentication. If credential scopes are configured, unauthenticated health views only include unscoped credentials; pass an API key to see that key's scoped view.

## LiteLLM API Key Auth

When [LiteLLM DB integration](../litellm-integration/litellm_db.md) is enabled, the router also validates API keys against the LiteLLM verification token table. This allows using LiteLLM-issued API keys alongside the master key.

## Request and response content in logs

Where body text appears in the router's own logs, and where it does not:

| Sink                                                                                                   | Body content                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Unified upstream error record (the `ERROR` line every final provider failure goes through)             | The provider's error body, truncated per field to 500 characters — except for CometAPI and ProMan credentials, whose bodies are replaced by a byte count                                                                                                                                                                 |
| Debug records: "Proxy request details", "Proxy response body", "Transformed response to OpenAI format" | The real body, truncated the same way; the request body additionally has base64/`file_data` payloads replaced by a byte count. For a CometAPI/ProMan credential, the "Proxy response body" record omits the body entirely on an error status and logs only the status. Debug level is off unless the operator enables it |
| Retry warnings ("Provider returned retryable error, will retry", and its stream-error sibling)         | The provider's error body, truncated the same way, subject to the same CometAPI/ProMan exception                                                                                                                                                                                                                         |
| The optional file error log (`monitoring.errors_log_path`)                                             | The request and response bodies verbatim; only the headers are masked                                                                                                                                                                                                                                                    |
| The spend log's `error_message` column (and the metadata copy of it)                                   | Never the provider's free text: the failure is classified from the status and body through an allowlisted classifier                                                                                                                                                                                                     |

A provider's error body may echo prompts, tool output or credentials in plain
text, and truncating a field does not protect that content — so these logs are
a place where a prompt can surface. The delivered client response is unchanged
either way: what the client receives is masked separately (see the upstream
error masking in `internal/proxy/errors.go`), independently of what the logs
keep. File error logs mask known authentication headers and cookies on both
request and response sides, regardless of header-name casing.

The CometAPI and ProMan exception exists because those providers have been
seen to echo the caller's own credential back inside their error bodies; for
them the log keeps only a byte count (or, for the debug response record, only
the status) — the same treatment the client-facing response gets.

The long-lived spend event is the one sink that never carries the provider's
free text: its `error_message` comes from the allowlisted classifier, so a
reflected prompt does not outlive the debugging session in the billing table.

Kafka body diagnostics are a separate path with their own controls.
`kafka.raw_bodies.redact_sensitive_fields` and
`kafka.router_responses.redact_sensitive_fields` both default to `true`: the
first replaces captured request content and omits the client-facing response
content before publishing, the second keeps the response's structure (model,
usage, ids, finish reasons) and replaces the model's own text with a
placeholder. An explicit `false` allows captured body text in those separate
debug topics.

One body in the raw-bodies topic is deliberately *not* covered by that
redaction: `response_body` carries the provider's own error text verbatim,
because it is the provider describing its own failure and nothing else says
*why* a request failed. The CometAPI/ProMan exception above applies there
too — those credentials' provider bodies are reduced to a byte count so the
echoed credential is not published with them.
See [Kafka logging](../litellm-integration/kafka_spend_log.md).

These controls do not remove historical logs or guarantee that arbitrary custom
headers, exception messages or application-defined metadata contain no secrets.
