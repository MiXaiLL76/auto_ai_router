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

The HTTP proxy's request/response body debug records, unified upstream error
records and optional file error logs omit body text and retain a byte count.
Provider errors may echo prompts, tool output or credentials in plain text;
truncation does not protect that content. The delivered client response is
unchanged. File error logs mask known authentication headers and cookies on
both request and response sides, regardless of header-name casing.

Kafka body diagnostics follow `kafka.raw_bodies.redact_sensitive_fields`, which
defaults to `true`. Captured request content is redacted and response content
is omitted before publishing. An explicit `false` allows captured body text
in this separate debug topic. See [Kafka logging](../litellm-integration/kafka_spend_log.md).
These controls do not remove historical logs or guarantee that arbitrary custom
headers, exception messages or application-defined metadata contain no secrets.
