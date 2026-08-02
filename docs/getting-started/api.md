# API Usage

Auto AI Router exposes an OpenAI-compatible API. Any client that works with the OpenAI API can be pointed at the router.

## Chat Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-your-master-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Streaming

Add `"stream": true` to enable Server-Sent Events streaming:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-your-master-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-your-master-key-here",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Anthropic Messages API

`POST /v1/messages` accepts Anthropic Messages requests and adapts them to the router's Chat Completions pipeline. Authentication, model routing, balancing, fallbacks, and usage accounting are the same as for `/v1/chat/completions`.

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H "x-api-key: sk-your-master-key-here" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Set `"stream": true` to receive Anthropic-compatible `message_start`, content block, `message_delta`, and `message_stop` SSE events. Both `x-api-key` and `Authorization: Bearer ...` authentication are supported.

## Responses API

The router supports the [OpenAI Responses API](../advanced/responses.md) with native provider integration for Anthropic, Comet API, Vertex AI, and AWS Bedrock.

```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Authorization: Bearer sk-your-master-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "input": "Hello!"
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Authorization: Bearer sk-your-master-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "input": "Hello!",
    "stream": true
  }'
```

### WebSocket

Connect to `ws://localhost:8080/v1/responses` (with `Upgrade: websocket`) for persistent multi-turn sessions. See [Responses API — WebSocket](../advanced/responses.md#websocket-protocol) for the full protocol.

### Compact

```bash
curl -X POST http://localhost:8080/v1/responses/compact \
  -H "Authorization: Bearer sk-your-master-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "input": "..."
  }'
```

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-your-master-key-here",
)

response = client.responses.create(
    model="claude-sonnet-4-20250514",
    input="Hello!",
)
print(response.output_text)
```

## Health Check

```bash
# JSON format
curl http://localhost:8080/health

# HTML dashboard
curl http://localhost:8080/vhealth
```

See [Health Endpoints](../monitoring/health.md) for details on the response format.

## Authentication

All API requests require the `Authorization` header with the master key:

```
Authorization: Bearer sk-your-master-key-here
```

Health endpoints (`/health`, `/vhealth`, `/metrics`) do not require authentication. If credential scopes are configured, unauthenticated health views only include unscoped credentials; pass an API key to see that key's scoped view.
