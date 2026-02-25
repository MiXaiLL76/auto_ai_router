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

Health endpoints (`/health`, `/vhealth`, `/metrics`) do not require authentication.
