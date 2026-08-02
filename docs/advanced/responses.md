# Responses API

Auto AI Router implements the [OpenAI Responses API](../refs/openai_responses_api.md) and routes requests natively to Anthropic, Comet API, Vertex AI, and AWS Bedrock — without converting through Chat Completions format as an intermediary.

## Endpoints

| Method | Path                    | Description                                    |
| ------ | ----------------------- | ---------------------------------------------- |
| `POST` | `/v1/responses`         | Create a response (HTTP, optionally streaming) |
| `GET`  | `/v1/responses`         | Create a response via WebSocket                |
| `GET`  | `/v1/responses/{id}`    | Retrieve a stored response by ID               |
| `POST` | `/v1/responses/compact` | Compact a conversation into a summary item     |

## Request Parameters

All standard Responses API parameters are supported. The table below lists the full set recognized by the router:

| Parameter                | Type             | Description                                             |
| ------------------------ | ---------------- | ------------------------------------------------------- |
| `model`                  | string           | Model ID (required)                                     |
| `input`                  | string \| array  | Conversation input: plain string or input items         |
| `instructions`           | string \| null   | System-level instructions prepended to the request      |
| `max_output_tokens`      | integer          | Maximum tokens in the response                          |
| `max_tool_calls`         | integer          | Maximum number of tool calls per response               |
| `temperature`            | float            | Sampling temperature                                    |
| `top_p`                  | float            | Top-p (nucleus) sampling                                |
| `presence_penalty`       | float            | Presence penalty                                        |
| `frequency_penalty`      | float            | Frequency penalty                                       |
| `top_logprobs`           | integer          | Number of log probabilities to return                   |
| `stop`                   | string \| array  | Stop sequences                                          |
| `stream`                 | boolean          | Enable SSE streaming                                    |
| `background`             | boolean          | Run as a background job                                 |
| `tools`                  | array            | Tools available to the model                            |
| `tool_choice`            | string \| object | Tool selection mode                                     |
| `reasoning`              | object           | Reasoning/thinking configuration                        |
| `text`                   | object           | Text output configuration (e.g. `response_format`)      |
| `store`                  | boolean          | Persist the response (enables `GET /v1/responses/{id}`) |
| `previous_response_id`   | string           | Continue a multi-turn conversation                      |
| `metadata`               | object           | Key-value metadata attached to the response             |
| `include`                | array            | Extra fields to include in the response                 |
| `truncation`             | string           | Truncation mode (`"auto"` \| `"disabled"`)              |
| `user`                   | string           | User identifier                                         |
| `parallel_tool_calls`    | boolean          | Allow parallel tool calls                               |
| `prompt_cache_key`       | string           | Cache key for prompt caching                            |
| `prompt_cache_retention` | string           | Cache retention duration                                |
| `conversation`           | interface        | Conversation context (passthrough)                      |

!!! note "Provider coverage"
Not all providers support every parameter. See the [Provider Support](#provider-support) table below.

!!! note "Service tier is router-controlled"
    `service_tier` is ignored and removed by Auto AI Router. Clients cannot select an upstream
    service tier. The effective tier is controlled by the router/provider configuration.

## Content Types

The `input` array accepts items of different types. Supported `ContentPart` types within messages:

| `type`        | Fields                                                       | Description               |
| ------------- | ------------------------------------------------------------ | ------------------------- |
| `input_text`  | `text`                                                       | Plain text                |
| `input_image` | `image_url` (string or `{url, detail}`), `file_id`, `detail` | Image from URL or file ID |
| `input_audio` | `data` (base64), `format`                                    | Audio clip                |
| `input_file`  | `file_id`, `filename`, `file_url`                            | File reference            |

Input items can also be function call / function call output items for multi-turn tool use:

```json
{"type": "function_call", "call_id": "call_abc", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
{"type": "function_call_output", "call_id": "call_abc", "output": "{\"temp\":22}"}
```

## Multi-Turn Conversations

### Storing Responses

Set `"store": true` to persist a response. A stored response can be retrieved later:

```bash
curl http://localhost:8080/v1/responses/resp_01abc... \
  -H "Authorization: Bearer sk-your-key"
```

### Continuing a Conversation

Pass `previous_response_id` to continue from a prior response. The router reconstructs the previous output as input context before sending to the provider:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-key")

first = client.responses.create(
    model="claude-sonnet-4-20250514",
    input="What is the capital of France?",
    store=True,
)

second = client.responses.create(
    model="claude-sonnet-4-20250514",
    input="And what language do they speak there?",
    previous_response_id=first.id,
    store=True,
)
```

## Streaming

Add `"stream": true` to receive Server-Sent Events. The event sequence follows the Responses API specification:

```
response.created
response.in_progress
  response.output_item.added
  response.content_part.added
  response.output_text.delta  (repeated)
  response.output_text.done
  response.content_part.done
  response.output_item.done
response.completed
[DONE]
```

```python
stream = client.responses.create(
    model="gemini-2.5-flash",
    input="Tell me about Paris",
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

## WebSocket Protocol

The router accepts WebSocket connections on `GET /v1/responses` (with `Upgrade: websocket` header). This allows multiple request-response turns on a single persistent connection.

### Connection

```javascript
const ws = new WebSocket("ws://localhost:8080/v1/responses", {
  headers: { "Authorization": "Bearer sk-your-key" }
});
```

### Sending a Request

Send a JSON message with `"type": "response.create"` and any standard Responses API fields:

```json
{
  "type": "response.create",
  "model": "claude-sonnet-4-20250514",
  "input": "Hello! What is 2+2?",
  "stream": true
}
```

The `type` field is stripped before forwarding to the provider.

### Receiving Events

The server sends each SSE event as a plain JSON text message (no `data:` prefix, no `[DONE]`). Turn completion is signaled by a terminal event (`response.completed`, `response.failed`, `response.incomplete`, `error`).

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "response.output_text.delta") {
    process.stdout.write(data.delta);
  } else if (data.type === "response.completed") {
    console.log("\nDone");
  } else if (data.type === "error") {
    console.error(data.error.message);
  }
};
```

### Error Events

HTTP errors are converted to structured WebSocket error events:

```json
{
  "type": "error",
  "sequence_number": 0,
  "error": {
    "code": "api_error",
    "message": "Rate limit exceeded",
    "type": "server_error",
    "param": null
  }
}
```

### Connection-Local Cache

When `store: false` is explicitly set, completed responses are cached in connection-local memory for the duration of the WebSocket connection. This allows `previous_response_id` continuations within the same session without a persistent store. The cache is cleared on reconnect.

When `store` is absent or `true`, the persistent response store handles continuations across reconnects.

### Multi-Turn Example

```javascript
// First turn
ws.send(JSON.stringify({
  type: "response.create",
  model: "claude-sonnet-4-20250514",
  input: "What is the capital of France?",
  store: false,
}));

// Wait for response.completed, capture response ID, then:
ws.send(JSON.stringify({
  type: "response.create",
  model: "claude-sonnet-4-20250514",
  input: "What language do they speak there?",
  previous_response_id: "<id from first turn>",
  store: false,
}));
```

## Compact API

`POST /v1/responses/compact` summarizes a conversation into a single compaction item. This is useful for reducing context size while preserving essential information.

### Request

```bash
curl -X POST http://localhost:8080/v1/responses/compact \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "input": [
      {"role": "user", "content": "What is photosynthesis?"},
      {"role": "assistant", "content": "Photosynthesis is the process by which plants..."}
    ]
  }'
```

**Requirements:**

- `model` is required
- Request body limit: 10 MB

### Response

```json
{
  "id": "resp_01abc...",
  "object": "response.compaction",
  "created_at": 1234567890,
  "output": [
    {
      "type": "compaction",
      "id": "compact_01xyz...",
      "encrypted_content": "<summary of the conversation>"
    }
  ],
  "usage": {
    "input_tokens": 120,
    "output_tokens": 45,
    "total_tokens": 165
  }
}
```

The `encrypted_content` field contains the model's summary. Use this item in `input` for subsequent requests to continue the conversation from the compacted context.

## Native vs Passthrough Mode

The router uses two modes for Responses API requests:

| Mode            | Description                                                                                |
| --------------- | ------------------------------------------------------------------------------------------ |
| **Native**      | Responses API request → provider-specific format directly. Preserves all provider features |
| **Passthrough** | Responses API request → Chat Completions → provider, then Chat Completions → Responses API |

Native mode is used automatically for **Anthropic**, **Comet API**, **Vertex AI**, and **AWS Bedrock**. Passthrough is used for OpenAI and other providers that already speak Responses API natively.

The mode can be overridden via model configuration:

```yaml
models:
  - name: "my-model"
    passthrough_responses: true  # force passthrough
```

## Provider Support

| Feature                  | Anthropic | Comet API | Vertex AI | Bedrock | OpenAI |
| ------------------------ | --------- | --------- | --------- | ------- | ------ |
| Non-streaming            | ✅        | ✅        | ✅        | ✅      | ✅     |
| Streaming (SSE)          | ✅        | ✅        | ✅        | ✅      | ✅     |
| WebSocket                | ✅        | ✅        | ✅        | ✅      | ✅     |
| `store` / response store | ✅        | ✅        | ✅        | ✅      | ✅     |
| `previous_response_id`   | ✅        | ✅        | ✅        | ✅      | ✅     |
| `tools` (function)       | ✅        | ✅        | ✅        | ✅      | ✅     |
| `reasoning`              | ✅        | ✅        | ✅        | ✅      | ✅     |
| `presence_penalty`       | ❌        | ❌        | ✅        | ❌      | ✅     |
| `frequency_penalty`      | ❌        | ❌        | ✅        | ❌      | ✅     |
| `top_logprobs`           | ❌        | ❌        | ✅        | ❌      | ✅     |
| `compact` endpoint       | ✅        | ✅        | ✅        | ✅      | ✅     |

## Retry and Fallback

When a provider credential returns a rate-limit error (429), the router automatically tries the next available credential of the same type. The original HTTP error code is preserved in the final response — the client receives 429 (not 502) when all credentials of the appropriate type are exhausted.

When no credentials are available at all, the router returns 503 Service Unavailable.
