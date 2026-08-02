# OpenAI Chat Completions API Reference

> **Status:** Partial
>
> **Verified against official sources on:** 2026-03-23
>
> **Confidence:** Medium. Core request and response structure are close to the official API, but some parameter details still need cleanup before this file can be marked verified.
>
> **Official sources:**
>
> - [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)

______________________________________________________________________

## Endpoint

```
POST https://api.openai.com/v1/chat/completions
```

______________________________________________________________________

## Request Parameters

### Required

| Parameter  | Type   | Description                                                       |
| ---------- | ------ | ----------------------------------------------------------------- |
| `model`    | string | Model ID (e.g. `gpt-4o`, `gpt-4o-mini`, `o3`, `o4-mini`, `gpt-5`) |
| `messages` | array  | Conversation history as an array of message objects               |

### Optional — Sampling & Output

| Parameter               | Type         | Default | Description                                                                                                       |
| ----------------------- | ------------ | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `temperature`           | number       | 1       | Sampling temperature 0-2. Higher = more random. **Not supported by reasoning models** (o1/o3/o4-mini/gpt-5).      |
| `top_p`                 | number       | 1       | Nucleus sampling (0-1). **Not supported by reasoning models.**                                                    |
| `frequency_penalty`     | number       | 0       | Penalize repeated tokens (-2.0 to 2.0). **Not supported by reasoning models.**                                    |
| `presence_penalty`      | number       | 0       | Penalize already-present tokens (-2.0 to 2.0). **Not supported by reasoning models.**                             |
| `max_tokens`            | integer      | —       | Legacy max output tokens. **Not supported by reasoning models** (use `max_completion_tokens`).                    |
| `max_completion_tokens` | integer      | —       | Upper bound for output tokens including reasoning tokens. Required for reasoning models.                          |
| `n`                     | integer      | 1       | Number of completions to generate per request.                                                                    |
| `stop`                  | string/array | null    | Up to 4 sequences where generation halts.                                                                         |
| `seed`                  | integer      | —       | Deterministic sampling (best-effort). **Not supported by reasoning models.**                                      |
| `logprobs`              | boolean      | false   | Return log probabilities of output tokens. **Not supported by reasoning models.**                                 |
| `top_logprobs`          | integer      | —       | Number of top log-prob tokens to return (0-20). Requires `logprobs: true`. **Not supported by reasoning models.** |
| `logit_bias`            | object       | null    | Token ID to bias (-100 to 100). **Not supported by reasoning models.**                                            |

### Optional — Tools & Functions

| Parameter             | Type       | Default | Description                                                                      |
| --------------------- | ---------- | ------- | -------------------------------------------------------------------------------- |
| `tools`               | array      | —       | List of tool objects the model may call.                                         |
| `tool_choice`         | string/obj | `auto`  | Controls tool selection: `"none"`, `"auto"`, `"required"`, or specific function. |
| `parallel_tool_calls` | boolean    | true    | Allow model to call multiple tools in one turn.                                  |
| `functions`           | array      | —       | **Deprecated.** Use `tools` instead.                                             |
| `function_call`       | string/obj | —       | **Deprecated.** Use `tool_choice` instead.                                       |

### Optional — Response Format

| Parameter         | Type   | Default | Description                                                                                        |
| ----------------- | ------ | ------- | -------------------------------------------------------------------------------------------------- |
| `response_format` | object | —       | `{"type": "text"}`, `{"type": "json_object"}`, or `{"type": "json_schema", "json_schema": {...}}`. |
| `modalities`      | array  | —       | Output modalities: `["text"]`, `["text", "audio"]`.                                                |
| `audio`           | object | —       | Audio output config: \`{"voice": "...", "format": "wav"                                            |
| `prediction`      | object | —       | Predicted output for speculative decoding (reduces latency).                                       |

### Optional — Reasoning (o1/o3/o4-mini/gpt-5)

| Parameter          | Type   | Default | Description                                              |
| ------------------ | ------ | ------- | -------------------------------------------------------- |
| `reasoning_effort` | string | —       | `"low"`, `"medium"`, `"high"`. Controls reasoning depth. |

### Optional — Streaming

| Parameter        | Type    | Default | Description                                                              |
| ---------------- | ------- | ------- | ------------------------------------------------------------------------ |
| `stream`         | boolean | false   | Enable SSE streaming.                                                    |
| `stream_options` | object  | —       | `{"include_usage": true}` to include usage in the final streaming chunk. |

### Optional — Metadata & Billing

| Parameter            | Type    | Default  | Description                                                   |
| -------------------- | ------- | -------- | ------------------------------------------------------------- |
| `user`               | string  | —        | Unique end-user identifier for abuse monitoring.              |
| `metadata`           | object  | —        | Up to 16 key-value pairs for request tracking.                |
| `store`              | boolean | false    | Store the completion for later retrieval/distillation.        |
| `web_search_options` | object  | —        | Web search integration settings (for models that support it). |

!!! note "Service tier is router-controlled"
    `service_tier` is ignored and removed by Auto AI Router. Clients cannot select an upstream
    service tier. The effective tier is controlled by the router/provider configuration.

______________________________________________________________________

## Message Types

### System Message

```json
{"role": "system", "content": "You are a helpful assistant."}
```

### Developer Message

```json
{"role": "developer", "content": "Follow these instructions..."}
```

Takes precedence over system messages for newer models.

### User Message

```json
{"role": "user", "content": "Hello"}
```

Or with content parts:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "https://...", "detail": "auto"}},
    {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}},
    {"type": "file", "file": {"file_id": "file-abc123"}}
  ]
}
```

### Assistant Message

```json
{
  "role": "assistant",
  "content": "Hello!",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
    }
  ]
}
```

May also contain `audio` (for audio output) and `refusal` fields.

### Tool Message

```json
{"role": "tool", "tool_call_id": "call_abc123", "content": "22 degrees, sunny"}
```

______________________________________________________________________

## Content Part Types

| Type          | Fields                                                             | Description             |
| ------------- | ------------------------------------------------------------------ | ----------------------- |
| `text`        | `text` (string)                                                    | Plain text              |
| `image_url`   | `image_url.url` (string), `image_url.detail` (`auto`/`low`/`high`) | Image via URL or base64 |
| `input_audio` | `input_audio.data` (base64), `input_audio.format` (wav/mp3/etc.)   | Audio input             |
| `file`        | `file.file_id` (string)                                            | Uploaded file reference |

______________________________________________________________________

## Tool Types

### Function Tool

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather",
    "parameters": {"type": "object", "properties": {}, "required": []},
    "strict": true
  }
}
```

### tool_choice Values

| Value                                                   | Behavior                         |
| ------------------------------------------------------- | -------------------------------- |
| `"none"`                                                | Never call tools                 |
| `"auto"`                                                | Model decides                    |
| `"required"`                                            | Must call at least one tool      |
| `{"type": "function", "function": {"name": "fn_name"}}` | Must call the specified function |

______________________________________________________________________

## Response Object

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1711000000,
  "model": "gpt-4o-2024-08-06",
  "system_fingerprint": "fp_abc123",
  "service_tier": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello!",
        "refusal": null,
        "tool_calls": null,
        "audio": null
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "audio_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  }
}
```

______________________________________________________________________

## Streaming Chunk Object

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1711000000,
  "model": "gpt-4o",
  "system_fingerprint": "fp_abc123",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Hello",
        "tool_calls": null,
        "refusal": null
      },
      "finish_reason": null,
      "logprobs": null
    }
  ],
  "usage": null
}
```

When `stream_options.include_usage` is true, the final chunk includes a `usage` object (same schema as non-streaming).

______________________________________________________________________

## finish_reason Values

| Value            | Description                                      |
| ---------------- | ------------------------------------------------ |
| `stop`           | Natural stop or hit a stop sequence              |
| `length`         | Hit `max_tokens` / `max_completion_tokens` limit |
| `tool_calls`     | Model invoked one or more tools                  |
| `content_filter` | Content was filtered by safety systems           |
| `function_call`  | **Deprecated.** Model invoked a function         |

______________________________________________________________________

## Usage Detail Fields

### prompt_tokens_details

| Field           | Type    | Description                     |
| --------------- | ------- | ------------------------------- |
| `cached_tokens` | integer | Tokens served from prompt cache |
| `audio_tokens`  | integer | Tokens from audio input         |

### completion_tokens_details

| Field                        | Type    | Description                                       |
| ---------------------------- | ------- | ------------------------------------------------- |
| `reasoning_tokens`           | integer | Internal reasoning tokens (not visible in output) |
| `audio_tokens`               | integer | Tokens for audio output                           |
| `accepted_prediction_tokens` | integer | Prediction tokens that matched                    |
| `rejected_prediction_tokens` | integer | Prediction tokens that were discarded             |

______________________________________________________________________

## Model-Specific Parameter Restrictions

### Reasoning Models (o1, o3, o3-mini, o3-pro, o4-mini, gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.1, gpt-5.2)

**Not supported** (will error):

- `temperature` (fixed at 1)
- `top_p` (fixed at 1)
- `frequency_penalty`
- `presence_penalty`
- `logprobs` / `top_logprobs`
- `logit_bias`
- `max_tokens` (use `max_completion_tokens` instead)
- `seed`
- `n` > 1

**Supported exclusively:**

- `reasoning_effort` — `"low"`, `"medium"`, `"high"`
- `max_completion_tokens` — includes both visible output and reasoning tokens

> **Note:** Some parameters (e.g., `temperature`) may be accepted on newer GPT-5.2 with `reasoning_effort: "none"`, but are rejected on other reasoning models.

______________________________________________________________________

## response_format Options

| Type          | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `text`        | Default free-form text output                                      |
| `json_object` | Must output valid JSON. Requires "JSON" in system/user message.    |
| `json_schema` | Structured output conforming to a provided JSON Schema (`strict`). |

______________________________________________________________________

## Prompt Caching

Automatically enabled for prompts at or above 1024 tokens. Cached tokens appear in `prompt_tokens_details.cached_tokens` and are billed at a reduced rate. Cache TTL is approximately 5-10 minutes of inactivity.
