# Anthropic

## Configuration

```yaml
credentials:
  - name: "anthropic_main"
    type: "anthropic"
    api_key: "sk-ant-xxxxx"
    base_url: "https://api.anthropic.com"
    rpm: 60
    tpm: 100000
```

### Anthropic-Compatible Providers

For Comet API, prefer the dedicated [`cometapi`](cometapi.md) provider type.
Other Anthropic-compatible providers can still use `type: "anthropic"` with
an optional `auth_type` override:

```yaml
credentials:
  - name: "compatible_anthropic"
    type: "anthropic"
    api_key: "os.environ/COMPATIBLE_API_KEY"
    auth_type: "bearer"  # optional; default is X-Api-Key
    base_url: "https://api.example.com/v1"
    rpm: 60
    tpm: -1

models:
  - name: "provider/claude-sonnet-4.5"
    model: "claude-sonnet-4-5-20250929"
    credential: compatible_anthropic
    rpm: 60
    tpm: -1
```

### CheapGPT / AIProductiv

CheapGPT can be configured as another Anthropic-compatible provider:

```yaml
credentials:
  - name: "cheapgpt_anthropic"
    type: "anthropic"
    api_key: "os.environ/CHEAPGPT_API_KEY"
    auth_type: "bearer"
    base_url: "https://api.aiproductiv.ru/v1"
    rpm: 60
    tpm: -1

models:
  - name: "cheapgpt/claude-sonnet-4.5"
    model: "claude-sonnet-4-5"
    credential: cheapgpt_anthropic
    rpm: 60
    tpm: -1
```

## Required Fields

| Field      | Description                                        |
| ---------- | -------------------------------------------------- |
| `api_key`  | Anthropic API key (supports `os.environ/VAR_NAME`) |
| `base_url` | API base URL (`https://api.anthropic.com`)         |

By default, Anthropic credentials send `X-Api-Key`. Set `auth_type: "bearer"` for
Anthropic-compatible providers that expect `Authorization: Bearer <token>`.

## Responses API

Anthropic is fully supported via the [Responses API](../advanced/responses.md). Requests are converted natively to the Anthropic Messages API — no Chat Completions intermediate format is used.

Supported features: streaming, `store` / `previous_response_id` multi-turn, tools, thinking/reasoning.

### Responses Files and Documents

For Anthropic-backed routes, Responses `input_file` supports:

| Input form | Anthropic mapping | Status |
| ---------- | ----------------- | ------ |
| `input_file.file_data: "data:application/pdf;base64,<BASE64>"` | `document/source:base64` with `media_type: application/pdf` | Supported |
| `input_file.file_url: "https://..."` | `document/source:url` | Supported |
| `input_file.file_id` | None | Unsupported; returns `400 Bad Request` |

AIR does not resolve OpenAI/LiteLLM file IDs into Anthropic file IDs. Provider file IDs are credential/provider scoped, so clients must send PDF bytes via `file_data` or a provider-accessible URL via `file_url`.

## OpenAI-Compatible API

The router accepts requests in **OpenAI Chat Completion format** and automatically converts them to Anthropic Messages API format. Responses are converted back to OpenAI format.

### Supported Parameters

| OpenAI Parameter        | Anthropic Mapping        | Notes                                               |
| ----------------------- | ------------------------ | --------------------------------------------------- |
| `temperature`           | `temperature`            | Set to 1.0 automatically when thinking is enabled   |
| `top_p`                 | `top_p`                  |                                                     |
| `max_tokens`            | `max_tokens`             | Defaults to 4096 if not set                         |
| `max_completion_tokens` | `max_tokens`             | Fallback if `max_tokens` is empty                   |
| `stop`                  | `stop_sequences`         | Accepts string or array                             |
| `stream`                | `stream`                 |                                                     |
| `tools`                 | `tools`                  | Full conversion (see [Tool Calling](#tool-calling)) |
| `tool_choice`           | `tool_choice`            | See [tool_choice](#tool_choice)                     |
| `user`                  | `metadata.user_id`       | User tracking                                       |
| `reasoning_effort`      | `thinking.budget_tokens` | See [Thinking](#extended-thinking)                  |

#### extra_body Parameters

| Parameter             | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| `extra_body.top_k`    | Top-K sampling                                                         |
| `extra_body.thinking` | Direct thinking config (`{"type": "enabled", "budget_tokens": 15000}`) |

#### Unsupported Parameters

These OpenAI parameters have no Anthropic equivalent and are silently ignored:

`n`, `frequency_penalty`, `presence_penalty`, `seed`, `response_format`, `logprobs`, `top_logprobs`, `modalities`, `service_tier`, `store`, `parallel_tool_calls`, `prediction`

### Message Conversion

| OpenAI Role | Anthropic Handling                                                |
| ----------- | ----------------------------------------------------------------- |
| `system`    | Extracted to top-level `system` field (multiple joined with `\n`) |
| `developer` | Same as `system`                                                  |
| `user`      | `{"role": "user", "content": [ContentBlocks]}`                    |
| `assistant` | `{"role": "assistant", "content": [text + tool_use blocks]}`      |
| `tool`      | `{"role": "user", "content": [{"type": "tool_result", ...}]}`     |

### Tool Calling

#### Standard Function Tools

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="your-key")

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    tool_choice="auto",
)
```

#### Anthropic Built-in Tools

The router also converts these special tool types:

| OpenAI Tool Type                    | Anthropic Type         |
| ----------------------------------- | ---------------------- |
| `computer_use`                      | `computer_20241022`    |
| `text_editor`                       | `text_editor_20241022` |
| `bash`                              | `bash_20241022`        |
| `web_search` / `web_search_preview` | `web_search_20250305`  |

#### tool_choice

| OpenAI Value                                       | Anthropic Mapping                |
| -------------------------------------------------- | -------------------------------- |
| `"none"`                                           | `{"type": "none"}`               |
| `"auto"`                                           | `{"type": "auto"}`               |
| `"required"`                                       | `{"type": "any"}`                |
| `{"type": "function", "function": {"name": "fn"}}` | `{"type": "tool", "name": "fn"}` |

#### Restricting allowed tools

Anthropic's `allowed_tools` tool_choice type is not supported by the Anthropic API or AWS Bedrock. The router emulates the behaviour by filtering the `tools` array to only the listed tools before forwarding the request.

Pass it via `extra_body` (the OpenAI Python SDK merges `extra_body` into the top-level request body):

```python
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "What is the weather and search the docs?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ],
    extra_body={
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "auto",  # "auto" or "any"
            "tools": [{"type": "tool", "name": "get_weather"}],
        }
    },
    max_tokens=200,
)
```

The router converts this to an equivalent Anthropic-compatible request:

- `tools` array is filtered to `[get_weather]` only — `search_docs` is removed
- `tool_choice` becomes `{"type": "auto"}` (or `{"type": "any"}` when `mode` is `"any"`)

| `mode`   | Resulting `tool_choice` | Behaviour                                  |
| -------- | ----------------------- | ------------------------------------------ |
| `"auto"` | `{"type": "auto"}`      | Model may or may not call a tool           |
| `"any"`  | `{"type": "any"}`       | Model must call one of the remaining tools |

### Extended Thinking

The router supports Anthropic's extended thinking for reasoning-capable models.

#### Via reasoning_effort (OpenAI format)

```python
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Solve this step by step..."}],
    reasoning_effort="high",  # minimal, low, medium, high
)
```

| reasoning_effort   | budget_tokens |
| ------------------ | ------------- |
| `minimal`          | 1,000         |
| `low`              | 5,000         |
| `medium`           | 15,000        |
| `high`             | 30,000        |
| `none` / `disable` | Disabled      |

#### Via extra_body.thinking (Anthropic native format)

```python
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    extra_body={"thinking": {"type": "enabled", "budget_tokens": 15000}},
)
```

> When thinking is enabled, `temperature` is automatically set to 1.0 (Anthropic requirement).
>
> Thinking content is returned in the `reasoning_content` field of the response message.

### Content Types

| Content Type          | Support     | Notes                                                                           |
| --------------------- | ----------- | ------------------------------------------------------------------------------- |
| Text                  | Full        | String or `{"type": "text"}` blocks                                             |
| Image (base64)        | Full        | `data:image/{jpeg,png,gif,webp};base64,...` → Anthropic base64 image source     |
| Image (URL)           | Full        | HTTP/HTTPS URLs are downloaded and sent as Anthropic base64 image sources       |
| PDF file data         | Full        | Chat `file.file_data` data URI → Anthropic `document/source:base64`             |
| PDF document block    | Full        | Chat `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"..."}}` is forwarded as an Anthropic document |
| PDF document URL      | Full        | Chat native document `source.type:"url"` is forwarded as `document/source:url`  |
| Chat `file.file_id`   | Unsupported | Returns `400 Bad Request`; AIR does not resolve OpenAI/LiteLLM file IDs         |
| PDF via `image_url`   | Unsupported | Returns `400 Bad Request`; send PDFs as `file` or `document`                    |
| Audio                 | Placeholder | Replaced with `[Audio input: <format> format - not supported by Anthropic API]` |
| Video             | Placeholder | Replaced with `[Video: <url>]`                                                  |

### Streaming

SSE streaming works transparently:

```python
stream = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Anthropic SSE events (`message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`) are converted to OpenAI streaming format in real-time. Tool call arguments are streamed incrementally.

### Finish Reasons

| Anthropic stop_reason | OpenAI finish_reason |
| --------------------- | -------------------- |
| `end_turn`            | `stop`               |
| `max_tokens`          | `length`             |
| `tool_use`            | `tool_calls`         |
| `stop_sequence`       | `stop`               |

### Token Counting

| Anthropic Field               | OpenAI Mapping                        |
| ----------------------------- | ------------------------------------- |
| `input_tokens`                | `prompt_tokens`                       |
| `output_tokens`               | `completion_tokens`                   |
| `cache_read_input_tokens`     | `prompt_tokens_details.cached_tokens` |
| `cache_creation_input_tokens` | Tracked internally for billing        |

### Schema Conversion

When converting tool parameter schemas from OpenAI to Anthropic format:

- `additionalProperties` field is stripped (not supported by Anthropic)
- `strict` field is stripped
- `type` field is normalized to lowercase
- All standard JSON Schema properties are preserved
