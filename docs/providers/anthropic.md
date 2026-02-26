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

## Required Fields

| Field      | Description                                        |
| ---------- | -------------------------------------------------- |
| `api_key`  | Anthropic API key (supports `os.environ/VAR_NAME`) |
| `base_url` | API base URL (`https://api.anthropic.com`)         |

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

| Content Type      | Support     | Notes                                                                           |
| ----------------- | ----------- | ------------------------------------------------------------------------------- |
| Text              | Full        | String or `{"type": "text"}` blocks                                             |
| Image (base64)    | Full        | `data:image/...;base64,...` → Anthropic base64 source                           |
| Image (URL)       | Full        | HTTP/HTTPS URLs → Anthropic URL source                                          |
| Document (base64) | Full        | `application/*` and `text/*` MIME types only                                    |
| Audio             | Placeholder | Replaced with `[Audio input: <format> format - not supported by Anthropic API]` |
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
