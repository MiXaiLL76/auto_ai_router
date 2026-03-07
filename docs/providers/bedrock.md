# AWS Bedrock

## Configuration

```yaml
credentials:
  - name: "bedrock_us_east"
    type: "bedrock"
    api_key: "os.environ/AWS_BEDROCK_API_KEY"
    base_url: "https://bedrock-runtime.us-east-1.amazonaws.com"
    rpm: 60
    tpm: 100000
```

## Required Fields

| Field      | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| `api_key`  | AWS Bedrock API key / token (supports `os.environ/VAR_NAME`) |
| `base_url` | Bedrock Runtime endpoint URL (region-specific)               |

The `api_key` is sent as `Authorization: Bearer <api_key>` to the Bedrock Runtime API.

### Common Base URLs by Region

| Region         | Base URL                                               |
| -------------- | ------------------------------------------------------ |
| us-east-1      | `https://bedrock-runtime.us-east-1.amazonaws.com`      |
| us-west-2      | `https://bedrock-runtime.us-west-2.amazonaws.com`      |
| eu-central-1   | `https://bedrock-runtime.eu-central-1.amazonaws.com`   |
| ap-northeast-1 | `https://bedrock-runtime.ap-northeast-1.amazonaws.com` |

## Model IDs

Use AWS Bedrock model IDs in the `model` field. Cross-region inference profiles are supported:

```
us.anthropic.claude-sonnet-4-20250514-v1:0
us.anthropic.claude-3-5-sonnet-20241022-v2:0
us.anthropic.claude-3-haiku-20240307-v1:0
anthropic.claude-3-5-haiku-20241022-v1:0
```

## OpenAI-Compatible API

The router accepts requests in **OpenAI Chat Completion format** and automatically converts them to AWS Bedrock InvokeModel format. The conversion pipeline:

1. OpenAI → Anthropic Messages API format (full parameter conversion)
2. Bedrock-specific adjustments: remove `model` (embedded in URL), remove `stream`, add `anthropic_version: "bedrock-2023-05-31"`
3. Response: Bedrock (Anthropic) → OpenAI format

### Request URL Construction

| Mode          | URL                                                       |
| ------------- | --------------------------------------------------------- |
| Non-streaming | `{base_url}/model/{model_id}/invoke`                      |
| Streaming     | `{base_url}/model/{model_id}/invoke-with-response-stream` |

### Supported Parameters

Same as the [Anthropic provider](./anthropic.md#supported-parameters):

| OpenAI Parameter        | Notes                                               |
| ----------------------- | --------------------------------------------------- |
| `temperature`           | Set to 1.0 automatically when thinking is enabled   |
| `top_p`                 |                                                     |
| `max_tokens`            | Defaults to 4096 if not set                         |
| `max_completion_tokens` | Fallback if `max_tokens` is empty                   |
| `stop`                  | Accepts string or array                             |
| `stream`                |                                                     |
| `tools`                 | Full conversion (see [Tool Calling](#tool-calling)) |
| `tool_choice`           |                                                     |
| `user`                  | Mapped to `metadata.user_id`                        |
| `reasoning_effort`      | See [Extended Thinking](#extended-thinking)         |

#### extra_body Parameters

| Parameter             | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| `extra_body.top_k`    | Top-K sampling                                                         |
| `extra_body.thinking` | Direct thinking config (`{"type": "enabled", "budget_tokens": 15000}`) |

#### Unsupported Parameters

`n`, `frequency_penalty`, `presence_penalty`, `seed`, `response_format`, `logprobs`, `top_logprobs`, `modalities`, `service_tier`, `store`, `parallel_tool_calls`, `prediction`

Embeddings and image generation are not supported by this provider.

### Message Conversion

Same as [Anthropic provider](./anthropic.md#message-conversion).

### Tool Calling

Same as [Anthropic provider](./anthropic.md#tool-calling). Supported tool types:

| OpenAI Tool Type                    | Anthropic (Bedrock) Type |
| ----------------------------------- | ------------------------ |
| `function`                          | Standard function tool   |
| `computer_use`                      | `computer_20241022`      |
| `text_editor`                       | `text_editor_20241022`   |
| `bash`                              | `bash_20241022`          |
| `web_search` / `web_search_preview` | `web_search_20250305`    |

### Extended Thinking

Supported for reasoning-capable models (e.g. `claude-sonnet-4`). Same interface as [Anthropic provider](./anthropic.md#extended-thinking).

```python
response = client.chat.completions.create(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
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

> When thinking is enabled, `temperature` is automatically set to 1.0.
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

AWS Bedrock uses a **binary Event Stream** framing protocol instead of SSE. The router decodes it transparently:

1. Binary AWS Event Stream frames decoded on the fly
2. Each frame payload (base64-encoded Anthropic event JSON) converted to SSE
3. Anthropic SSE events converted to OpenAI streaming format

The client receives standard OpenAI-compatible SSE:

```python
stream = client.chat.completions.create(
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Finish Reasons

| Bedrock/Anthropic stop_reason | OpenAI finish_reason |
| ----------------------------- | -------------------- |
| `end_turn`                    | `stop`               |
| `max_tokens`                  | `length`             |
| `tool_use`                    | `tool_calls`         |
| `stop_sequence`               | `stop`               |

### Token Counting

| Anthropic Field               | OpenAI Mapping                        |
| ----------------------------- | ------------------------------------- |
| `input_tokens`                | `prompt_tokens`                       |
| `output_tokens`               | `completion_tokens`                   |
| `cache_read_input_tokens`     | `prompt_tokens_details.cached_tokens` |
| `cache_creation_input_tokens` | Tracked internally for billing        |
