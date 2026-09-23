# Vertex AI

## Configuration

### With Service Account File

```yaml
credentials:
  - name: "vertex_ai"
    type: "vertex-ai"
    project_id: "your-gcp-project"
    location: "global"
    credentials_file: "path/to/service-account.json"
    rpm: 100
    tpm: 50000
```

### With Credentials JSON (environment variable)

```yaml
credentials:
  - name: "vertex_ai"
    type: "vertex-ai"
    project_id: "os.environ/GCP_PROJECT_ID"
    location: "us-central1"
    credentials_json: "os.environ/VERTEX_CREDENTIALS"
    rpm: 100
    tpm: 50000
```

## Required Fields

| Field              | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `project_id`       | GCP project ID                                             |
| `location`         | GCP region (e.g., `global`, `us-central1`, `europe-west1`) |
| `credentials_file` | Path to service account JSON file                          |
| `credentials_json` | **Or** service account JSON content as a string            |

!!! note
Provide either `credentials_file` or `credentials_json`, not both.

## Authentication

Vertex AI uses OAuth2 tokens obtained from the service account. The router automatically manages token refresh with coalesced concurrent requests.

## Multiple Credentials

You can configure multiple Vertex AI credentials for load balancing:

```yaml
credentials:
  - name: "vertex_project_a"
    type: "vertex-ai"
    project_id: "project-a"
    location: "global"
    credentials_file: "sa-a.json"
    rpm: 100
    tpm: 50000

  - name: "vertex_project_b"
    type: "vertex-ai"
    project_id: "project-b"
    location: "global"
    credentials_file: "sa-b.json"
    rpm: 100
    tpm: 50000
```

Requests are distributed across credentials using round-robin. See [Load Balancing](../advanced/balancing.md).

## Responses API

Vertex AI is fully supported via the [Responses API](../advanced/responses.md). Requests are converted natively to the Vertex AI GenerateContent API — no Chat Completions intermediate format is used.

Supported features: streaming, `store` / `previous_response_id` multi-turn, tools, thinking/reasoning.

## OpenAI-Compatible API

The router accepts requests in **OpenAI Chat Completion format** and automatically converts them to Vertex AI (GenAI) format. Responses are converted back to OpenAI format, so any OpenAI SDK works transparently.

For thinking-capable Gemini models, the router treats "thinking depth" and "thought disclosure" separately:

- `reasoning_effort`, `thinking_budget`, `thinking_level`, and Anthropic-style `thinking` control reasoning depth only.
- These shorthands do **not** enable `include_thoughts`; internal thoughts are hidden by default.
- To receive `reasoning_content`, explicitly set `extra_body.thinking_config.include_thoughts=true`.

### Supported Parameters

| OpenAI Parameter      | Vertex Mapping                        | Notes                                    |
| --------------------- | ------------------------------------- | ---------------------------------------- |
| `temperature`         | `Temperature`                         |                                          |
| `top_p`               | `TopP`                                |                                          |
| `seed`                | `Seed`                                |                                          |
| `frequency_penalty`   | `FrequencyPenalty`                    |                                          |
| `presence_penalty`    | `PresencePenalty`                     |                                          |
| `max_tokens`          | `MaxOutputTokens`                     |                                          |
| max_completion_tokens | `MaxOutputTokens`                     | Takes precedence over `max_tokens`       |
| `n`                   | `CandidateCount`                      |                                          |
| `stop`                | `StopSequences`                       | Accepts string or array                  |
| `response_format`     | `ResponseMIMEType` + `ResponseSchema` | Supports `json_schema` and `json_object` |
| `logprobs`            | `ResponseLogprobs`                    |                                          |
| `top_logprobs`        | `Logprobs`                            |                                          |

#### extra_body Parameters

Additional parameters can be passed via `extra_body` for Vertex-specific features:

| Parameter                                          | Description                                                                                      |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `extra_body.generation_config.top_k`               | Top-K sampling                                                                                   |
| `extra_body.generation_config.response_modalities` | Output modalities (`["TEXT"]`, `["IMAGE"]`, `["AUDIO"]`)                                         |
| `extra_body.generation_config.temperature`         | Override temperature                                                                             |
| `extra_body.audio`                                 | Audio output config (see [Audio Output](#audio-output))                                          |
| `extra_body.thinking_config`                       | Gemini-native thinking config (see [Thinking](#reasoning-thinking))                              |
| `extra_body.thinking_budget`                       | Gemini 2.5 token budget shorthand (see [Thinking](#reasoning-thinking))                          |
| `extra_body.thinking_level`                        | Gemini 3+ level shorthand: `minimal`/`low`/`medium`/`high` (see [Thinking](#reasoning-thinking)) |
| `extra_body.thinking`                              | Anthropic-style thinking config (see [Thinking](#reasoning-thinking))                            |
| `extra_body.reasoning_effort`                      | OpenAI-style effort: `low`/`medium`/`high`/`disable` (see [Thinking](#reasoning-thinking))       |

#### Unsupported Parameters

These OpenAI parameters have no Vertex AI equivalent and are silently ignored:

`logit_bias`, `user`, `store`, `service_tier`, `metadata`, `parallel_tool_calls`, `stream_options`, `prediction`

### Tool Calling

All OpenAI tool types are supported:

| OpenAI Tool Type                    | Vertex Mapping                                        |
| ----------------------------------- | ----------------------------------------------------- |
| `function`                          | `FunctionDeclarations` (grouped in one Tool)          |
| `computer_use`                      | `ComputerUse` (separate Tool)                         |
| `web_search` / `web_search_preview` | `GoogleSearch` (separate Tool)                        |
| `google_search_retrieval`           | `GoogleSearchRetrieval` with dynamic retrieval config |
| `google_maps`                       | `GoogleMaps` (separate Tool)                          |
| `code_execution`                    | `ToolCodeExecution` (separate Tool)                   |
| `url_context`                       | `URLContext` (separate Tool)                          |

#### tool_choice

| OpenAI Value                                       | Vertex Behavior                        |
| -------------------------------------------------- | -------------------------------------- |
| `"none"`                                           | Tool calling disabled                  |
| `"auto"`                                           | Model decides whether to call tools    |
| `"required"`                                       | Model must call at least one tool      |
| `{"type": "function", "function": {"name": "fn"}}` | Model must call the specified function |

Example with tools:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="your-key")

response = client.chat.completions.create(
    model="gemini-2.5-flash",
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

### Reasoning / Thinking

Gemini 2.5 and Gemini 3+ models support configurable reasoning. The router supports four ways to configure it, applied in priority order:

1. `extra_body.thinking_config` — Gemini-native nested config (highest priority)
2. `extra_body.thinking_budget` / `extra_body.thinking_level` — Gemini-native top-level shorthands
3. `extra_body.thinking` — Anthropic-style format
4. `extra_body.reasoning_effort` — OpenAI format (lowest priority)

If none are specified, the router explicitly suppresses autonomous thinking for **predictable latency**. Exception: `gemini-2.5-pro` cannot disable thinking and uses dynamic budget (`-1`) by default.

#### reasoning_effort mapping

**Gemini 2.5** models use a token budget:

| `reasoning_effort` | `ThinkingBudget` | Notes                                                             |
| ------------------ | ---------------- | ----------------------------------------------------------------- |
| `minimal`          | 1,024 tokens     |                                                                   |
| `low`              | 1,024 tokens     |                                                                   |
| `medium`           | 8,192 tokens     |                                                                   |
| `high`             | 24,576 tokens    |                                                                   |
| `none` / `disable` | 0 (disabled)     | Not supported on `gemini-2.5-pro` — thinking cannot be turned off |

**Gemini 3+** models use a thinking level enum:

| `reasoning_effort` | Flash / Flash-Lite | Pro (non-flash)    |
| ------------------ | ------------------ | ------------------ |
| `minimal`          | `Minimal`          | `Low` (clamped)    |
| `low`              | `Low`              | `Low`              |
| `medium`           | `Medium`           | `High` (clamped) ¹ |
| `high`             | `High`             | `High`             |
| `none` / `disable` | `Minimal` (lowest) | `Low` (lowest)     |

¹ Gemini 3 Pro does not support `MEDIUM` — it is clamped to `HIGH`.

!!! note "gemini-2.5-pro always thinks"
`gemini-2.5-pro` does not support disabling thinking (`budget=0` is invalid).
When `reasoning_effort` is `"none"` / `"disable"`, the model uses dynamic budget (`-1`),
letting it decide the appropriate thinking depth.

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Solve this step by step..."}],
    reasoning_effort="high",
)
```

#### Via extra_body.thinking_budget / thinking_level (Gemini shorthands)

Top-level shorthands — simpler than `thinking_config`, but with the same Gemini-native semantics.
Priority is lower than `thinking_config` but higher than `thinking` and `reasoning_effort`.

```python
# Gemini 2.5 — token budget
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    extra_body={"thinking_budget": 8192},
)

# Gemini 3+ — level enum
response = client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    extra_body={"thinking_level": "high"},  # minimal | low | medium | high
)
```

Special values for `thinking_budget` (Gemini 2.5):

| Value | Flash                   | Pro                                                      |
| ----- | ----------------------- | -------------------------------------------------------- |
| `0`   | Disables thinking       | Converted to `-1` (dynamic) — budget=0 is invalid on Pro |
| `-1`  | Dynamic (model decides) | Dynamic (model decides)                                  |
| `> 0` | Fixed token budget      | Fixed token budget                                       |

#### Via extra_body.thinking_config (Gemini-native format)

Pass `ThinkingConfig` directly in Gemini's native format. This has the **highest priority**
and overrides all other thinking parameters.

For **Gemini 2.5** use `thinking_budget` (token count):

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    extra_body={
        "thinking_config": {
            "thinking_budget": 8192,
            "include_thoughts": True,
        }
    },
)
```

If you want reasoning depth without exposing thoughts, omit `include_thoughts` or set it to `False`:

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    extra_body={
        "thinking_config": {
            "thinking_budget": 8192,
            "include_thoughts": False,
        }
    },
)
```

For **Gemini 3+** use `thinking_level` (enum string):

```python
response = client.chat.completions.create(
    model="gemini-3.1-pro-preview",
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    extra_body={
        "thinking_config": {
            "thinking_level": "high",  # minimal | low | medium | high
            "include_thoughts": True,
        }
    },
)
```

`thinking_level` values for Gemini 3+:

| `thinking_level` | Flash / Flash-Lite | Pro (non-flash)    |
| ---------------- | ------------------ | ------------------ |
| `"minimal"`      | `Minimal`          | `Low` (clamped)    |
| `"low"`          | `Low`              | `Low`              |
| `"medium"`       | `Medium`           | `High` (clamped) ¹ |
| `"high"`         | `High`             | `High`             |

¹ `"minimal"` and `"medium"` are not supported on Pro variants and are automatically clamped.

#### Via extra_body.thinking (Anthropic format)

```python
response = client.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    extra_body={"thinking": {"type": "enabled", "budget_tokens": 15000}},
)
```

For Gemini 2.5, `budget_tokens` is passed directly as `ThinkingBudget`. For Gemini 3+,
`budget_tokens` is mapped to the nearest `ThinkingLevel`:

| `budget_tokens`                          | Gemini 3 Flash | Gemini 3 Pro     |
| ---------------------------------------- | -------------- | ---------------- |
| ≥ 15,000                                 | `High`         | `High`           |
| ≥ 5,000                                  | `Medium`       | `High` (clamped) |
| < 5,000                                  | `Minimal`      | `Low` (clamped)  |
| `type: "disabled"` or `budget_tokens: 0` | `Minimal`      | `Low`            |

### Content Types

The router supports multi-modal input:

| Content Type   | Format                                                                     | Example                   |
| -------------- | -------------------------------------------------------------------------- | ------------------------- |
| Text           | string or `{"type": "text"}` block                                         | Standard text messages    |
| Image (URL)    | `{"type": "image_url", "image_url": {"url": "https://..."}}`               | HTTP, HTTPS, `gs://` URLs |
| Image (inline) | `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` | Base64 encoded            |
| Audio          | `{"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}` | Base64 encoded audio      |
| Video          | `{"type": "video_url", "video_url": {"url": "https://..."}}`               | HTTP, HTTPS, `gs://` URLs |
| File           | `{"type": "file", "file": {"file_id": "gs://bucket/path"}}`                | Cloud Storage or URLs     |

Supported MIME types:

- **Images**: jpeg, png, gif, webp
- **Video**: mp4, mpeg, mov, avi, mkv, webm, flv
- **Audio**: wav, mp3, ogg, opus, aac, flac, m4a, weba
- **Documents**: pdf, txt

### Audio Output

To enable voice responses:

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Tell me a story"}],
    extra_body={"audio": {"voice": "Kore", "format": "wav"}},
)
```

This sets Vertex AI `SpeechConfig` with the specified voice name.

### Structured Output

JSON schema-based structured output is fully supported:

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "colors",
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["colors"],
            },
        },
    },
)
```

Supported schema features: `type`, `properties`, `required`, `items`, `enum`, `anyOf`, `format`, `pattern`, `minimum`/`maximum`, `minLength`/`maxLength`, `minItems`/`maxItems`, `default`, `example`, `propertyOrdering`.

### Image Generation

Gemini models with image generation capabilities can be used through the standard chat API:

```python
response = client.chat.completions.create(
    model="gemini-2.0-flash-preview-image-generation",
    messages=[{"role": "user", "content": "Generate an image of a sunset"}],
    extra_body={"generation_config": {"response_modalities": ["IMAGE"]}},
)
```

OpenAI image endpoints are also supported for Gemini image-capable models:

```python
# Text-to-image
resp = client.images.generate(
    model="gemini-2.5-flash-image-preview",
    prompt="A sunset over snowy mountains",
    size="1792x1024",
    n=1,
)

# Image edit / composition
resp = client.images.edit(
    model="gemini-2.5-flash-image-preview",
    image=[open("base.png", "rb"), open("style.png", "rb")],
    prompt="Blend these into one cinematic scene",
    size="1024x1024",
    n=1,
)
```

For Gemini-backed `images.generate` / `images.edit`, the router converts the OpenAI request to a multimodal Gemini chat request with `response_modalities=["IMAGE"]`.

- `images.generate` maps prompt and size to Gemini image config.
- `images.edit` accepts multipart image uploads and sends them as inline image parts alongside the text prompt.
- `response_format="b64_json"` is supported naturally because Gemini image responses are returned as inline image bytes and converted to `b64_json`.

The router also supports the dedicated Imagen API endpoint for image generation models.

### Streaming

SSE streaming works transparently:

```python
stream = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Usage metadata (token counts) is included in streaming chunks when available.

### Finish Reasons

Vertex AI finish reasons are mapped to OpenAI format:

| Vertex Reason | OpenAI Reason    | Notes                                                |
| ------------- | ---------------- | ---------------------------------------------------- |
| `STOP`        | `stop`           | Overridden to `tool_calls` if function calls present |
| `MAX_TOKENS`  | `length`         |                                                      |
| `SAFETY`      | `content_filter` |                                                      |
| `RECITATION`  | `content_filter` |                                                      |
| `TOOL_CALL`   | `tool_calls`     |                                                      |

### Token Counting

The router provides accurate token counting with modality breakdown:

- **Prompt tokens**: Total input tokens
- **Completion tokens**: Total output tokens (includes thinking tokens)
- **Cached tokens**: Reported separately (deducted from base cost to avoid double-charging)
- **Audio tokens**: Tracked separately for accurate billing
- **Thinking tokens**: Included in completion count, tracked in `completion_tokens_details.reasoning_tokens`
