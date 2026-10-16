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

The `image_generation` tool turns on image output (`responseModalities: ["IMAGE"]`), and its `size` maps to Gemini's
image config the same way `size` does on [`images.generate`](#image-generation) (`"auto"` or no size leaves it to
the model).

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
| `web_search` / `web_search_preview` | `GoogleSearch` (separate Tool, see below)             |
| `google_search`                     | Same as `web_search` (Google's own name for the tool) |
| `google_search_retrieval`           | `GoogleSearchRetrieval` with dynamic retrieval config |
| `google_maps`                       | `GoogleMaps` (separate Tool)                          |
| `code_execution`                    | `ToolCodeExecution` (separate Tool)                   |
| `url_context`                       | `URLContext` (separate Tool)                          |

#### Google Search types

Search tools accept `search_types` (on Chat Completions and the Responses API alike): `["web_search"]` (the
default), `["image_search"]`, or both. Image search (Grounding with Google Image Search) is supported by
`gemini-nano-banana-2.1` and `gemini-3.1-flash-image`. `search_types` also takes Gemini's object form
`{"webSearch": {}, "imageSearch": {}}`, where an entry set to `false` or `null` is off. An unknown type or value
is rejected with 400 instead of silently becoming a web search. All search tools of a request are merged into
one `GoogleSearch` tool.

```python
response = client.chat.completions.create(
    model="gemini-nano-banana-2.1",
    messages=[
        {
            "role": "user",
            "content": "A detailed painting of a Timareta butterfly resting on a flower",
        }
    ],
    tools=[{"type": "web_search", "search_types": ["web_search", "image_search"]}],
)
```

Every search query Google ran — web and image alike, from the grounding metadata — is reported in
`usage.server_tool_use.web_search_requests` and billed per query (`web_search_billing_unit: per_query`).
The context Google Search retrieved (`toolUsePromptTokenCount`) is billed as input, except on
`gemini-nano-banana-2.1`, whose pricing does not charge it: there it is left out of `prompt_tokens` when Google
Search is the only tool that produced it. With `url_context`, code execution, Maps or retrieval grounding in the
same response it stays in, as their context is billed. The model is recognized by Gemini's `modelVersion`, so
this holds under an alias too, and the other way round: a response whose `modelVersion` names another model is billed
as that model (the routed model name counts only when Google sends no `modelVersion`).

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

If none are specified, the router explicitly suppresses autonomous thinking for **predictable latency**. Exceptions: `gemini-2.5-pro` cannot disable thinking and uses dynamic budget (`-1`) by default; [`gemini-nano-banana-2.1`](#gemini-nano-banana-21) keeps its own default level (`MEDIUM`).

Each source is read at the top level (where the OpenAI SDKs put `extra_body` keys) first, then inside a literal
`extra_body` object; `thinking_config` is also read from `extra_body.generation_config`, as in Gemini's REST
shape. Gemini's camelCase spellings are accepted as well: `thinkingLevel`, and `thinkingConfig` with
`thinkingLevel` / `thinkingBudget` / `includeThoughts`.

Every source resolves by the same rules on every model: level and effort names are case-insensitive and may use
Gemini's enum spelling (`THINKING_LEVEL_HIGH`); `xhigh` and `max` mean `high`; `disable` means `none`. A model
that takes a level (Gemini 3) reads a budget-only config as the level of the same depth (`0` → its floor, `-1` →
the model's own default, ≥5,000 → `medium`, ≥15,000 → `high`), and a Gemini 2.5 model reads a level-only config as
that level's budget (see the tables below). An effort the router does not know leaves the depth to the model: no
level on Gemini 3, a dynamic budget (`-1`) on Gemini 2.5 — never a zero budget `gemini-2.5-pro` would reject.

A source that sets nothing — an empty `thinking_config`, a `thinking_budget` that is not a number — counts as absent:
the next source or the default applies. `none` / `disable` (and Anthropic's `{"type": "disabled"}`) on a model without
thinking, such as `gemini-2.0-flash`, sends no thinking config at all.

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
- `images.edit` accepts multipart image uploads (or JSON `image` / `images`) and sends them as inline image parts alongside the text prompt, in order.
- `response_format="b64_json"` is supported naturally because Gemini image responses are returned as inline image bytes and converted to `b64_json`.
- Both endpoints, JSON and multipart alike, also accept `aspect_ratio` / `aspectRatio`, `image_size` / `imageSize` and `image_config` / `imageConfig` (an object, or the same object as a JSON string), which override what `size` maps to. `image_size` is sent upper-case (`"2k"` → `"2K"`). A model with configurable thinking levels ([`gemini-nano-banana-2.1`](#gemini-nano-banana-21)) also takes `thinking_level` / `thinkingLevel` / `reasoning_effort`; other image models get no thinking config, as on the chat route. In a JSON body, a value of the wrong type (`"image_size": 2048`, an unparsable `image_config`) is rejected with 400 for `gemini-nano-banana-2.1` and ignored for other models; a multipart `image_config` that is not valid JSON is rejected for every model.
- Thinking tokens are part of `usage.output_tokens` and broken out as `usage.output_tokens_details.reasoning_tokens`; interim "thought" images are not returned.

#### Gemini Nano Banana 2.1

`gemini-nano-banana-2.1` has no `image` or `gemini-3` in its ID, so the router pins its capabilities by model ID
(also for versioned IDs such as `-preview` or `-001`) instead of deriving them from the name:

| Capability          | Behavior                                                                                                                                                                                                                                                                                                          |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image sizes         | `1K` (default), `2K`, `4K`. `512` is not supported: `size` never maps to it and an explicit `image_size: "512"` is rejected with 400. `image_size` is case-insensitive (`"2k"`).                                                                                                                                  |
| Aspect ratios       | `1:1`, `1:4`, `1:8`, `2:3`, `3:2`, `3:4`, `4:1`, `4:3`, `4:5`, `5:4`, `8:1`, `9:16`, `16:9`, `21:9`. Others are rejected with 400.                                                                                                                                                                                |
| `9:21`              | Listed by Vertex only, not by the Gemini API; rejected on both routes until verified live on both.                                                                                                                                                                                                                |
| Thinking            | `ThinkingLevel` `MINIMAL` / `MEDIUM` / `HIGH`, default `MEDIUM`. `low` maps to `MINIMAL`, `xhigh` to `HIGH`, `none` to `MINIMAL`; a thinking budget is never sent: `thinking_budget` / `budget_tokens` map to the level of the same depth (`0` → `MINIMAL`, `-1` → default, ≥5,000 → `MEDIUM`, ≥15,000 → `HIGH`). |
| Sampling parameters | `temperature`, `top_p`, `top_k`, `seed`, `logprobs` / `top_logprobs` are dropped from every source (top level, `extra_body`, `generation_config`); Google rejects them.                                                                                                                                           |
| Reference images    | Up to 14 images per request (references, mask and images from earlier turns together); more is rejected with 400 `too_many_images` before the upstream call.                                                                                                                                                      |
| Search              | Web and image search; the retrieved context is not billed as input. See [Google Search types](#google-search-types).                                                                                                                                                                                              |

Image output is billed by the image tokens each route reports: 1K = 1,120 and 2K = 1,680 on both routes, 4K =
2,520 on the Gemini API and 3,780 on Vertex.

The router also supports the dedicated Imagen API endpoint for image generation models.

### Embeddings

`/v1/embeddings` works with `vertex-ai` and `gemini` credentials. Text-only models (`gemini-embedding-001`, `text-embedding-*`) take the usual OpenAI input: a string or an array of strings.

Gemini Embedding 2 (`gemini-embedding-2`, `gemini-embedding-2-preview`) is multimodal. Every item of `input` gives one vector (`data[i].index` is the item's position). An item is a string, a content part in the same shape as in chat (see [Content Types](#content-types)), or an array of strings and parts that are embedded together as **one** vector:

```python
client.embeddings.create(
    model="gemini-embedding-2",
    dimensions=768,
    input=[
        "task: search result | query: dog on a beach",  # vector 0
        [  # vector 1: text + image
            "title: none | text: a dog",
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBOR..."},
            },
        ],
        {
            "type": "file",
            "file": {"file_data": "data:application/pdf;base64,JVBER..."},
        },  # vector 2
    ],
)
```

- Media goes inline (`data:` URL or base64) or as a public `https://` / `gs://` link; a link needs a MIME type, taken from the file extension or from `mime_type` on the part.
- Input the router cannot convert is rejected with 400 before any call to Google. At most 100 items per request.
- There is no `task_type`: put the task into the text, as in the example.
- On Vertex AI (use `location: global`) each vector is a separate `embedContent` call, sent in parallel. A retry on the next credential re-sends only the inputs that have no vector yet; an input Google rejects (400/413/422) while the others succeed is not retried.
- Each of these calls counts against the credential's `rpm` and `tpm`, like a request of its own, so set them to the Vertex AI quota in calls per minute. When a credential runs out partway through a request, the inputs still missing go to the next credential; when every credential has run out, the client gets `429` with `Retry-After`. One request spreads over at most `max_provider_retries + 1` credentials (see [Retry](../advanced/retry.md)).
- If the request fails after some inputs were embedded, Google has already billed those calls, and the key pays for them: the spend log row (status `failure`) carries their usage and `spend_logs_metadata.billed_partial_embeddings` (`embedded_inputs` of `inputs`).

`usage.prompt_tokens_details` splits the tokens by modality (`text_tokens`, `image_tokens`, `audio_tokens`, `video_tokens`; PDF pages count as images), and each modality is billed at its own rate (see [Model Pricing](../litellm-integration/pricing.md)). If Google returns no usage for some or all inputs, their text is estimated at ~4 characters per token (media is not counted) and a warning is logged.

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
- **Image and video tokens**: Reported separately (`prompt_tokens_details.image_tokens` / `video_tokens`), each at its own price
- **Thinking tokens**: Included in completion count, tracked in `completion_tokens_details.reasoning_tokens`
- **Generated image tokens**: Tracked in `completion_tokens_details.image_tokens` and billed at the image output rate, apart from text and thinking
- **Google Search**: Queries in `server_tool_use.web_search_requests`; search-retrieved context counts as prompt tokens except on `gemini-nano-banana-2.1` (see [Google Search types](#google-search-types))
