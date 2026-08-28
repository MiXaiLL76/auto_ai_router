# Comet API

Comet API is supported as an Anthropic-compatible provider via the dedicated
`cometapi` credential type. By default, requests are converted from OpenAI
Chat Completions or Responses API format to Anthropic Messages API and sent to
`/v1/messages`. Comet also exposes an OpenAI-compatible `/v1/chat/completions`
endpoint for the same models; set `openai_proto: true` on a credential to use
that wire protocol instead (see [OpenAI Protocol Mode](#openai-protocol-mode)).
For Google/Gemini models Comet additionally exposes a Google GenAI-compatible
`/v1beta/models/{model}:generateContent` endpoint; set `google_proto: true` to
use it (see [Google Protocol Mode](#google-protocol-mode)).

## Configuration

```yaml
credentials:
  - name: "comet_anthropic"
    type: "cometapi"
    api_key: "os.environ/COMET_API_KEY"
    base_url: "https://api.cometapi.com/v1"
    rpm: 60
    tpm: -1
```

## OpenAI Protocol Mode

Set `openai_proto: true` to switch a `cometapi` credential from Comet's
Anthropic-compatible `/v1/messages` endpoint to its OpenAI-compatible
`/v1/chat/completions` endpoint. Requests and responses are then passed
through unconverted (the same code path used for `type: openai`), instead of
being translated to/from Anthropic Messages format:

```yaml
credentials:
  - name: "comet_openai"
    type: "cometapi"
    api_key: "os.environ/COMET_API_KEY"
    base_url: "https://api.cometapi.com/v1"
    openai_proto: true
    rpm: 60
    tpm: -1
```

`openai_proto` is only valid on `type: cometapi` credentials — setting it on
any other provider type fails config validation. It is also mutually exclusive
with `google_proto`.

With `openai_proto: true`:

- The upstream URL is built from `base_url` + the incoming request path
  (typically `/v1/chat/completions`), not `/v1/messages`.
- Auth is a plain `Authorization: Bearer <api_key>` header; `auth_type` and
  the `anthropic-version` header are not used.
- The [Claude Model Aliases](#claude-model-aliases) and
  [Prompt Caching](#prompt-caching) sections below (Anthropic `cache_control`
  format, the `anthropic-beta` header) do not apply — the request body is
  never translated to Anthropic's shape.
- [Error Masking](#error-masking) still applies, since it is keyed on the
  credential's identity (`cometapi`), not its wire protocol.

Use separate credentials (e.g. `comet_anthropic` and `comet_openai`) if you
want some models routed via Anthropic protocol and others via OpenAI
protocol at the same time.

## Google Protocol Mode

Set `google_proto: true` to switch a `cometapi` credential from Comet's
Anthropic-compatible `/v1/messages` endpoint to its Google GenAI-compatible
`/v1beta/models/{model}:generateContent` (and `:streamGenerateContent?alt=sse`)
endpoint. This reuses the exact same request/response path as `type: gemini`
(Google AI Studio): incoming OpenAI Chat Completions / Responses / embeddings
requests are translated to Google's `contents` / `generationConfig` shape and
the Gemini response is translated back to OpenAI format.

```yaml
credentials:
  - name: "comet_google"
    type: "cometapi"
    api_key: "os.environ/COMET_API_KEY"
    base_url: "https://api.cometapi.com"
    google_proto: true
    rpm: 60
    tpm: -1

models:
  - name: "gemini-3.1-flash"
    credential: comet_google
  - name: "gemini-3.1-flash-image-preview"
    credential: comet_google
```

Note the `base_url` has **no `/v1` suffix** — the `/v1beta/models/...` path is
appended by the router (matching the `google-genai` SDK's
`base_url="https://api.cometapi.com"`, `api_version="v1beta"`).

With `google_proto: true`:

- The upstream URL is `base_url` + `/v1beta/models/<model>:generateContent`
  (streaming: `:streamGenerateContent?alt=sse`), not `/v1/messages`.
- Auth travels as the `x-goog-api-key: <api_key>` header (the CometAPI key);
  `auth_type` and the `anthropic-version` header are not used.
- Image generation (`responseModalities`, `imageConfig`), streaming, and
  `/v1/responses` all work exactly as they do for `type: gemini`.
- The [Claude Model Aliases](#claude-model-aliases) and
  [Prompt Caching](#prompt-caching) sections below (Anthropic `cache_control`
  format, the `anthropic-beta` header) do not apply — the request body is
  never translated to Anthropic's shape.
- [Error Masking](#error-masking) still applies, since it is keyed on the
  credential's identity (`cometapi`), not its wire protocol.

`google_proto` is only valid on `type: cometapi` credentials and is mutually
exclusive with `openai_proto`. Use separate credentials (e.g. `comet_anthropic`,
`comet_openai`, `comet_google`) to route different models through different
protocols at the same time.

## Claude Model Aliases

Use public `name` values for clients, and map them to Comet model IDs with
`model` where the provider names differ:

```yaml
models:
  - name: "anthropic/claude-haiku-4.5"
    model: "claude-haiku-4-5-20251001"
    credential: comet_anthropic
  - name: "claude-haiku-4.5"
    model: "claude-haiku-4-5-20251001"
    credential: comet_anthropic
  - name: "claude-haiku-4-5-20251001"
    credential: comet_anthropic

  - name: "anthropic/claude-opus-4.1"
    model: "claude-opus-4-1-20250805"
    credential: comet_anthropic
  - name: "claude-opus-4.1"
    model: "claude-opus-4-1-20250805"
    credential: comet_anthropic
  - name: "claude-opus-4-1-20250805"
    credential: comet_anthropic

  - name: "anthropic/claude-opus-4.5"
    model: "claude-opus-4-5-20251101"
    credential: comet_anthropic
  - name: "claude-opus-4.5"
    model: "claude-opus-4-5-20251101"
    credential: comet_anthropic
  - name: "claude-opus-4-5-20251101"
    credential: comet_anthropic

  - name: "anthropic/claude-opus-4.6"
    model: "claude-opus-4-6"
    credential: comet_anthropic
  - name: "claude-opus-4-6"
    credential: comet_anthropic

  - name: "anthropic/claude-opus-4.7"
    model: "claude-opus-4-7"
    credential: comet_anthropic
  - name: "claude-opus-4-7"
    credential: comet_anthropic

  - name: "anthropic/claude-sonnet-4"
    model: "claude-sonnet-4-20250514"
    credential: comet_anthropic
  - name: "claude-sonnet-4"
    model: "claude-sonnet-4-20250514"
    credential: comet_anthropic

  - name: "anthropic/claude-sonnet-4.5"
    model: "claude-sonnet-4-5-20250929"
    credential: comet_anthropic
  - name: "claude-sonnet-4.5"
    model: "claude-sonnet-4-5-20250929"
    credential: comet_anthropic
  - name: "claude-sonnet-4-5-20250929"
    credential: comet_anthropic

  - name: "anthropic/claude-sonnet-4.6"
    model: "claude-sonnet-4-6"
    credential: comet_anthropic
  - name: "claude-sonnet-4-6"
    credential: comet_anthropic
```

## Prompt Caching

`cometapi` uses the Anthropic Messages cache-control format:

```json
{"cache_control": {"type": "ephemeral"}}
```

When session-sticky routing is active, the router can automatically inject cache
markers for stable conversation history. Cache accounting is preserved in OpenAI
responses:

| Comet/Anthropic usage field   | OpenAI-compatible response field              |
| ----------------------------- | --------------------------------------------- |
| `cache_read_input_tokens`     | `prompt_tokens_details.cached_tokens`         |
| `cache_creation_input_tokens` | `prompt_tokens_details.cache_creation_tokens` |

For `claude-sonnet-4.5`, prefer the dated Comet model
`claude-sonnet-4-5-20250929` for more stable prompt-cache behavior.

## Error Masking

Comet upstream error bodies are masked before being returned to clients or logged.
The HTTP status is preserved, but the response body is replaced with a neutral
OpenAI-compatible error object.
