# Sosana.art Integration Handoff

Last reviewed: 2026-06-26

This document summarizes the implemented Sosana.art integration, the model
mapping research against VSELLM, and the remaining product decisions. It is
intended as an engineering handoff, not end-user documentation.

## Current Scope

Sosana.art is integrated as a minimal image-only provider:

- Provider type: `sosana`
- Supported public endpoints:
  - `POST /v1/images/generations`
  - `POST /v1/images/edits`
- Unsupported endpoints for `sosana` credentials:
  - Chat Completions
  - Responses API
  - Embeddings
  - Audio
  - Video
  - Slides

Unsupported endpoints return a local OpenAI-compatible 400 error and do not call
Sosana.

Main implementation files:

- `internal/config/config.go` - provider type, aliases, validation.
- `internal/converter/sosana/images.go` - pure OpenAI Images to Sosana Banana
  conversion and Sosana task to OpenAI Images response conversion.
- `internal/proxy/sosana.go` - create + poll flow, retry behavior, error
  mapping, LiteLLM spend logging.
- `internal/proxy/proxy_log.go` - unified upstream error logging and masking.
- `internal/proxy/upstream_masking.go` - proxy-chain Sosana masking helpers.
- `internal/proxy/image_response.go` - normalization for masked proxy image
  responses.
- `docs/providers/sosana.md` - short public provider doc.
- `config.yaml.example` - example Sosana credential and model.

## Provider Configuration

The provider type accepts these forms:

- `sosana`
- `sosana-art`
- `sosana_art`

Config validation requires:

- `api_key`
- `base_url`

Recommended base URL:

```yaml
credentials:
  - name: "sosana_images"
    type: "sosana"
    api_key: "os.environ/SOSANA_API_KEY"
    base_url: "https://sosana.art"
```

Production Sosana credentials should use request and server write timeouts of at
least 2 minutes. Sosana documents async image tasks as typically completing in
20-120 seconds.

## Request Flow

### Image Generation

Client request:

```http
POST /v1/images/generations
```

OpenAI-compatible JSON is converted to a Sosana Banana create request:

```json
{
  "prompt": "...",
  "model": "nano-banana",
  "aspect_ratio": "1:1"
}
```

The model sent to Sosana is the resolved real model id from `models[].model` when
configured. This allows VSELLM public model ids to map to Sosana internal model
ids without changing the client-visible model name.

### Image Edits

Client request:

```http
POST /v1/images/edits
Content-Type: multipart/form-data
```

The converter:

- reads `prompt`, `model`, `size`, and `n` fields;
- reads uploaded `image`, `images`, or `image[]` file parts;
- encodes uploaded images as `data:image/...;base64,...`;
- sends them to Sosana as `image_urls`;
- rejects `mask` locally because Sosana Banana does not support masks in the
  implemented flow.

Each multipart image part is capped at 20 MB.

### Async Sosana Flow

The proxy does not expose Sosana async tasks to clients. It performs the async
flow synchronously inside the request:

1. `POST {base_url}/api/banana/create-async`
2. Immediate first poll
3. `GET {base_url}/api/banana/{uid}` every 2 seconds
4. Stop on `COMPLETED`, `FAILED`, `MODERATED`, unknown terminal error, or
   request deadline

`request_timeout` is used as the polling deadline when positive. A timeout value
of `-1` means no extra deadline is added by the proxy.

## Current Response Behavior

On successful Sosana completion, the router downloads Sosana's
`result_file_url`, verifies that the body is an image, encodes it as base64, and
returns:

```json
{
  "created": 1782478551,
  "data": [
    {
      "b64_json": "iVBORw0KGgo...",
      "revised_prompt": "..."
    }
  ]
}
```

This matches the VSELLM public image generation examples, which decode
`data[].b64_json`. Sosana's public object URL remains an internal implementation
detail and is not returned to clients.

Important implementation details:

- The object download uses the request context and must fit into
  `request_timeout`.
- The object download does not include the Sosana `Authorization` header.
- Raw image bytes are capped at 32 MiB and are never written to logs.
- `response_format: "url"` is rejected locally with 400 until VSELLM-owned
  rehosting exists.

## Local Validation

The integration intentionally rejects unsupported OpenAI Images features before
calling Sosana:

- `n > 1` returns local 400: `sosana supports n=1 only`.
- Missing prompt returns local 400.
- Missing image on edits returns local 400.
- `mask` on edits returns local 400.
- Non-image endpoints with a `sosana` credential return local 400:
  `sosana provider supports only image generation`.
- `response_format: "url"` returns local 400 because direct Sosana storage URLs
  would reveal the upstream provider.

## Error Mapping

Sosana details must not leak to clients.

Client-facing mapping:

| Condition | HTTP status | Client body |
| --- | ---: | --- |
| Local validation error | 400 | Specific local OpenAI-compatible error |
| Unsupported endpoint | 400 | `sosana provider supports only image generation` |
| Create HTTP error | Upstream status | Generic upstream OpenAI-compatible error |
| Poll HTTP error | Upstream status | Generic upstream OpenAI-compatible error |
| Transport error | 502 | Generic upstream OpenAI-compatible error |
| Transport timeout / deadline | 408 | Generic upstream OpenAI-compatible error |
| Task `FAILED` | 502 | Generic upstream OpenAI-compatible error |
| Task `MODERATED` | 400 | Neutral `content_policy_violation` |
| Task `COMPLETED` without result URL | 502 | Generic upstream OpenAI-compatible error |
| Result image download HTTP error | 502 | Generic upstream OpenAI-compatible error |
| Result image download timeout | 408 | Generic upstream OpenAI-compatible error |
| Result image too large / non-image / read error | 502 | Generic upstream OpenAI-compatible error |
| Unknown task status | 502 | Generic upstream OpenAI-compatible error |

`FAILED` and upstream HTTP errors deliberately do not return Sosana's raw
`error`, `detail`, balance text, policy text, uid, or provider-specific codes.

## Debug Logging And Masking

Sosana is considered a masked upstream provider when any of these are true:

- credential type is `sosana`;
- credential base URL host is `sosana.art` or a subdomain;
- credential name contains `sosana` or `sasana`;
- `mask_upstream_errors: true` is set.

For masked upstream failures:

- client receives only generic/neutral OpenAI-compatible errors;
- structured logs include `response_body_masked=true`;
- structured logs also include raw `response_body` for internal debugging.

This mirrors the Comet API style: external users do not see provider internals,
while operators can still debug the actual upstream response.

For proxy-chain setups, masking is reliable when:

- the upstream router propagates `X-Credential-Name` with a Sosana marker; or
- the proxy credential has `mask_upstream_errors: true`.

Older external proxies without a credential marker remain a configuration risk.

## Retry And Fallback Behavior

Sosana retry behavior is intentionally narrower than generic direct LLM
providers because image creation can be billable once a task is accepted.

Implemented retry:

- Retry same provider type (`sosana`) with another Sosana credential.
- Retry only when the create request returns a retryable HTTP status before a
  task is accepted, for example 429 or 5xx according to the shared retry
  classifier.

No retry across credentials after task creation:

- Poll HTTP errors are not retried against another credential.
- Task `FAILED` is not retried against another credential.
- Transport errors are not retried by the Sosana special-case because it can be
  ambiguous whether the create request reached Sosana and became billable.

This avoids duplicate image charges and duplicate generated assets.

## Billing

Successful Sosana image requests set:

```go
logCtx.TokenUsage = &converter.TokenUsage{ImageCount: logCtx.ImageCount}
```

`ImageCount` is extracted from `n` and defaults to 1. Since Sosana currently
supports only `n=1`, successful spend should be one image per request.

LiteLLM spend calculation uses:

```text
spend = image_count * output_cost_per_image
```

Price lookup tries the real model name first, then the public alias. Production
pricing should include the public VSELLM model id and/or the Sosana real model
id used by `models[].model`.

If future support for `n > 1` is added by spawning multiple Sosana tasks, billing
must keep `ImageCount` equal to the number of completed/generated images.

## VSELLM Image Model Research

VSELLM public docs and pricing list these image-generation models:

- `google/gemini-3-pro-image-preview`
- `vertex_ai/imagen-4.0-fast-generate-001`
- `google/gemini-3.1-flash-image-preview`
- `vertex_ai/imagen-4.0-generate-001`
- `vertex_ai/imagen-4.0-ultra-generate-001`
- `google/gemini-2.5-flash-image`
- `openai/gpt-image-1-mini`
- `openai/gpt-image-1`

VSELLM docs for `/v1/images/edits` mention these edit-capable models:

- `openai/gpt-image-1-mini`
- `openai/gpt-image-1`
- `google/gemini-2.5-flash-image`
- `google/gemini-3-pro-image-preview`
- `google/gemini-3.1-flash-image-preview`

Sources:

- `https://vsellm.ru/docs`
- `https://vsellm.ru/provider/VSELLM`

## Sosana Model Research

Sosana Banana OpenAPI currently lists these image model ids:

- `nano-banana`
- `nano-banana-2-1k`
- `nano-banana-2-2k`
- `nano-banana-2-2k-thinking`
- `nano-banana-pro-1k`
- `nano-banana-pro-2k`
- `nano-banana-pro-4k`
- `gpt-image-2`

Sosana pricing also lists "Nano Banana 2 4K", but the OpenAPI enum does not
currently expose a confirmed `nano-banana-2-4k` id. Do not use that id until it
is live-tested or confirmed by Sosana.

Sources:

- `https://sosana.art/openapi.json`
- `https://sosana.art/`

## Recommended VSELLM To Sosana Mapping

Use `models[].model` for mapping. The `name` remains the public VSELLM model id,
and `model` is the Sosana model sent upstream.

Recommended minimal production mapping:

```yaml
models:
  - name: "google/gemini-2.5-flash-image"
    model: "nano-banana"
    credential: sosana_images

  - name: "google/gemini-3.1-flash-image-preview"
    model: "nano-banana-2-1k"
    credential: sosana_images

  - name: "google/gemini-3-pro-image-preview"
    model: "nano-banana-pro-1k"
    credential: sosana_images
```

Why these map cleanly:

- VSELLM describes `google/gemini-2.5-flash-image` as Gemini 2.5 Flash Image;
  Sosana `nano-banana` is the matching Nano Banana family.
- VSELLM describes `google/gemini-3.1-flash-image-preview` as Nano Banana 2 /
  Gemini 3.1 Flash Image Preview; Sosana `nano-banana-2-*` is the matching
  family.
- VSELLM describes `google/gemini-3-pro-image-preview` as Nano Banana Pro /
  Gemini 3 Pro Image Preview; Sosana `nano-banana-pro-*` is the matching family.

Do not map these VSELLM models to Sosana Banana:

- `vertex_ai/imagen-4.0-fast-generate-001`
- `vertex_ai/imagen-4.0-generate-001`
- `vertex_ai/imagen-4.0-ultra-generate-001`
- `openai/gpt-image-1-mini`
- `openai/gpt-image-1`

Those are different model families with different quality, pricing, latency, and
behavior expectations. Substituting Sosana Banana behind those names would be a
model mismatch.

`gpt-image-2` note:

- Sosana exposes `gpt-image-2`.
- VSELLM did not show `openai/gpt-image-2` in the reviewed public image model
  list.
- Do not expose it under an existing OpenAI model id. Add a new public model id
  only as an explicit product decision.

## 1K, 2K, 4K, And Thinking Variants

The Sosana model suffixes should be treated as explicit cost/quality tiers, not
as hidden automatic upgrades.

Recommended default behavior:

- `google/gemini-3.1-flash-image-preview` -> `nano-banana-2-1k`
- `google/gemini-3-pro-image-preview` -> `nano-banana-pro-1k`

Reasons:

- 1K is the safest default for cost and latency.
- Silent upgrades to 2K, 4K, or thinking variants can surprise users and billing.
- Routing based on prompt complexity is subjective and hard to explain.

If product needs higher tiers, prefer explicit public model aliases:

```yaml
models:
  - name: "google/gemini-3.1-flash-image-preview-2k"
    model: "nano-banana-2-2k"
    credential: sosana_images

  - name: "google/gemini-3.1-flash-image-preview-thinking"
    model: "nano-banana-2-2k-thinking"
    credential: sosana_images

  - name: "google/gemini-3-pro-image-preview-2k"
    model: "nano-banana-pro-2k"
    credential: sosana_images

  - name: "google/gemini-3-pro-image-preview-4k"
    model: "nano-banana-pro-4k"
    credential: sosana_images
```

A future `size`/`quality` based selector is possible, but it should be explicit,
tested, documented, and priced. It should not be inferred from prompt text.

## Known Gaps Before Full VSELLM Parity

1. VSELLM-owned URL responses:
   - Default and `b64_json` responses hide Sosana storage.
   - `response_format: "url"` is rejected locally.
   - If URL responses are required, re-host in VSELLM-owned storage before
     returning to clients.

2. `mask` support:
   - VSELLM edit docs mention optional masks.
   - Current Sosana converter rejects masks.
   - Do not route mask-dependent public models only to Sosana unless local 400 is
     acceptable for that model.

3. `n > 1` support:
   - Current Sosana integration supports only `n=1`.
   - VSELLM docs describe `n` as number of variants.
   - Future support should create multiple tasks and bill by completed image
     count.

4. Tier selection:
   - Current mapping is static via `models[].model`.
   - 1K/2K/4K selection by `size` or `quality` is not implemented.

## Suggested Verification

Unit/regression tests:

```bash
go test ./internal/converter/sosana ./internal/proxy ./internal/config ./internal/models ./internal/litellmdb/model_table -count=1
go test ./... -count=1
git diff --check
```

Live smoke test:

1. Start router with a Sosana credential.
2. Check `/health` shows the Sosana credential and configured model.
3. Call `/v1/images/generations` with `n=1`.
4. Call `/v1/images/generations` with `n=2`; verify local 400 and no upstream
   call.
5. Call `/v1/chat/completions` with a Sosana model; verify local 400 and no
   upstream call.
6. Start with a bad Sosana key; verify the client response is masked and logs
   contain `response_body_masked=true` plus raw `response_body` for debugging.

For live bad-key tests, mount the config by absolute path. A missing host file in
`docker run -v` can be created as a directory by Docker, causing
`config.yaml: is a directory`.
